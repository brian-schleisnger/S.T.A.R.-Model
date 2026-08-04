import traceback
from typing import Any, Dict, List, Optional, Union

import mlflow
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import linprog
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from agent.memory import DataFrameMemory
from .base import TABLE_DIMENSIONS
from .transformations import _link_tables


__all__ = [
    "run_ols_regression_tool",
    "run_forecasting_tool",
    "run_random_forest_tool",
    "run_pca_tool",
    "run_kmeans_clustering_tool",
    "run_scenario_planning_tool",
    "run_neural_network_tool",
    "run_optimization_tool",
    "run_sac_optimization_tool",
    "calculate_mutual_information_tool",
]


YEARLY_WACC = 0.1
MONTHLY_WACC = (1 + YEARLY_WACC) ** (1 / 12) - 1
max_rows = 100000
min_rows = 10
scenario_min_rows = 3
Random_state = 42
split = .2
Max_depth = 5
max_iterations = 500
rf_min_leaf_size = 3


# ─── Statistical Tools ───────────────────────────────────────────
@mlflow.trace(name="calculate_mutual_information_tool")
def calculate_mutual_information_tool(
    target_variable: str, 
    feature_variables: list, 
    TABLE_NAME: Optional[Union[str, List[str]]] = None,
    dataframe_id: Optional[str] = None,
    target_type: str = "continuous", 
    df_memory: DataFrameMemory = None 
) -> Dict[str, Any]:
    """
    Calculates Shannon Mutual Information between a target variable and multiple features.
    Automatically handles categorical variables using ordinal encoding, preventing the 
    dilution of information scores that occurs with one-hot encoding.
    """
    columns_to_fetch = [target_variable] + feature_variables

    try:
        # ── 1. Data Loading ──────────────────────────────────────────────
        if dataframe_id:
            df = df_memory.get_df(dataframe_id) if df_memory else None
            if df is None:
                return {"text": f"Error: No DataFrame found for ID '{dataframe_id}'.", "data": None}
        elif TABLE_NAME:
            df = _link_tables(TABLE_NAME, columns=columns_to_fetch, random_order=True, limit=max_rows)
        else:
            return {"text": "Error: Must provide either TABLE_NAME or dataframe_id.", "data": None}

        # ── 2. Column Validation & Cleaning ──────────────────────────────
        missing = [c for c in columns_to_fetch if c not in df.columns]
        if missing:
            return {
                "text": f"Error: The following columns were not found in the data: {missing}. "
                        f"Available columns: {df.columns.tolist()}",
                "data": None
            }

        df = df[columns_to_fetch].copy()

        # ── 3. Target Preparation ────────────────────────────────────────
        task = target_type.lower()
        if task == "continuous":
            df[target_variable] = pd.to_numeric(df[target_variable], errors="coerce")
        else:
            df[target_variable] = df[target_variable].astype(str)

        df = df.dropna(subset=[target_variable])

        # ── 4. Feature Encoding (Ordinal for MI) ─────────────────────────
        current_features = [col for col in feature_variables if col in df.columns]
        
        # Track which features are categorical so sklearn processes them correctly
        discrete_mask = []
        
        for col in current_features:
            if df[col].dtype == "object" or str(df[col].dtype) == "category":
                # Automatically map strings to arbitrary integers (e.g., 0, 1, 2)
                df[col] = df[col].astype('category').cat.codes
                # Any NaN strings become -1, which is treated as its own discrete bucket
                discrete_mask.append(True)
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                discrete_mask.append(False)

        # Drop rows where numeric features are missing
        df = df.dropna(subset=[target_variable] + current_features)

        if len(df) < min_rows:
            return {"text": "Error: Data size too small after cleaning to calculate reliable mutual information.", "data": None}

        # ── 5. Calculate Mutual Information ──────────────────────────────
        X = df[current_features]
        y = df[target_variable]

        if task == "continuous":
            mi_scores = mutual_info_regression(X, y, discrete_features=discrete_mask, random_state=Random_state)
        else:
            mi_scores = mutual_info_classif(X, y, discrete_features=discrete_mask, random_state=Random_state)

        # CONVERSION: Convert nats to bits
        mi_scores = mi_scores / np.log(2)

        # ── 6. Format Outputs ────────────────────────────────────────────
        mi_results = sorted(
            zip(current_features, mi_scores),
            key=lambda x: x[1],
            reverse=True
        )
        
        result_text = f"Mutual Information Analysis ({target_type.capitalize()} Target: '{target_variable}'):\n"
        result_text += f"Calculated using Shannon information theory on {len(df):,} observations.\n"
        
        # TEXT UPDATE: Changed 'nats' to 'bits'
        result_text += "Higher values indicate stronger dependency (measured in bits).\n\n"
        
        result_text += "Feature Information Scores:\n"
        for feat, score in mi_results:
            result_text += f"  • {feat}: {score:.4f}\n"

        results_df = pd.DataFrame(mi_results, columns=["Feature", "Mutual_Information_Score"])

        return {"text": result_text, "data": results_df}

    except Exception as e:
        return {"text": f"Mutual Information Error: {e}", "data": None}


@mlflow.trace(name="run_kmeans_clustering_tool")
def run_kmeans_clustering_tool(
    feature_variables: list, 
    TABLE_NAME: Optional[Union[str, List[str]]] = None,
    dataframe_id: Optional[str] = None,
    n_clusters: int = 3,
    df_memory: DataFrameMemory = None
) -> Dict[str, Any]:
    """
    Standardizes features and fits a K-Means model to partition data into
    n_clusters groups. Returns cluster population sizes, the top 5 defining
    standardized centroid values per cluster, and the fitted KMeans object.
    """
    try:
        if dataframe_id:
            df = df_memory.get_df(dataframe_id) if df_memory else None
            if df is None:
                return {"text": f"Error: No DataFrame found for ID '{dataframe_id}'.", "model": None}
        elif TABLE_NAME:
            df = _link_tables(TABLE_NAME, columns=feature_variables, random_order=True, limit=max_rows)
        else:
            return {"text": "Error: Must provide either TABLE_NAME or dataframe_id.", "model": None}
            
        if df.empty or len(df) < n_clusters:
            return {"text": f"Error: Not enough data points fetched to perform {n_clusters}-means clustering.", "model": None}
            
        df = df[[col for col in feature_variables if col in df.columns]]
        df = pd.get_dummies(df, columns=[col for col in df.columns if df[col].dtype == 'object'], drop_first=True)
        df = df.dropna()
        current_features = df.columns.tolist()
        
        if len(df) < n_clusters or len(current_features) < 1:
            return {"text": "Error: Data size too small after cleaning to perform clustering.", "model": None}
        
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=Random_state, n_init='auto')
        kmeans.fit(scaled_data)
        
        df['Cluster'] = kmeans.labels_
        cluster_counts = df['Cluster'].value_counts().sort_index()
        
        result_text = f"K-Means Clustering Results (n_clusters={n_clusters}):\n"
        result_text += "Cluster Population Sizes:\n"
        for cluster_id, count in cluster_counts.items():
            result_text += f"  • Cluster {cluster_id}: {count} data points\n"
            
        result_text += "\nCluster Profiles (Standardized Centroids):\n"
        
        centroids = kmeans.cluster_centers_
        for i in range(n_clusters):
            result_text += f"  Cluster {i} Defining Features (Top 5):\n"
            feat_centroids = sorted(zip(current_features, centroids[i]), key=lambda x: abs(x[1]), reverse=True)
            for feat, val in feat_centroids[:5]:
                if abs(val) > 0.15: 
                    result_text += f"    - {feat}: {val:.4f}\n"
                    
        return {"text": result_text, "model": kmeans}
        
    except Exception as e:
        return {"text": f"K-Means Error: {e}", "model": None}


@mlflow.trace(name="run_ols_regression_tool")
def run_ols_regression_tool(
    dependent_variable: str, 
    independent_variables: list,
    TABLE_NAME: Optional[Union[str, List[str]]] = None,
    dataframe_id: Optional[str] = None,
    where_clause: Optional[str] = None,  # <-- ADD THIS
    df_memory: DataFrameMemory = None
) -> dict:
    """
    Fits an OLS multiple regression model using statsmodels and returns the full
    summary table as text plus the fitted model object.
    """
    columns_to_fetch = [dependent_variable] + independent_variables
    
    try:
        if dataframe_id:
            df = df_memory.get_df(dataframe_id) if df_memory else None
            if df is None:
                return {"text": f"Error: No DataFrame found for ID '{dataframe_id}'.", "data": None}
        elif TABLE_NAME:
            # <-- PASS IT TO LINK_TABLES HERE
            df = _link_tables(TABLE_NAME, columns=columns_to_fetch, where_clause=where_clause, limit=max_rows) 
        else:
            return {"text": "Error: Must provide either TABLE_NAME or dataframe_id.", "data": None}
            
        df = df.dropna(subset=[col for col in columns_to_fetch if col in df.columns])
        
        if df.empty or len(df) <= len(independent_variables):
            return {"text": "Error: Not enough valid data points to perform regression.", "data": None}
            
        Y = pd.to_numeric(df[dependent_variable])
        X = df[independent_variables].apply(pd.to_numeric, errors='coerce')
        X = sm.add_constant(X)
        
        model = sm.OLS(Y, X).fit()
        
        # Convert regression summary stats into a pandas DataFrame for Excel export
        results_df = pd.DataFrame({
            "Coefficient": model.params,
            "Std Error": model.bse,
            "t-statistic": model.tvalues,
            "p-value": model.pvalues
        }).reset_index().rename(columns={"index": "Variable"})
        
        # Return text for the LLM, the DataFrame for Excel, and the raw model for memory
        return {"text": model.summary().as_text(), "data": results_df, "model": model}
        
    except Exception as e:
        return {"text": f"Regression Error: {e}", "data": None}
    

@mlflow.trace(name="run_pca_tool")
def run_pca_tool(
    feature_variables: list, 
    TABLE_NAME: Optional[Union[str, List[str]]] = None,
    dataframe_id: Optional[str] = None,
    n_components: int = None,
    df_memory: DataFrameMemory = None
) -> Dict[str, Any]:
    """
    Standardizes the requested feature columns and fits a PCA model to identify
    the principal components that explain the most variance. Returns per-component
    explained variance ratios and the top feature loadings (|loading| > 0.3) for
    the first two components, plus the fitted PCA object.
    """
    try:
        if dataframe_id:
            df = df_memory.get_df(dataframe_id) if df_memory else None
            if df is None:
                return {"text": f"Error: No DataFrame found for ID '{dataframe_id}'.", "model": None}
        elif TABLE_NAME:
            df = _link_tables(TABLE_NAME, columns=feature_variables, random_order=True, limit=max_rows)
        else:
            return {"text": "Error: Must provide either TABLE_NAME or dataframe_id.", "model": None}
            
        if df.empty or len(df) < min_rows:
            return {"text": f"Error: Not enough data points fetched to perform PCA (minimum {min_rows} required).", "model": None}
            
        df = df[[col for col in feature_variables if col in df.columns]]
        df = pd.get_dummies(df, columns=[col for col in df.columns if df[col].dtype == 'object'], drop_first=True)
        df = df.dropna()
        
        current_features = df.columns.tolist()
        
        if len(df) < 2 or len(current_features) < 1:
            return {"text": "Error: Data size too small after cleaning to perform PCA.", "model": None}
        
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df)
        
        max_components = min(len(df), len(current_features))
        actual_components = max_components if n_components is None or n_components > max_components else n_components
            
        pca = PCA(n_components=actual_components)
        pca.fit(scaled_data)
        
        result_text = f"PCA Results (n_components={actual_components}):\n"
        explained_variance = pca.explained_variance_ratio_
        
        result_text += "Explained Variance Ratio per Component:\n"
        for i, var in enumerate(explained_variance):
            result_text += f"  • PC{i+1}: {var:.4f} ({(var*100):.1f}%)\n"
        result_text += f"Total Explained Variance: {sum(explained_variance):.4f} ({(sum(explained_variance)*100):.1f}%)\n\n"
        
        components_to_show = min(2, actual_components)
        result_text += "Top Feature Loadings (absolute magnitude > 0.3):\n"
        
        for i in range(components_to_show):
            result_text += f"  PC{i+1} Signficant Loadings:\n"
            loadings = pca.components_[i]
            feat_loadings = sorted(zip(current_features, loadings), key=lambda x: abs(x[1]), reverse=True)
            for feat, load in feat_loadings:
                if abs(load) > 0.3:
                    result_text += f"    - {feat}: {load:.4f}\n"
        
        return {"text": result_text, "model": pca}
        
    except Exception as e:
        return {"text": f"PCA Error: {e}", "model": None}
    

# ─── Machine Learning Tools ───────────────────────────────────────────
@mlflow.trace(name="run_neural_network_tool")
def run_neural_network_tool(
    target_variable: str,
    feature_variables: List[str],
    task_type: str,
    TABLE_NAME: Optional[Union[str, List[str]]] = None,
    dataframe_id: Optional[str] = None,
    hidden_layer_sizes: List[int] = [100, 50],
    max_iter: int = max_iterations,
    predict_on: Optional[Dict[str, Any]] = None,
    df_memory: DataFrameMemory = None
) -> Dict[str, Any]:
    """
    Trains a scikit-learn MLPRegressor or MLPClassifier on the provided features.
    Features are ordinal-encoded for categoricals and StandardScaler-normalized before
    training. Uses early stopping to avoid overfitting on small datasets.

    Returns:
      • Train + test performance metrics (R²/RMSE/MAE for regression; accuracy +
        per-class report for classification)
      • Training loss curve chart (convergence visualization)
      • Actual vs. Predicted scatter + Residual plot (regression)
      • Confusion matrix heatmap (classification)
      • Permutation feature importance chart (model-agnostic, works for both task types)
      • Architecture feedback: overfitting / convergence / low-signal warnings
      • Optional single-row prediction when predict_on dict is supplied
    """
    try:
        # ── 1. Data Loading ──────────────────────────────────────────────────
        if dataframe_id:
            df = df_memory.get_df(dataframe_id) if df_memory else None
            if df is None:
                return {"text": f"Error: No DataFrame found for ID '{dataframe_id}'.", "data": None, "model": None, "figures": []}
        elif TABLE_NAME:
            columns_to_fetch = [target_variable] + feature_variables
            df = _link_tables(TABLE_NAME, columns=columns_to_fetch, random_order=True, limit=max_rows)
        else:
            return {"text": "Error: Must provide either TABLE_NAME or dataframe_id.", "data": None, "model": None, "figures": []}

        # ── 2. Column Validation ─────────────────────────────────────────────
        all_cols = [target_variable] + feature_variables
        missing = [c for c in all_cols if c not in df.columns]
        if missing:
            return {
                "text": f"Error: Columns not found in data: {missing}. Available: {df.columns.tolist()}",
                "data": None, "model": None, "figures": []
            }

        df_clean = df[all_cols].copy()

        # ── 3. Target Preparation ────────────────────────────────────────────
        task = task_type.lower()
        if task == "regression":
            df_clean[target_variable] = pd.to_numeric(df_clean[target_variable], errors="coerce")
        else:
            df_clean[target_variable] = df_clean[target_variable].astype(str)

        df_clean = df_clean.dropna(subset=[target_variable])

        if len(df_clean) < min_rows:
            return {"text": f"Error: Not enough valid rows after cleaning (minimum {min_rows} required).", "data": None, "model": None, "figures": []}

        # ── 4. Feature Encoding (Ordinal for categoricals — same as RF) ──────
        categorical_features = [c for c in feature_variables if c in df_clean.columns and df_clean[c].dtype == "object"]
        numeric_features     = [c for c in feature_variables if c in df_clean.columns and c not in categorical_features]

        for col in numeric_features:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

        encoder = None
        if categorical_features:
            encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            df_clean[categorical_features] = encoder.fit_transform(df_clean[categorical_features].astype(str))

        df_clean = df_clean.dropna(subset=[target_variable] + feature_variables)

        X = df_clean[feature_variables]
        y = df_clean[target_variable]

        # ── 5. Train / Test Split ────────────────────────────────────────────
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=split, random_state=Random_state
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled  = scaler.transform(X_test)

        # ── 6. Model Fitting ─────────────────────────────────────────────────
        arch = tuple(hidden_layer_sizes)
        if task == "regression":
            model = MLPRegressor(
                hidden_layer_sizes=arch, max_iter=max_iter,
                early_stopping=True, validation_fraction=0.1,
                random_state=Random_state
            )
            model.fit(X_train_scaled, y_train)

            train_preds = model.predict(X_train_scaled)
            test_preds  = model.predict(X_test_scaled)

            train_r2   = r2_score(y_train, train_preds)
            test_r2    = r2_score(y_test,  test_preds)
            test_rmse  = mean_squared_error(y_test, test_preds) ** 0.5
            test_mae   = mean_absolute_error(y_test, test_preds)

        else:
            model = MLPClassifier(
                hidden_layer_sizes=arch, max_iter=max_iter,
                early_stopping=True, validation_fraction=0.1,
                random_state=Random_state
            )
            model.fit(X_train_scaled, y_train)

            train_preds = model.predict(X_train_scaled)
            test_preds  = model.predict(X_test_scaled)

            train_acc = accuracy_score(y_train, train_preds)
            test_acc  = accuracy_score(y_test,  test_preds)

        # ── 7. Result Text: Metrics ──────────────────────────────────────────
        arch_str = " → ".join(str(n) for n in hidden_layer_sizes)
        result_text  = f"Neural Network ({task_type.title()}) Results\n"
        result_text += f"{'=' * 50}\n"
        result_text += f"Architecture:  input({len(feature_variables)}) → {arch_str} → output\n"
        result_text += f"Training rows: {len(X_train):,}   Test rows: {len(X_test):,}\n"
        result_text += f"Iterations:    {model.n_iter_} / {max_iter}"
        result_text += " (converged)\n" if model.n_iter_ < max_iter else " ⚠ max_iter reached — may not have converged\n"
        result_text += "\n"

        if task == "regression":
            result_text += f"Performance:\n"
            result_text += f"  • Train R²:  {train_r2:.4f}\n"
            result_text += f"  • Test  R²:  {test_r2:.4f}\n"
            result_text += f"  • Test RMSE: {test_rmse:.4f}\n"
            result_text += f"  • Test MAE:  {test_mae:.4f}\n"
        else:
            result_text += f"Performance:\n"
            result_text += f"  • Train Accuracy: {train_acc:.4f}\n"
            result_text += f"  • Test  Accuracy: {test_acc:.4f}\n"
            result_text += f"\nClassification Report (test set):\n{classification_report(y_test, test_preds)}\n"

        # ── 8. Architecture Feedback (Improvement 5) ─────────────────────────
        feedback = []
        if task == "regression":
            overfit_gap = train_r2 - test_r2
            if overfit_gap > 0.15:
                feedback.append(
                    f"⚠ Overfitting detected (train R²={train_r2:.3f} vs test R²={test_r2:.3f}, "
                    f"gap={overfit_gap:.3f}). Consider using fewer neurons, adding more data, "
                    f"or trying Random Forest which is more robust on small datasets."
                )
            if test_r2 < 0.3:
                feedback.append(
                    f"⚠ Low predictive signal (test R²={test_r2:.3f}). The neural network may not "
                    f"be finding a meaningful pattern. Try run_ols_regression_tool or "
                    f"run_random_forest_tool first to confirm whether any linear/non-linear "
                    f"relationship exists."
                )
        else:
            overfit_gap = train_acc - test_acc
            if overfit_gap > 0.1:
                feedback.append(
                    f"⚠ Overfitting detected (train acc={train_acc:.3f} vs test acc={test_acc:.3f}, "
                    f"gap={overfit_gap:.3f}). Consider fewer neurons or more training data."
                )
            if test_acc < 0.5:
                feedback.append(
                    f"⚠ Test accuracy ({test_acc:.3f}) is near chance level. The model may not "
                    f"have found a useful classification signal in these features."
                )
        if model.n_iter_ >= max_iter:
            feedback.append(
                f"⚠ Training stopped at max_iter={max_iter} without confirmed convergence. "
                f"The model may improve with a higher max_iter value."
            )
        if feedback:
            result_text += "\nDiagnostic Feedback:\n"
            for f_line in feedback:
                result_text += f"  {f_line}\n"

        # ── 9. Figures ───────────────────────────────────────────────────────
        figures = []

        # ── 9a. Training Loss Curve (Improvement 3) ──────────────────────────
        try:
            loss_curve = model.loss_curve_
            val_curve  = getattr(model, "validation_scores_", None)

            loss_df = pd.DataFrame({"Epoch": range(1, len(loss_curve) + 1), "Training Loss": loss_curve})
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(
                x=loss_df["Epoch"], y=loss_df["Training Loss"],
                mode="lines", name="Training Loss", line=dict(color="#1f77b4", width=2)
            ))
            if val_curve is not None:
                fig_loss.add_trace(go.Scatter(
                    x=list(range(1, len(val_curve) + 1)), y=val_curve,
                    mode="lines", name="Validation Score",
                    line=dict(color="#ff7f0e", width=2, dash="dash")
                ))
            fig_loss.update_layout(
                title=f"Training Loss Curve — {target_variable}",
                xaxis_title="Epoch", yaxis_title="Loss",
                template="plotly_white", margin=dict(l=40, r=40, t=60, b=40),
                legend=dict(x=0.7, y=0.95)
            )
            figures.append(fig_loss)
        except Exception:
            pass  # best-effort

        # ── 9b. Regression: Actual vs. Predicted + Residuals (Improvement 2) ─
        if task == "regression":
            try:
                residuals = np.array(y_test) - test_preds
                # Actual vs Predicted
                fig_avp = go.Figure()
                fig_avp.add_trace(go.Scatter(
                    x=list(y_test), y=list(test_preds),
                    mode="markers",
                    marker=dict(color="#1f77b4", opacity=0.6, size=5),
                    name="Predictions",
                    hovertemplate="Actual: %{x:.3f}<br>Predicted: %{y:.3f}<extra></extra>"
                ))
                ref_min = float(min(min(y_test), min(test_preds)))
                ref_max = float(max(max(y_test), max(test_preds)))
                fig_avp.add_trace(go.Scatter(
                    x=[ref_min, ref_max], y=[ref_min, ref_max],
                    mode="lines", line=dict(color="gray", dash="dash"),
                    name="Perfect Prediction"
                ))
                fig_avp.update_layout(
                    title=f"Actual vs. Predicted — {target_variable}",
                    xaxis_title=f"Actual {target_variable}",
                    yaxis_title=f"Predicted {target_variable}",
                    template="plotly_white", margin=dict(l=40, r=40, t=60, b=40)
                )
                figures.append(fig_avp)

                # Residual plot
                fig_resid = go.Figure()
                fig_resid.add_trace(go.Scatter(
                    x=list(test_preds), y=list(residuals),
                    mode="markers",
                    marker=dict(color="#d62728", opacity=0.6, size=5),
                    name="Residuals",
                    hovertemplate="Predicted: %{x:.3f}<br>Residual: %{y:.3f}<extra></extra>"
                ))
                fig_resid.add_hline(y=0, line_dash="dash", line_color="gray")
                fig_resid.update_layout(
                    title=f"Residual Plot — {target_variable}",
                    xaxis_title=f"Predicted {target_variable}",
                    yaxis_title="Residual (Actual − Predicted)",
                    template="plotly_white", margin=dict(l=40, r=40, t=60, b=40)
                )
                figures.append(fig_resid)
            except Exception:
                pass

        # ── 9c. Classification: Confusion Matrix heatmap (Improvement 1) ─────
        if task == "classification":
            try:
                classes   = model.classes_
                cm        = confusion_matrix(y_test, test_preds, labels=classes)
                class_strs = [str(c) for c in classes]
                fig_cm = px.imshow(
                    cm,
                    x=class_strs, y=class_strs,
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    title=f"Confusion Matrix — {target_variable}",
                    text_auto=True,
                    color_continuous_scale="Blues",
                    template="plotly_white"
                )
                fig_cm.update_layout(margin=dict(l=40, r=40, t=60, b=40))
                figures.append(fig_cm)
            except Exception:
                pass

        # ── 9d. Permutation Feature Importance (Improvement 4) ───────────────
        perm_imp_df = None
        try:
            scoring = "r2" if task == "regression" else "accuracy"
            perm_result = permutation_importance(
                model, X_test_scaled, y_test,
                n_repeats=10, random_state=Random_state, scoring=scoring
            )
            perm_means  = perm_result.importances_mean
            perm_stds   = perm_result.importances_std

            perm_imp_df = pd.DataFrame({
                "Feature":           feature_variables,
                "Importance_Mean":   perm_means,
                "Importance_Std":    perm_stds,
                "Importance_%":      perm_means / (perm_means.sum() + 1e-9) * 100
            }).sort_values("Importance_Mean", ascending=False).reset_index(drop=True)

            result_text += f"\nPermutation Feature Importance (Top {min(10, len(perm_imp_df))}):\n"
            for _, row in perm_imp_df.head(10).iterrows():
                result_text += (
                    f"  • {row['Feature']}: {row['Importance_Mean']:.4f} "
                    f"(±{row['Importance_Std']:.4f})\n"
                )
            result_text += (
                "  Interpretation: shuffling a feature with high importance causes a large "
                f"drop in test {scoring.upper()}, meaning the model genuinely relies on it.\n"
                "  Features with near-zero importance add noise and could be removed.\n"
            )

            # Chart: horizontal bar, sorted ascending so largest is at top
            top_perm = perm_imp_df.head(15).sort_values("Importance_Mean", ascending=True)
            fig_perm = go.Figure()
            fig_perm.add_trace(go.Bar(
                x=top_perm["Importance_Mean"],
                y=top_perm["Feature"],
                orientation="h",
                error_x=dict(type="data", array=top_perm["Importance_Std"].tolist(), visible=True),
                marker=dict(
                    color=top_perm["Importance_Mean"],
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="Importance", thickness=12)
                ),
                hovertemplate="%{y}: %{x:.4f} ± %{error_x.array:.4f}<extra></extra>"
            ))
            fig_perm.update_layout(
                title=f"Permutation Feature Importance — {target_variable}",
                xaxis_title=f"Mean Decrease in {scoring.upper()} when Feature is Shuffled",
                yaxis_title="Feature",
                template="plotly_white",
                margin=dict(l=40, r=40, t=60, b=40),
                height=max(300, 35 * len(top_perm) + 80)
            )
            figures.append(fig_perm)
        except Exception as perm_err:
            result_text += f"\n(Permutation importance unavailable: {perm_err})\n"

        # ── 10. Predict on New Inputs (Improvement 6) ────────────────────────
        prediction_text = ""
        if predict_on is not None:
            try:
                input_row = {}
                for col in feature_variables:
                    if col not in predict_on:
                        return {
                            "text": (
                                f"Error: predict_on is missing a value for feature '{col}'. "
                                f"Required features: {feature_variables}"
                            ),
                            "data": perm_imp_df, "model": model, "figures": figures
                        }
                    input_row[col] = predict_on[col]

                input_df = pd.DataFrame([input_row])

                # Apply the same ordinal encoding the training set used
                if encoder is not None and categorical_features:
                    input_df[categorical_features] = encoder.transform(
                        input_df[categorical_features].astype(str)
                    )
                for col in numeric_features:
                    input_df[col] = pd.to_numeric(input_df[col], errors="coerce")

                input_scaled = scaler.transform(input_df[feature_variables])
                pred_val     = model.predict(input_scaled)[0]

                prediction_text = f"\nPrediction for Supplied Inputs:\n"
                for k, v in predict_on.items():
                    prediction_text += f"  • {k}: {v}\n"
                prediction_text += f"  → Predicted {target_variable}: {pred_val}"
                if task == "regression":
                    prediction_text += f" (model test RMSE: ±{test_rmse:.4f})\n"
                else:
                    # Classification: also show probability if available
                    prediction_text += f"\n"
                    try:
                        proba = model.predict_proba(input_scaled)[0]
                        for cls, p in zip(model.classes_, proba):
                            prediction_text += f"    P({cls}): {p:.3f}\n"
                    except Exception:
                        pass

                result_text += prediction_text
            except Exception as pred_err:
                result_text += f"\n(Prediction on new inputs failed: {pred_err})\n"

        # ── 11. Export DataFrame ─────────────────────────────────────────────
        # Prefer permutation importance for Excel; fall back to empty summary
        export_df = perm_imp_df if perm_imp_df is not None else pd.DataFrame({
            "Metric": ["Train Score", "Test Score"],
            "Value": [
                train_r2 if task == "regression" else train_acc,
                test_r2  if task == "regression" else test_acc
            ]
        })

        return {
            "text":    result_text,
            "data":    export_df,
            "figure":  figures[0] if figures else None,   # backward compat
            "figures": figures,
            "model":   model,
        }

    except Exception as e:
        return {
            "text":    f"Neural Network Error: {e}\n{traceback.format_exc()}",
            "data":    None, "figure": None, "figures": [], "model": None
        }


@mlflow.trace(name="run_optimization_tool")
def run_optimization_tool(
    objective_coefficients: List[float],
    inequality_constraints_matrix: Optional[List[List[float]]] = None,
    inequality_constraints_bounds: Optional[List[float]] = None,
    equality_constraints_matrix: Optional[List[List[float]]] = None,
    equality_constraints_bounds: Optional[List[float]] = None,
    bounds: Optional[List[List[Optional[float]]]] = None,
    variable_names: Optional[List[str]] = None,
    maximize: bool = False,
) -> Dict[str, Any]:
    """
    Solves a linear programming problem using scipy.optimize.linprog (HiGHS solver).
    Minimizes (or maximizes) c·x subject to A_ub·x ≤ b_ub, A_eq·x = b_eq, and
    per-variable bounds.  Set maximize=True instead of manually negating coefficients.
    Optionally supply variable_names so output labels are human-readable rather than
    'Variable_1', 'Variable_2', etc.
    Returns the optimal objective value and decision variable values on success,
    or a structured failure message on infeasibility.
    """
    try:
        n_vars = len(objective_coefficients)

        # ── Pre-solve feasibility checks ───────────────────────────────────
        # 1. Mismatched variable_names length
        if variable_names and len(variable_names) != n_vars:
            return {
                "text": (
                    f"Error: variable_names has {len(variable_names)} entries but "
                    f"objective_coefficients has {n_vars}. They must be the same length."
                ),
                "data": None, "model": None,
            }

        # 2. All-zero objective — every variable will be pushed to its bound
        if all(c == 0 for c in objective_coefficients):
            return {
                "text": (
                    "Warning: All objective coefficients are zero. "
                    "The solver has no preference between variables, so results will be "
                    "arbitrary (each variable will be pushed to its bound). "
                    "Please supply non-zero coefficients representing the value of each variable."
                ),
                "data": None, "model": None,
            }

        # 3. Per-variable lower-bound sum vs. total-budget inequality constraint
        if bounds is not None and inequality_constraints_bounds is not None:
            lower_bounds = [b[0] if b and b[0] is not None else 0.0 for b in bounds]
            lb_sum = sum(lower_bounds)
            # If any single-row ≤ constraint (budget ceiling) is tighter than floor sum
            for i, rhs in enumerate(inequality_constraints_bounds):
                if lb_sum > rhs and inequality_constraints_matrix is not None:
                    # Only flag if the row's coefficients are all +1 (looks like a budget constraint)
                    row = inequality_constraints_matrix[i]
                    if all(c > 0 for c in row):
                        names = variable_names or [f"Variable_{j+1}" for j in range(n_vars)]
                        floor_lines = ", ".join(
                            f"{names[j]}: {lower_bounds[j]:,.0f}" for j in range(n_vars) if lower_bounds[j] > 0
                        )
                        return {
                            "text": (
                                f"Infeasibility Detected (before solving): "
                                f"Your minimum spend floors sum to {lb_sum:,.0f} but the budget "
                                f"ceiling for constraint {i+1} is {rhs:,.0f}. "
                                f"The problem has no feasible solution as specified.\n"
                                f"Floors: {floor_lines}\n"
                                f"Suggestion: Reduce one or more minimum floors, or increase the budget."
                            ),
                            "data": None, "model": None,
                        }

        # ── Coefficient sign flip for maximization ──────────────────────────
        c = [-x for x in objective_coefficients] if maximize else list(objective_coefficients)

        # ── Solve ───────────────────────────────────────────────────────────
        formatted_bounds = None
        if bounds is not None:
            formatted_bounds = [(b[0], b[1]) if len(b) >= 2 else (None, None) for b in bounds]

        res = linprog(
            c=c,
            A_ub=inequality_constraints_matrix,
            b_ub=inequality_constraints_bounds,
            A_eq=equality_constraints_matrix,
            b_eq=equality_constraints_bounds,
            bounds=formatted_bounds,
            method='highs'
        )

        if not res.success:
            return {
                "text": (
                    f"Optimization could not find a solution.\n"
                    f"Solver message: {res.message}\n\n"
                    f"Common causes:\n"
                    f"  • Minimum spend floors (variable lower bounds) exceed the total budget\n"
                    f"  • An equality constraint conflicts with an inequality constraint\n"
                    f"  • A variable has no feasible range (lower bound > upper bound)\n"
                    f"Try relaxing the tightest constraint and re-running."
                ),
                "data": None, "model": None,
            }

        # ── Human-readable output ───────────────────────────────────────────
        names = variable_names if variable_names else [f"Variable_{i+1}" for i in range(n_vars)]
        # The true objective value: un-flip the sign when maximizing
        reported_obj = -res.fun if maximize else res.fun
        direction_label = "Maximized" if maximize else "Minimized"

        contributions = np.array(objective_coefficients) * res.x

        result_text = f"Optimization Successful — {direction_label} Objective Value: {reported_obj:,.4f}\n\n"
        result_text += "Optimal Allocation:\n"
        for name, val, contrib in zip(names, res.x, contributions):
            result_text += f"  • {name}: {val:,.4f}  (contribution: {contrib:,.4f})\n"
        result_text += f"\nTotal of all variables: {res.x.sum():,.4f}\n"

        # ── Sensitivity / shadow prices (HiGHS exposes these) ──────────────
        sensitivity_lines = []
        try:
            if hasattr(res, "ineqlin") and res.ineqlin is not None:
                marginals = np.atleast_1d(res.ineqlin.marginals)
                for idx, shadow in enumerate(marginals):
                    # Flip sign back to match maximization framing
                    display_shadow = -shadow if maximize else shadow
                    if abs(display_shadow) > 1e-6:
                        sensitivity_lines.append(
                            f"  • Inequality constraint {idx + 1}: each additional 1-unit "
                            f"relaxation {'increases' if display_shadow > 0 else 'decreases'} "
                            f"the objective by {abs(display_shadow):,.4f}"
                        )
            if hasattr(res, "eqlin") and res.eqlin is not None:
                marginals = np.atleast_1d(res.eqlin.marginals)
                for idx, shadow in enumerate(marginals):
                    display_shadow = -shadow if maximize else shadow
                    if abs(display_shadow) > 1e-6:
                        sensitivity_lines.append(
                            f"  • Equality constraint {idx + 1}: shadow price = {display_shadow:,.4f}"
                        )
            if hasattr(res, "lower") and res.lower is not None:
                rc = np.atleast_1d(res.lower.marginals)
                for idx, reduced in enumerate(rc):
                    display_rc = -reduced if maximize else reduced
                    if abs(display_rc) > 1e-6:
                        sensitivity_lines.append(
                            f"  • {names[idx]} is at its lower bound — "
                            f"relaxing it by 1 unit {'gains' if display_rc < 0 else 'costs'} "
                            f"{abs(display_rc):,.4f} in the objective"
                        )
        except Exception:
            pass  # Sensitivity is best-effort; never block the primary result

        if sensitivity_lines:
            result_text += "\nSensitivity Analysis (binding constraints):\n"
            result_text += "\n".join(sensitivity_lines) + "\n"

        # ── Plain-English interpretation ────────────────────────────────────
        sorted_alloc = sorted(zip(names, res.x), key=lambda x: x[1], reverse=True)
        top = sorted_alloc[0]
        result_text += (
            f"\nKey Insight: The largest allocation goes to '{top[0]}' ({top[1]:,.4f}). "
        )
        if len(sorted_alloc) > 1:
            bottom = sorted_alloc[-1]
            result_text += f"The smallest allocation is '{bottom[0]}' ({bottom[1]:,.4f})."

        # ── DataFrame for Excel export ──────────────────────────────────────
        df_results = pd.DataFrame({
            "Variable": names,
            "Objective_Coefficient": objective_coefficients,
            "Optimal_Value": res.x,
            "Total_Contribution": contributions,
        })
        summary_row = pd.DataFrame([{
            "Variable": f"TOTAL / {direction_label.upper()} OBJECTIVE",
            "Objective_Coefficient": None,
            "Optimal_Value": res.x.sum(),
            "Total_Contribution": reported_obj,
        }])
        df_results = pd.concat([df_results, summary_row], ignore_index=True)

        return {"text": result_text, "data": df_results, "model": res}

    except Exception as e:
        return {"text": f"Optimization Error: {str(e)}", "data": None, "model": None}


@mlflow.trace(name="run_sac_optimization_tool")
def run_sac_optimization_tool(
    total_budget: float,
    objective: str = "npv",
    tactic_filters: Optional[List[str]] = None,
    min_spend_by_tactic: Optional[Dict[str, float]] = None,
    max_spend_by_tactic: Optional[Dict[str, float]] = None,
    target_activations: Optional[float] = None,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    df_memory: DataFrameMemory = None,
) -> Dict[str, Any]:
    """
    Data-aware SAC optimizer.  Pulls historical data from acquisition_data_v3 and
    dbs_marketing_sync, computes per-tactic efficiency metrics (average NPV per
    activation and average marketing spend per activation), then runs a linear
    program to find the spend allocation across marketing tactics that maximises
    total projected NPV (or minimises total SAC) subject to:
      - a total budget ceiling
      - optional per-tactic minimum and maximum spend floors/ceilings
      - an optional target activation count (equality constraint)

    Returns a plain-English allocation recommendation, projected outcomes, and
    full sensitivity analysis on binding constraints — all labelled with real
    tactic names rather than 'Variable_1' etc.
    """
    ACQ_TABLE  = '"sandbox"."acquisition_data_v3"'
    MKT_TABLE  = '"sandbox"."dbs_marketing_sync"'

    try:
        # ── 1. Build date filter ────────────────────────────────────────────
        year_clauses = []
        if start_year:
            year_clauses.append(f'"Activation_Year" >= {int(start_year)}')
        if end_year:
            year_clauses.append(f'"Activation_Year" <= {int(end_year)}')
        acq_where = " AND ".join(year_clauses) if year_clauses else None

        # ── 2. Pull per-tactic acquisition economics from acquisition table ─
        #    We need: tactic, count of activations, avg NPV per activation, avg SAC
        tactic_filter_sql = ""
        if tactic_filters:
            quoted = ", ".join(f"'{t}'" for t in tactic_filters)
            tactic_filter_sql = f' AND "Tactic" IN ({quoted})'

        acq_where_full = (
            f'({acq_where}){tactic_filter_sql}' if acq_where else
            tactic_filter_sql.lstrip(" AND ") if tactic_filter_sql else None
        )

        acq_query = f"""
            SELECT
                "Tactic",
                COUNT(*)                    AS activations,
                AVG("npv")                  AS avg_npv,
                AVG("sac")                  AS avg_sac,
                AVG("Marketing")            AS avg_marketing_cost
            FROM {ACQ_TABLE}
            WHERE "Tactic" IS NOT NULL
              AND "Tactic" NOT IN ('unknown', 'Unknown', 'other', 'Other')
              AND "npv" IS NOT NULL
              AND "sac" IS NOT NULL
              AND "Marketing" IS NOT NULL
              AND "Marketing" < 0
              {"AND " + acq_where_full if acq_where_full else ""}
            GROUP BY "Tactic"
            HAVING COUNT(*) >= 50
            ORDER BY AVG("npv") DESC
        """

        from .base import run_sql_query
        econ_df = run_sql_query(acq_query)

        if econ_df.empty:
            return {
                "text": (
                    "SAC Optimizer Error: No qualifying tactic data found in acquisition_data_v3. "
                    "Check your tactic_filters or date range — each tactic needs at least 50 "
                    "historical activations to produce reliable estimates."
                ),
                "data": None, "model": None,
            }

        # ── 3. Convert avg_marketing_cost (negative cost) to positive spend ─
        #    avg_marketing_cost is stored as a negative number (cost convention).
        #    We want: cost_per_activation = |avg_marketing_cost|
        econ_df["avg_marketing_cost"] = pd.to_numeric(econ_df["avg_marketing_cost"], errors="coerce")
        econ_df["avg_npv"]            = pd.to_numeric(econ_df["avg_npv"],            errors="coerce")
        econ_df["avg_sac"]            = pd.to_numeric(econ_df["avg_sac"],            errors="coerce")
        econ_df["activations"]        = pd.to_numeric(econ_df["activations"],        errors="coerce")
        econ_df = econ_df.dropna(subset=["avg_marketing_cost", "avg_npv", "activations"])

        # cost_per_activation: positive dollars spent per acquired subscriber via marketing
        econ_df["cost_per_activation"] = econ_df["avg_marketing_cost"].abs()

        # Remove tactics where cost_per_activation is zero (can't build a sensible LP)
        econ_df = econ_df[econ_df["cost_per_activation"] > 0].reset_index(drop=True)

        if econ_df.empty:
            return {
                "text": (
                    "SAC Optimizer Error: All qualifying tactics have zero average marketing cost, "
                    "which makes it impossible to define a spend-to-activation conversion rate. "
                    "Ensure the Marketing column contains non-zero values for the selected tactics."
                ),
                "data": None, "model": None,
            }

        tactics     = econ_df["Tactic"].tolist()
        n           = len(tactics)

        # ── 4. Objective vector ─────────────────────────────────────────────
        # Decision variable x[i] = dollars allocated to tactic i
        # Activations from tactic i  = x[i] / cost_per_activation[i]
        # NPV from tactic i          = activations[i] * avg_npv[i]
        #                            = x[i] * (avg_npv[i] / cost_per_activation[i])
        # SAC from tactic i          = x[i] * (avg_sac[i]  / cost_per_activation[i])   (negative)

        npv_per_dollar = (
            econ_df["avg_npv"] / econ_df["cost_per_activation"]
        ).values  # positive → maximise

        sac_per_dollar = (
            econ_df["avg_sac"].abs() / econ_df["cost_per_activation"]
        ).values  # SAC magnitude per dollar; minimise means lower total SAC cost

        act_per_dollar = (1.0 / econ_df["cost_per_activation"]).values

        obj_label = objective.lower()
        if obj_label == "sac":
            # Minimise total SAC — linprog minimises, SAC is already positive magnitude
            c_vec = sac_per_dollar.tolist()
            maximize_flag = False
            obj_description = "Minimise Total Subscriber Acquisition Cost (SAC)"
        else:
            # Default: maximise NPV — linprog minimises, so negate
            c_vec = (-npv_per_dollar).tolist()
            maximize_flag = True
            obj_description = "Maximise Total Projected NPV"

        # ── 5. Constraints ──────────────────────────────────────────────────
        A_ub: List[List[float]] = []
        b_ub: List[float]       = []

        # 5a. Total budget ceiling: sum(x[i]) <= total_budget
        A_ub.append([1.0] * n)
        b_ub.append(float(total_budget))

        # 5b. Per-tactic max_spend ceilings
        max_spend = max_spend_by_tactic or {}
        for i, tactic in enumerate(tactics):
            cap = max_spend.get(tactic)
            if cap is not None:
                row = [0.0] * n
                row[i] = 1.0
                A_ub.append(row)
                b_ub.append(float(cap))

        # 5c. Target activations equality (optional)
        A_eq: Optional[List[List[float]]] = None
        b_eq: Optional[List[float]]       = None
        if target_activations is not None:
            # sum(x[i] * act_per_dollar[i]) = target_activations
            A_eq = [act_per_dollar.tolist()]
            b_eq = [float(target_activations)]

        # 5d. Per-variable bounds: [min_spend, max_spend or None]
        min_spend = min_spend_by_tactic or {}
        var_bounds: List[List[Optional[float]]] = []
        for tactic in tactics:
            lo = float(min_spend.get(tactic, 0.0))
            hi = max_spend.get(tactic)
            var_bounds.append([lo, float(hi) if hi is not None else None])

        # ── 5e. Pre-solve feasibility check: floor sum vs. budget ───────────
        floor_sum = sum(b[0] for b in var_bounds if b[0] is not None)
        if floor_sum > total_budget:
            floor_lines = ", ".join(
                f"{tactics[i]}: {var_bounds[i][0]:,.0f}"
                for i in range(n)
                if var_bounds[i][0] and var_bounds[i][0] > 0
            )
            return {
                "text": (
                    f"Infeasibility Detected: Your minimum spend floors sum to "
                    f"${floor_sum:,.0f} but the total budget is ${total_budget:,.0f}. "
                    f"The problem has no feasible solution as specified.\n"
                    f"Floors: {floor_lines}\n"
                    f"Suggestion: Reduce one or more minimum floors, or increase the total budget."
                ),
                "data": None, "model": None,
            }

        # ── 6. Solve ────────────────────────────────────────────────────────
        formatted_bounds = [(b[0], b[1]) for b in var_bounds]

        res = linprog(
            c=c_vec,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=formatted_bounds,
            method='highs',
        )

        if not res.success:
            return {
                "text": (
                    f"SAC Optimization could not find a feasible solution.\n"
                    f"Solver message: {res.message}\n\n"
                    f"Common fixes:\n"
                    f"  • Minimum spend floors may exceed the total budget\n"
                    f"  • A target_activations value may be unreachable with this budget\n"
                    f"  • A per-tactic max_spend cap may conflict with a min_spend floor\n"
                    f"Try loosening the tightest constraint and re-running."
                ),
                "data": None, "model": None,
            }

        # ── 7. Compute projected outcomes ───────────────────────────────────
        alloc    = res.x                                     # dollars per tactic
        proj_act = alloc * act_per_dollar                    # projected activations
        proj_npv = alloc * npv_per_dollar                    # projected NPV contribution
        proj_sac = alloc * sac_per_dollar                    # projected SAC magnitude

        total_alloc    = alloc.sum()
        total_act      = proj_act.sum()
        total_npv      = proj_npv.sum()
        total_sac_cost = proj_sac.sum()
        reported_obj   = total_npv if obj_label != "sac" else total_sac_cost

        # ── 8. Human-readable result text ───────────────────────────────────
        result_text  = f"SAC Optimization Results — {obj_description}\n"
        result_text += f"{'=' * 60}\n\n"
        result_text += f"Budget:               ${total_budget:>14,.0f}\n"
        result_text += f"Allocated:            ${total_alloc:>14,.0f}\n"
        result_text += f"Projected Activations:{total_act:>15,.0f}\n"
        result_text += f"Projected Total NPV:  ${total_npv:>14,.0f}\n"
        result_text += f"Projected Total SAC:  ${total_sac_cost:>14,.0f}\n"
        if total_act > 0:
            result_text += f"NPV per Activation:   ${total_npv / total_act:>14,.2f}\n"
            result_text += f"SAC per Activation:   ${total_sac_cost / total_act:>14,.2f}\n"
        result_text += "\n"

        if start_year or end_year:
            yr_range = f"{start_year or 'all'} – {end_year or 'present'}"
            result_text += f"Historical data range: {yr_range}\n\n"

        result_text += "Optimal Spend Allocation by Tactic:\n"
        sorted_idx = np.argsort(-alloc)  # descending by allocation
        for i in sorted_idx:
            if alloc[i] >= 1.0:  # suppress ~$0 tactics
                pct = (alloc[i] / total_alloc * 100) if total_alloc > 0 else 0
                result_text += (
                    f"  • {tactics[i]:<25}  ${alloc[i]:>12,.0f}  ({pct:5.1f}%)  "
                    f"→ ~{proj_act[i]:,.0f} activations  |  NPV: ${proj_npv[i]:,.0f}\n"
                )

        # ── 9. Historical benchmarks for context ────────────────────────────
        result_text += "\nHistorical Benchmarks (from acquisition data):\n"
        result_text += f"  {'Tactic':<25}  {'Hist. Activations':>18}  {'Avg NPV/Sub':>12}  {'Avg Mkt Cost/Sub':>16}\n"
        result_text += f"  {'-'*25}  {'-'*18}  {'-'*12}  {'-'*16}\n"
        for _, row in econ_df.iterrows():
            result_text += (
                f"  {row['Tactic']:<25}  {int(row['activations']):>18,}  "
                f"${row['avg_npv']:>11,.2f}  ${row['cost_per_activation']:>15,.2f}\n"
            )

        # ── 10. Sensitivity analysis (binding constraints) ──────────────────
        sensitivity_lines = []
        constraint_labels = ["Total Budget Ceiling"]
        for i, tactic in enumerate(tactics):
            if max_spend.get(tactic) is not None:
                constraint_labels.append(f"Max Spend Cap — {tactic}")

        try:
            if hasattr(res, "ineqlin") and res.ineqlin is not None:
                marginals = np.atleast_1d(res.ineqlin.marginals)
                for idx, shadow in enumerate(marginals):
                    # For maximisation the solver negated c, so marginals need to be flipped
                    display_shadow = -shadow if maximize_flag else shadow
                    if abs(display_shadow) > 1e-4:
                        label = constraint_labels[idx] if idx < len(constraint_labels) else f"Constraint {idx+1}"
                        direction = "increases" if display_shadow > 0 else "decreases"
                        result_text += (
                            f"\nSensitivity — {label}:\n"
                            f"  Relaxing this constraint by $1 {direction} the objective by "
                            f"${abs(display_shadow):,.4f}\n"
                        )
                        if label == "Total Budget Ceiling" and display_shadow != 0:
                            result_text += (
                                f"  → Every additional $1,000 of budget yields approximately "
                                f"${abs(display_shadow) * 1000:,.2f} more in projected NPV.\n"
                            )

            if hasattr(res, "upper") and res.upper is not None:
                rc = np.atleast_1d(res.upper.marginals)
                for idx, reduced in enumerate(rc):
                    display_rc = -reduced if maximize_flag else reduced
                    if abs(display_rc) > 1e-4 and idx < n:
                        result_text += (
                            f"\nSensitivity — {tactics[idx]} is at its max-spend cap:\n"
                            f"  Increasing the cap by $1,000 would add approximately "
                            f"${abs(display_rc) * 1000:,.2f} to the objective.\n"
                        )
        except Exception:
            pass  # Sensitivity is best-effort

        # ── 11. Plain-English recommendation ────────────────────────────────
        top_tactic = tactics[sorted_idx[0]]
        top_pct    = (alloc[sorted_idx[0]] / total_alloc * 100) if total_alloc > 0 else 0
        result_text += (
            f"\nRecommendation: Concentrate the largest share of the budget on "
            f"'{top_tactic}' ({top_pct:.1f}% of total spend), which historically "
            f"delivers the best return per marketing dollar invested. "
        )
        if len(sorted_idx) > 1:
            second = tactics[sorted_idx[1]]
            result_text += f"The second-highest allocation goes to '{second}'."

        # ── 12. DataFrame for Excel export ──────────────────────────────────
        df_out = pd.DataFrame({
            "Tactic":                    tactics,
            "Optimal_Spend_$":           alloc,
            "Share_of_Budget_%":         alloc / total_alloc * 100 if total_alloc > 0 else 0,
            "Projected_Activations":     proj_act,
            "Projected_NPV_$":           proj_npv,
            "Projected_SAC_Cost_$":      proj_sac,
            "Hist_Avg_NPV_per_Sub_$":    econ_df["avg_npv"].values,
            "Hist_Cost_per_Activation_$": econ_df["cost_per_activation"].values,
            "Hist_Activation_Count":     econ_df["activations"].values,
        })
        summary_row = pd.DataFrame([{
            "Tactic": "TOTAL",
            "Optimal_Spend_$":           total_alloc,
            "Share_of_Budget_%":         100.0,
            "Projected_Activations":     total_act,
            "Projected_NPV_$":           total_npv,
            "Projected_SAC_Cost_$":      total_sac_cost,
            "Hist_Avg_NPV_per_Sub_$":    None,
            "Hist_Cost_per_Activation_$": None,
            "Hist_Activation_Count":     econ_df["activations"].sum(),
        }])
        df_out = pd.concat([df_out, summary_row], ignore_index=True)

        # ── 13. Bar chart of optimal allocation ─────────────────────────────
        chart_df = pd.DataFrame({
            "Tactic":      tactics,
            "Spend ($)":   alloc,
        }).sort_values("Spend ($)", ascending=True)

        fig = px.bar(
            chart_df,
            x="Spend ($)",
            y="Tactic",
            orientation="h",
            title=f"Optimal Marketing Spend Allocation (Budget: ${total_budget:,.0f})",
            labels={"Spend ($)": "Recommended Spend ($)", "Tactic": "Marketing Tactic"},
            template="plotly_white",
            color="Spend ($)",
            color_continuous_scale="Viridis",
        )
        fig.update_layout(margin=dict(l=40, r=40, t=60, b=40))

        return {
            "text":   result_text,
            "data":   df_out,
            "figure": fig,
            "model":  res,
        }

    except Exception as e:
        return {
            "text": f"SAC Optimization Error: {str(e)}\n{traceback.format_exc()}",
            "data": None, "figure": None, "model": None,
        }


@mlflow.trace(name="run_random_forest_tool")
def run_random_forest_tool(
    target_variable: str, 
    feature_variables: list, 
    TABLE_NAME: Optional[Union[str, List[str]]] = None,
    dataframe_id: Optional[str] = None,
    task_type: str = "regression", 
    n_estimators: int = 100,
    df_memory: DataFrameMemory = None
) -> Dict[str, Any]:
    columns_to_fetch = [target_variable] + feature_variables

    try:
        # ── 1. Data Loading ──
        if dataframe_id:
            df = df_memory.get_df(dataframe_id) if df_memory else None
            if df is None:
                return {"text": f"Error: No DataFrame found for ID '{dataframe_id}'.", "model": None}
        elif TABLE_NAME:
            df = _link_tables(TABLE_NAME, columns=columns_to_fetch, random_order=True, limit=max_rows)
        else:
            return {"text": "Error: Must provide either TABLE_NAME or dataframe_id.", "model": None}

        # ── 2. Column Validation & Cleaning ──
        missing = [c for c in columns_to_fetch if c not in df.columns]
        if missing:
            return {
                "text": f"Error: Columns missing: {missing}. Available: {df.columns.tolist()}",
                "model": None
            }

        df = df[columns_to_fetch].copy()
        if df.empty or len(df) <= len(feature_variables):
            return {"text": "Error: Not enough data points.", "model": None}

        # ── 3. Target Preparation ──
        task = task_type.lower()
        if task == "regression":
            df[target_variable] = pd.to_numeric(df[target_variable], errors="coerce")
        else:
            df[target_variable] = df[target_variable].astype(str)

        df = df.dropna(subset=[target_variable])
        if len(df) < min_rows:
            return {"text": "Error: Not enough valid target rows to train a model.", "model": None}

        # ── 4. Feature Encoding ──
        categorical_features = [col for col in feature_variables if col in df.columns and df[col].dtype == "object"]
        numeric_features = [col for col in feature_variables if col in df.columns and col not in categorical_features]

        for col in numeric_features:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        if categorical_features:
            encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            df[categorical_features] = encoder.fit_transform(df[categorical_features].astype(str))

        current_features = feature_variables
        df = df.dropna(subset=[target_variable] + current_features)

        # ── 5. Train / Test Split ──
        X = df[current_features]
        y = df[target_variable]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=split, random_state=Random_state
        )

        # ── 6. Model Fitting & Evaluation ──
        if task == "regression":
            model = RandomForestRegressor(
                n_estimators=n_estimators, max_depth=Max_depth,
                min_samples_leaf=rf_min_leaf_size, random_state=Random_state, n_jobs=-1
            )
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            r2 = r2_score(y_test, preds)
            rmse = mean_squared_error(y_test, preds) ** 0.5

            result_text = f"Random Forest Regression Results (n_estimators={n_estimators}):\n"
            result_text += f"  • Test R²:   {r2:.4f}\n"
            result_text += f"  • Test RMSE: {rmse:.4f}\n\n"

            # Diagnostic Chart: Actual vs. Predicted
            diag_fig = px.scatter(
                x=y_test, y=preds,
                labels={"x": f"Actual {target_variable}", "y": f"Predicted {target_variable}"},
                title=f"Actual vs. Predicted ({target_variable})",
                template="plotly_white", opacity=0.7
            )
            diag_fig.add_shape(
                type="line", line=dict(dash="dash", color="gray"),
                x0=y_test.min(), y0=y_test.min(), x1=y_test.max(), y1=y_test.max()
            )
        else:
            model = RandomForestClassifier(
                n_estimators=n_estimators, max_depth=Max_depth,
                min_samples_leaf=rf_min_leaf_size, random_state=Random_state, n_jobs=-1
            )
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)

            result_text = f"Random Forest Classification Results (n_estimators={n_estimators}):\n"
            result_text += f"  • Test Accuracy: {acc:.4f}\n"
            result_text += f"Classification Report:\n{classification_report(y_test, preds)}\n\n"

            diag_fig = None

        # ── 7. Feature Importance Dataframe (For Excel Export) ──
        importances = model.feature_importances_
        feat_imp_df = pd.DataFrame({
            "Feature": current_features,
            "Importance_Score": importances,
            "Importance_Percentage": importances * 100
        }).sort_values(by="Importance_Score", ascending=False).reset_index(drop=True)

        result_text += f"Feature Importances (Top {min(10, len(feat_imp_df))}):\n"
        for _, row in feat_imp_df.head(10).iterrows():
            result_text += f"  • {row['Feature']}: {row['Importance_Score']:.4f} ({row['Importance_Percentage']:.1f}%)\n"

        # ── 8. Feature Importance Plotly Chart ──
        top_features = feat_imp_df.head(15).sort_values(by="Importance_Score", ascending=True)
        fig_importance = px.bar(
            top_features,
            x="Importance_Score",
            y="Feature",
            orientation="h",
            title=f"Random Forest Feature Importances ({target_variable})",
            labels={"Importance_Score": "Gini Importance", "Feature": "Feature"},
            template="plotly_white",
            color="Importance_Score",
            color_continuous_scale="Viridis"
        )
        fig_importance.update_layout(margin=dict(l=40, r=40, t=60, b=40))

        # ── 9. Representative Tree Visualization ─────────────────────────
        # A Random Forest has no single merged tree.  The most informative
        # visualization is the *most representative* individual tree — the one
        # whose per-sample predictions on the test set best match the full
        # ensemble's predictions.  We find it by minimising MSE between each
        # tree's leaf-level predictions and the ensemble's predictions on X_test.
        fig_tree = None
        try:
            ensemble_preds = model.predict(X_test)

            best_tree_idx = 0
            best_mse = float("inf")
            for i, tree in enumerate(model.estimators_):
                tree_preds = tree.predict(X_test)
                mse = float(np.mean((tree_preds - ensemble_preds) ** 2))
                if mse < best_mse:
                    best_mse = mse
                    best_tree_idx = i

            best_tree = model.estimators_[best_tree_idx]

            # ── Convert the sklearn DecisionTree to a Plotly figure ──────
            # sklearn exposes the full internal tree arrays; we walk them to
            # build a node/edge trace without needing graphviz or pydot.
            tree_ = best_tree.tree_
            n_nodes      = tree_.node_count
            children_l   = tree_.children_left
            children_r   = tree_.children_right
            feature_idx  = tree_.feature
            thresholds   = tree_.threshold
            values       = tree_.value          # shape (n_nodes, n_outputs, n_classes)
            n_samples_node = tree_.n_node_samples

            is_leaf = children_l == -1  # TREE_LEAF sentinel

            # ── Compute (x, y) positions via BFS level-order layout ───────
            positions: Dict[int, tuple] = {}
            queue = [(0, 0.5, 1.0)]   # (node_id, x_centre, depth_y)
            x_offsets: Dict[int, float] = {0: 0.5}
            level_widths: Dict[int, float] = {0: 1.0}

            while queue:
                node, x_c, depth = queue.pop(0)
                positions[node] = (x_c, -depth)  # negative so root is at top
                width = level_widths.get(node, 1.0) / 2.0

                if not is_leaf[node]:
                    left  = children_l[node]
                    right = children_r[node]
                    level_widths[left]  = width
                    level_widths[right] = width
                    queue.append((left,  x_c - width / 2, depth + 1))
                    queue.append((right, x_c + width / 2, depth + 1))

            # ── Build edge traces ─────────────────────────────────────────
            edge_x, edge_y = [], []
            for node in range(n_nodes):
                if not is_leaf[node]:
                    px_c, py_c = positions[node]
                    for child in (children_l[node], children_r[node]):
                        cx, cy = positions[child]
                        edge_x += [px_c, cx, None]
                        edge_y += [py_c, cy, None]

            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                mode="lines",
                line=dict(color="#AAAAAA", width=1),
                hoverinfo="none",
                showlegend=False,
            )

            # ── Build node traces ─────────────────────────────────────────
            node_x, node_y, node_text, node_hover, node_color = [], [], [], [], []

            for node in range(n_nodes):
                nx_c, ny_c = positions[node]
                node_x.append(nx_c)
                node_y.append(ny_c)
                n_samp = int(n_samples_node[node])

                if is_leaf[node]:
                    # Leaf: show predicted value
                    val_arr = values[node]
                    if task == "regression":
                        pred_val = float(val_arr[0][0])
                        label = f"{pred_val:.2f}"
                        hover = f"Leaf<br>Prediction: {pred_val:.4f}<br>Samples: {n_samp}"
                        node_color.append(pred_val)
                    else:
                        class_counts = val_arr[0]
                        majority = int(np.argmax(class_counts))
                        # Map ordinal index back to original class label if possible
                        try:
                            class_label = model.classes_[majority]
                        except Exception:
                            class_label = str(majority)
                        label = str(class_label)
                        purity = float(class_counts[majority] / class_counts.sum())
                        hover = (
                            f"Leaf<br>Class: {class_label}<br>"
                            f"Purity: {purity:.1%}<br>Samples: {n_samp}"
                        )
                        node_color.append(purity)
                else:
                    # Split node: show feature name and threshold
                    feat_name = current_features[feature_idx[node]]
                    thresh    = thresholds[node]
                    label = f"{feat_name}<br>≤ {thresh:.3g}"
                    hover = (
                        f"Split on: {feat_name}<br>"
                        f"Threshold: {thresh:.4g}<br>"
                        f"Samples: {n_samp}"
                    )
                    node_color.append(0.0)  # interior nodes stay neutral

                node_text.append(label)
                node_hover.append(hover)

            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode="markers+text",
                marker=dict(
                    size=22,
                    color=node_color,
                    colorscale="RdYlGn",
                    showscale=True,
                    colorbar=dict(
                        title="Pred. Value" if task == "regression" else "Purity",
                        thickness=12, len=0.5
                    ),
                    line=dict(color="#555555", width=1),
                ),
                text=node_text,
                textposition="middle center",
                textfont=dict(size=8, color="black"),
                hovertext=node_hover,
                hoverinfo="text",
                showlegend=False,
            )

            n_leaves = int(is_leaf.sum())
            depth    = int(best_tree.get_depth())
            fig_tree = go.Figure(
                data=[edge_trace, node_trace],
                layout=go.Layout(
                    title=dict(
                        text=(
                            f"Most Representative Tree (tree #{best_tree_idx} of {n_estimators}) — "
                            f"depth {depth}, {n_leaves} leaves<br>"
                            f"<sup>This individual tree's predictions most closely match the full "
                            f"ensemble on the test set.</sup>"
                        ),
                        font=dict(size=13),
                    ),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    template="plotly_white",
                    margin=dict(l=20, r=20, t=80, b=20),
                    height=600,
                ),
            )

            result_text += (
                f"\nRepresentative Tree: tree #{best_tree_idx} (depth {depth}, {n_leaves} leaves). "
                f"Selected because its predictions most closely match the full {n_estimators}-tree "
                f"ensemble on the held-out test set.\n"
            )

        except Exception as tree_err:
            # Tree viz is best-effort — never block the main result
            result_text += f"\n(Tree visualization unavailable: {tree_err})\n"

        # ── 10. Return Unified Payload ────────────────────────────────────
        figures = [fig_importance]
        if fig_tree is not None:
            figures.append(fig_tree)

        return {
            "text":    result_text,
            "data":    feat_imp_df,
            "figure":  fig_importance,   # kept for backward compat with loop.py "figure" key
            "figures": figures,          # both charts for the UI renderer
            "model":   model,
        }

    except Exception as e:
        return {"text": f"Random Forest Error: {e}", "data": None, "figure": None, "figures": [], "model": None}
  

# ─── Forecasting & Scenario Planning Tools ───────────────────────────────────────────
@mlflow.trace(name="run_forecasting_tool")
def run_forecasting_tool(
    value_column: str, 
    TABLE_NAME: Optional[Union[str, List[str]]] = None,
    dataframe_id: Optional[str] = None,
    where_clause: Optional[str] = None,
    aggregation: str = "SUM", 
    steps: int = 6,
    trend: str = "add",      # 'add' or 'mul'
    seasonal: str = "add",   # 'add' or 'mul'
    seasonal_periods: int = 12,
    df_memory: DataFrameMemory = None
) -> dict:
    """
    Aggregates value_column to one observation per calendar month, then fits a
    Holt-Winters Exponential Smoothing model and forecasts `steps` periods ahead.
    Year/month column names are resolved automatically from TABLE_DIMENSIONS so
    this function works across all registered tables without manual configuration.
    Returns forecast values as formatted text and the fitted model object.
    """
    safe_value = '"{}"'.format(value_column.replace('"', ''))
    agg_func = aggregation.upper() if aggregation.upper() in ["SUM", "AVG", "COUNT"] else "SUM"
    val_col_clean = value_column.replace('"', '').strip()

    try:
        if dataframe_id:
            df = df_memory.get_df(dataframe_id) if df_memory else None
            if df is None:
                return {"text": f"Error: No DataFrame found for ID '{dataframe_id}'.", "data": None}

            # Detect whichever year/month columns are present in the dataframe.
            # Check TABLE_DIMENSIONS values first, then fall back to common synonyms.
            known_year_cols = {dims["year"] for dims in TABLE_DIMENSIONS.values()}
            known_month_cols = {dims["month"] for dims in TABLE_DIMENSIONS.values()}

            year_col = next((c for c in known_year_cols if c in df.columns), None)
            month_col = next((c for c in known_month_cols if c in df.columns), None)

            if year_col and month_col and val_col_clean in df.columns:
                # Aggregate down to one row per period
                agg_map = {"SUM": "sum", "AVG": "mean", "COUNT": "count"}
                df = (
                    df.groupby([year_col, month_col], as_index=False)[val_col_clean]
                    .agg(agg_map[agg_func])
                    .sort_values(by=[year_col, month_col])
                    .rename(columns={val_col_clean: "target_value"})
                )
            elif val_col_clean in df.columns:
                # Dataframe is already a clean time series — use it as-is
                df = df.copy()
                df["target_value"] = df[val_col_clean]
            else:
                return {"text": f"Error: Column '{val_col_clean}' not found in the provided dataframe. Available columns: {df.columns.tolist()}", "data": None}

        elif TABLE_NAME:
            table_key = TABLE_NAME if isinstance(TABLE_NAME, str) else TABLE_NAME[0]
            dims = TABLE_DIMENSIONS.get(table_key)

            if dims is None:
                return {"text": f"Error: Table '{table_key}' not found in TABLE_DIMENSIONS. Please add it to base.py.", "data": None}

            year_col = dims["year"]
            month_col = dims["month"]
            
            # Combine the hardcoded date requirement with the user's where_clause
            base_where = f'"{year_col}" IS NOT NULL AND "{month_col}" IS NOT NULL'
            final_where = f'({base_where}) AND ({where_clause})' if where_clause else base_where

            columns_to_fetch = [
                f'"{year_col}"',
                f'"{month_col}"',
                f'{agg_func}({safe_value}) AS target_value'
            ]
            
            df = _link_tables(
                tables=TABLE_NAME,
                columns=columns_to_fetch,
                where_clause=final_where,
                group_by=[year_col, month_col],
                order_by=f'"{year_col}" ASC, "{month_col}" ASC',
                limit=None
            )
        else:
            return {"text": "Error: Must provide either TABLE_NAME or dataframe_id.", "data": None}

        df["target_value"] = pd.to_numeric(df.get("target_value", pd.Series(dtype=float)), errors="coerce")
        df = df.dropna(subset=["target_value"])

        if df.empty or len(df) < min_rows:
            return {"text": f"Error: Not enough historical data points (minimum {min_rows} required) to perform projections.", "data": None}

        series = df["target_value"].values

        model = ExponentialSmoothing(
            series, 
            trend=trend, 
            seasonal=seasonal, 
            seasonal_periods=seasonal_periods
        )
        # Using optimized=True allows statsmodels to find the best smoothing weights
        model_fit = model.fit(optimized=True) 
        forecast = model_fit.forecast(steps=steps)

        result_text = f"Holt-Winters Forecasting Results for {agg_func} of {value_column}:\n"
        result_text += f"Based on {len(series)} periods (Trend: {trend}, Seasonal: {seasonal}, Periods: {seasonal_periods})\n"
        result_text += f"Predictions for the next {steps} periods:\n"
        for i, val in enumerate(forecast, start=1):
            result_text += f"  • Period +{i}: {val:.4f}\n"

        return {"text": result_text, "data": model_fit}

    except Exception as e:
        return {"text": f"Holt-Winters Forecasting Error: {e}", "data": None}    


@mlflow.trace(name="run_scenario_planning_tool")
def run_scenario_planning_tool(
    target_variable: str, 
    scenario_changes: list,
    feature_variables: Optional[list] = None,
    hold_constant_variables: Optional[list] = None,
    TABLE_NAME: Optional[Union[str, List[str]]] = None, 
    dataframe_id: Optional[str] = None,
    where_clause: Optional[str] = None,
    confidence_level: float = 0.95,
    df_memory: DataFrameMemory = None
) -> Dict[str, Any]:
    """
    Table-agnostic scenario planning tool. Fits an OLS regression on historical data, 
    then predicts the target variable under a hypothetical scenario where specified 
    features are set to new values and specified control features are held constant at 
    their historical means.
    
    Dynamically joins any combination of tables using link_tables and TABLE_DIMENSIONS.
    """
    hold_constant_variables = hold_constant_variables or feature_variables or []
    # Clean input strings
    target_variable = str(target_variable).replace('"', '').replace("'", "").strip()
    hold_constant_variables = [str(col).replace('"', '').replace("'", "").strip() for col in hold_constant_variables]
    
    # Map out the changes
    changes_map = {}
    for item in scenario_changes:
        col_name = item.get("column_name", "") if isinstance(item, dict) else getattr(item, "column_name", "")
        val = item.get("new_value", 0.0) if isinstance(item, dict) else getattr(item, "new_value", 0.0)
        clean_col = str(col_name).replace('"', '').replace("'", "").strip()
        changes_map[clean_col] = float(val)
        
    all_features = list(set(hold_constant_variables + list(changes_map.keys())))
    all_columns = [target_variable] + all_features
    
    try:
        # 1. Fetch Data
        if dataframe_id:
            df = df_memory.get_df(dataframe_id) if df_memory else None
            if df is None:
                return {"text": f"Error: No DataFrame found for ID '{dataframe_id}'.", "data": None, "model": None}
        elif TABLE_NAME:
            # Delegate all complex cross-table joining to the centralized helper
            df = _link_tables(
                tables=TABLE_NAME, 
                columns=all_columns, 
                where_clause=where_clause,
                random_order=True, 
                limit=max_rows
            )
        else:
            return {"text": "Error: Must provide either TABLE_NAME or dataframe_id.", "data": None, "model": None}
            
        # 2. Validate and Clean Data
        missing_cols = [col for col in all_columns if col not in df.columns]
        if missing_cols:
            return {
                "text": f"Error: Missing required columns: {missing_cols}. Available: {df.columns.tolist()}", 
                "data": None, 
                "model": None
            }

        for col in all_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=all_columns)
        
        if df.empty or len(df) <= len(all_features) + scenario_min_rows:
            return {"text": f"Error: Not enough data points to build a reliable scenario model (minimum {scenario_min_rows} required).", "data": None, "model": None}
            
        # 3. Fit OLS Model
        historical_target_mean = df[target_variable].mean()
        
        Y = df[target_variable]
        X = df[all_features]
        X_with_const = sm.add_constant(X)
        
        model = sm.OLS(Y, X_with_const).fit()
        
        # 4. Construct Scenario Point
        scenario_point = pd.Series(index=X_with_const.columns, dtype=float)
        scenario_point['const'] = 1.0
        
        held_constant_log = []
        
        for col in all_features:
            col_mean = X[col].mean()
            
            if col in changes_map:
                scenario_point[col] = float(changes_map[col])
            else:
                scenario_point[col] = col_mean
                held_constant_log.append(f"{col} (held at avg: {col_mean:,.2f})")
                
        # 5. Predict and Evaluate
        prediction_results = model.get_prediction(scenario_point)
        pred_df = prediction_results.summary_frame(alpha=1.0 - confidence_level)
        
        predicted_val = pred_df['mean'].values[0]
        ci_lower = pred_df['obs_ci_lower'].values[0]
        ci_upper = pred_df['obs_ci_upper'].values[0]
        diff_from_baseline = predicted_val - historical_target_mean
        
        # 6. Format Output
        result_text = f"--- Scenario Analysis for Target: '{target_variable}' ---\n\n"
            
        result_text += f"1. Baseline Context ({len(df)} observations analyzed):\n"
        result_text += f"  * Historical Average of {target_variable}: {historical_target_mean:,.2f}\n"
        result_text += f"  * Model R-Squared: {model.rsquared:.4f}\n\n"
        
        result_text += f"2. Scenario Conditions & Sensitivity:\n"
        for col, new_val in changes_map.items():
            hist_mean = X[col].mean()
            pct_change = ((new_val - hist_mean) / hist_mean) * 100 if hist_mean != 0 else 0
            coef_val = model.params.get(col, 0.0)
            
            result_text += f"  * CHANGED: '{col}' set to {new_val:,.2f}\n"
            result_text += f"    - Historical Avg: {hist_mean:,.2f} ({pct_change:+.1f}% change)\n"
            result_text += f"    - Marginal Impact (β): {coef_val:+.4f} {target_variable} per +1.0 unit of {col}\n"
            
        if held_constant_log:
            result_text += "\n  * HELD CONSTANT:\n    - " + "\n    - ".join(held_constant_log) + "\n\n"
            
        result_text += f"3. Scenario Prediction ({int(confidence_level*100)}% Confidence):\n"
        result_text += f"  * Expected {target_variable}: {predicted_val:,.2f}\n"
        result_text += f"  * Net Impact vs Baseline: {diff_from_baseline:+,.2f}\n"
        result_text += f"  * Interval: [{ci_lower:,.2f} to {ci_upper:,.2f}]\n"
        
        return {"text": result_text, "data": df, "model": model}
        
    except Exception as e:
        return {"text": f"Scenario Planning Error: {str(e)}", "data": None, "model": None}
