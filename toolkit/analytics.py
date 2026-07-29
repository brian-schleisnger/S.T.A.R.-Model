import traceback
from typing import Any, Dict, List, Optional, Union

import mlflow
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from agent.memory import DataFrameMemory
from .base import TABLE_DIMENSIONS
from .transformations import _link_tables


YEARLY_WACC = 0.1
MONTHLY_WACC = (1 + YEARLY_WACC) ** (1 / 12) - 1

__all__ = [
    "run_ols_regression_tool",
    "run_forecasting_tool",
    "run_random_forest_tool",
    "run_pca_tool",
    "run_kmeans_clustering_tool",
    "run_scenario_planning_tool",
    "run_neural_network_tool",
    "run_optimization_tool",
    "calculate_mutual_information_tool",
]


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
            df = _link_tables(TABLE_NAME, columns=columns_to_fetch, random_order=True, limit=100000)
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

        if len(df) < 10:
            return {"text": "Error: Data size too small after cleaning to calculate reliable mutual information.", "data": None}

        # ── 5. Calculate Mutual Information ──────────────────────────────
        X = df[current_features]
        y = df[target_variable]

        if task == "continuous":
            mi_scores = mutual_info_regression(X, y, discrete_features=discrete_mask, random_state=42)
        else:
            mi_scores = mutual_info_classif(X, y, discrete_features=discrete_mask, random_state=42)

        # ── 6. Format Outputs ────────────────────────────────────────────
        mi_results = sorted(
            zip(current_features, mi_scores),
            key=lambda x: x[1],
            reverse=True
        )
        
        result_text = f"Mutual Information Analysis ({target_type.capitalize()} Target: '{target_variable}'):\n"
        result_text += f"Calculated using Shannon information theory on {len(df):,} observations.\n"
        result_text += "Higher values indicate stronger dependency (measured in nats).\n\n"
        
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
            df = _link_tables(TABLE_NAME, columns=feature_variables, random_order=True, limit=100000)
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
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
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
            df = _link_tables(TABLE_NAME, columns=columns_to_fetch, where_clause=where_clause, limit=100000) 
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
            df = _link_tables(TABLE_NAME, columns=feature_variables, random_order=True, limit=100000)
        else:
            return {"text": "Error: Must provide either TABLE_NAME or dataframe_id.", "model": None}
            
        if df.empty or len(df) < 2:
            return {"text": "Error: Not enough data points fetched to perform PCA.", "model": None}
            
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
    max_iter: int = 500,
    df_memory: DataFrameMemory = None
) -> Dict[str, Any]:
    """
    Trains a scikit-learn MLPRegressor or MLPClassifier on the provided features.
    Features are one-hot encoded for categoricals and StandardScaler-normalized before
    training. Uses early stopping to avoid overfitting on small datasets.
    Returns the R² score (regression) or accuracy (classification) on the held-out
    test set, plus the fitted model and the cleaned DataFrame.
    """
    try:
        if dataframe_id:
            df = df_memory.get_df(dataframe_id) if df_memory else None
            if df is None:
                return {"text": f"Error: No DataFrame found for ID '{dataframe_id}'.", "data": None, "model": None}
        elif TABLE_NAME:
            df = _link_tables(TABLE_NAME, random_order=True, limit=100000)
        else:
            return {"text": "Error: Must provide either TABLE_NAME or dataframe_id.", "data": None, "model": None}
            
        df_clean = df.dropna(subset=[target_variable] + feature_variables)
        X = df_clean[feature_variables]
        y = df_clean[target_variable]
        
        X = pd.get_dummies(X, drop_first=True)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        if task_type == 'regression':
            model = MLPRegressor(hidden_layer_sizes=tuple(hidden_layer_sizes), max_iter=max_iter, early_stopping=True, random_state=42)
            model.fit(X_train_scaled, y_train)
            score = model.score(X_test_scaled, y_test)
            result_text = f"MLP Regression completed.\nTarget: {target_variable}\nR^2 Score on test set: {score:.4f}"
        else:
            model = MLPClassifier(hidden_layer_sizes=tuple(hidden_layer_sizes), max_iter=max_iter, early_stopping=True, random_state=42)
            model.fit(X_train_scaled, y_train)
            score = model.score(X_test_scaled, y_test)
            result_text = f"MLP Classification completed.\nTarget: {target_variable}\nAccuracy on test set: {score:.4f}"
            
        return {"text": result_text, "data": df_clean, "model": model}
        
    except Exception as e:
        return {"text": f"Neural Network Error: {e}\n{traceback.format_exc()}", "data": None, "model": None}


@mlflow.trace(name="run_optimization_tool")
def run_optimization_tool(
    objective_coefficients: List[float],
    inequality_constraints_matrix: Optional[List[List[float]]] = None,
    inequality_constraints_bounds: Optional[List[float]] = None,
    equality_constraints_matrix: Optional[List[List[float]]] = None,
    equality_constraints_bounds: Optional[List[float]] = None,
    bounds: Optional[List[List[Optional[float]]]] = None
) -> Dict[str, Any]:
    """
    Solves a linear programming problem using scipy.optimize.linprog (HiGHS solver).
    Minimizes c·x subject to A_ub·x ≤ b_ub, A_eq·x = b_eq, and per-variable bounds.
    To maximize instead of minimize, pass negative objective coefficients.
    Returns the optimal objective value and decision variable values on success,
    or the solver failure message on infeasibility.
    """
    try:
        formatted_bounds = None
        if bounds is not None:
            formatted_bounds = [(b[0], b[1]) if len(b) >= 2 else (None, None) for b in bounds]
            
        res = linprog(
            c=objective_coefficients,
            A_ub=inequality_constraints_matrix,
            b_ub=inequality_constraints_bounds,
            A_eq=equality_constraints_matrix,
            b_eq=equality_constraints_bounds,
            bounds=formatted_bounds,
            method='highs'
        )
        
        if res.success:
            result_text = f"Optimization Successful!\nOptimal Objective Value: {res.fun:.4f}\nOptimal Variables: {res.x}"
            
            # 1. Build a structured DataFrame for Excel export
            df_results = pd.DataFrame({
                "Variable": [f"Variable_{i+1}" for i in range(len(res.x))],
                "Coefficient": objective_coefficients,
                "Optimal_Value": res.x,
                "Total_Contribution": np.array(objective_coefficients) * res.x
            })
            
            # 2. Append an overall total summary row at the bottom
            summary_row = pd.DataFrame([{
                "Variable": "TOTAL / OPTIMAL OBJECTIVE",
                "Coefficient": None,
                "Optimal_Value": res.x.sum(),
                "Total_Contribution": res.fun
            }])
            df_results = pd.concat([df_results, summary_row], ignore_index=True)
            
            # 3. Return the DataFrame in the "data" key instead of None
            return {"text": result_text, "data": df_results, "model": res}
        
    except Exception as e:
        return {"text": f"Optimization Error: {str(e)}", "data": None, "model": None}


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
        # ── 1. Data Loading ──────────────────────────────────────────────
        if dataframe_id:
            df = df_memory.get_df(dataframe_id) if df_memory else None
            if df is None:
                return {"text": f"Error: No DataFrame found for ID '{dataframe_id}'.", "model": None}
        elif TABLE_NAME:
            df = _link_tables(TABLE_NAME, columns=columns_to_fetch, random_order=True, limit=100000)
        else:
            return {"text": "Error: Must provide either TABLE_NAME or dataframe_id.", "model": None}

        # ── 2. Column Validation ─────────────────────────────────────────
        # Check requested columns actually exist before doing any work.
        missing = [c for c in columns_to_fetch if c not in df.columns]
        if missing:
            return {
                "text": f"Error: The following columns were not found in the data: {missing}. "
                        f"Available columns: {df.columns.tolist()}",
                "model": None
            }

        # Narrow the dataframe to only the columns we care about so stray
        # underscore-named columns from the broader table can never leak in.
        df = df[columns_to_fetch].copy()

        if df.empty or len(df) <= len(feature_variables):
            return {"text": "Error: Not enough data points.", "model": None}

        # ── 3. Target Preparation ────────────────────────────────────────
        task = task_type.lower()
        if task == "regression":
            df[target_variable] = pd.to_numeric(df[target_variable], errors="coerce")
        else:
            # For classification keep target as string so class labels are readable
            df[target_variable] = df[target_variable].astype(str)

        # Drop rows where the target is missing before encoding features
        df = df.dropna(subset=[target_variable])

        if len(df) < 10:
            return {"text": "Error: Not enough valid target rows to train a model.", "model": None}

        # ── 4. Feature Encoding ──────────────────────────────────────────
        # Identify which of the *requested* feature columns are categorical
        categorical_features = [
            col for col in feature_variables
            if col in df.columns and df[col].dtype == "object"
        ]
        numeric_features = [
            col for col in feature_variables
            if col in df.columns and col not in categorical_features
        ]

        # Convert numeric features, coercing unparseable values to NaN
        for col in numeric_features:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # One-hot encode categoricals; drop_first avoids perfect multicollinearity
        if categorical_features:
            df = pd.get_dummies(df, columns=categorical_features, drop_first=True)

        # Rebuild the feature list from the current df columns — this correctly
        # picks up the new one-hot columns (e.g. 'channel_TV', 'channel_Digital')
        # while excluding the target and any other columns that might have slipped in.
        current_features = [col for col in df.columns if col != target_variable]

        # Drop any rows with NaN in features or target
        df = df.dropna(subset=[target_variable] + current_features)

        if len(df) < 10:
            return {"text": "Error: Data size too small after cleaning to train a valid model.", "model": None}

        # ── 5. Train / Test Split ────────────────────────────────────────
        X = df[current_features]
        y = df[target_variable]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # ── 6. Model Fitting & Evaluation ────────────────────────────────
        if task == "regression":
            model = RandomForestRegressor(
                n_estimators=n_estimators, max_depth=7,
                min_samples_leaf=3, random_state=42, n_jobs=-1
            )
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            result_text = f"Random Forest Regression Results (n_estimators={n_estimators}):\n"
            result_text += f"  • Test R²:   {r2_score(y_test, preds):.4f}\n"
            result_text += f"  • Test RMSE: {mean_squared_error(y_test, preds) ** 0.5:.4f}\n\n"
        else:
            model = RandomForestClassifier(
                n_estimators=n_estimators, max_depth=7,
                min_samples_leaf=3, random_state=42, n_jobs=-1
            )
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            result_text = f"Random Forest Classification Results (n_estimators={n_estimators}):\n"
            result_text += f"  • Test Accuracy: {accuracy_score(y_test, preds):.4f}\n"
            result_text += f"Classification Report:\n{classification_report(y_test, preds)}\n\n"

        # ── 7. Feature Importances ───────────────────────────────────────
        feat_imp = sorted(
            zip(current_features, model.feature_importances_),
            key=lambda x: x[1],
            reverse=True
        )
        result_text += f"Feature Importances — top {min(10, len(feat_imp))} of {len(feat_imp)} features "
        result_text += f"(trained on {len(X_train):,} rows, tested on {len(X_test):,} rows):\n"
        for feat, imp in feat_imp[:10]:
            result_text += f"  • {feat}: {imp:.4f}\n"

        return {"text": result_text, "model": model}

    except Exception as e:
        return {"text": f"Random Forest Error: {e}", "model": None}
  

# ─── Forecasting & Scenario Planning Tools ───────────────────────────────────────────
@mlflow.trace(name="run_forecasting_tool")
def run_forecasting_tool(
    value_column: str, 
    TABLE_NAME: Optional[Union[str, List[str]]] = None,
    dataframe_id: Optional[str] = None,
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
            # Resolve the canonical year/month column names for this table from TABLE_DIMENSIONS.
            # Normalize TABLE_NAME to a single string key for the lookup.
            table_key = TABLE_NAME if isinstance(TABLE_NAME, str) else TABLE_NAME[0]
            dims = TABLE_DIMENSIONS.get(table_key)

            if dims is None:
                return {"text": f"Error: Table '{table_key}' not found in TABLE_DIMENSIONS. Please add it to base.py.", "data": None}

            year_col = dims["year"]
            month_col = dims["month"]

            columns_to_fetch = [
                f'"{year_col}"',
                f'"{month_col}"',
                f'{agg_func}({safe_value}) AS target_value'
            ]
            df = _link_tables(
                tables=TABLE_NAME,
                columns=columns_to_fetch,
                where_clause=f'"{year_col}" IS NOT NULL AND "{month_col}" IS NOT NULL',
                group_by=[year_col, month_col],
                order_by=f'"{year_col}" ASC, "{month_col}" ASC',
                limit=None
            )
        else:
            return {"text": "Error: Must provide either TABLE_NAME or dataframe_id.", "data": None}

        df["target_value"] = pd.to_numeric(df.get("target_value", pd.Series(dtype=float)), errors="coerce")
        df = df.dropna(subset=["target_value"])

        if df.empty or len(df) < 10:
            return {"text": "Error: Not enough historical data points (minimum 10 required) to perform ARIMA.", "data": None}

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
                limit=100000
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
        
        if df.empty or len(df) <= len(all_features) + 3:
            return {"text": "Error: Not enough data points to build a reliable scenario model.", "data": None, "model": None}
            
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
