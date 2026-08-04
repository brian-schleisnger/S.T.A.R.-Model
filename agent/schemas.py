from typing import List, Optional, Literal, Union

from pydantic import BaseModel, Field

from agent.categories import CATEGORY_REGISTRY


# -------------------- DECOMPOSITION SCHEMAS --------------------
def _build_subquestion_category_description() -> str:
    rules = [
        "The single most appropriate execution category for this sub-question. "
        "Choose using these EXCLUSIVE rules — apply the FIRST rule that matches:\n"
    ]
    for cat, data in CATEGORY_REGISTRY.items():
        rules.append(f"• {cat} — {data['rule']}")
    return "\n\n".join(rules)
class SubQuestion(BaseModel):
    """A single decomposed data question paired with its required execution category."""
    question: str = Field(
        ..., 
        description="The self-contained, specific data question with all pronouns resolved."
    )
    target_category: Literal[
        "SQL_RETRIEVAL",
        "RATIO_ANALYSIS",
        "STATISTICAL_MODELING",
        "ML_MODELING",
        "SAC_OPTIMIZATION",
        "FORECASTING_AND_SCENARIOS",
        "VISUALIZATION",
        "CUSTOM_PYTHON",
        "DATA_TRANSFORMATION"
    ] = Field(
        ..., 
        description=_build_subquestion_category_description()
    )
class DecomposedQuestions(BaseModel):
    """The broken-down data queries based on the user's prompt, enriched with routing categories."""
    questions: List[SubQuestion] = Field(
        description="A list of specific, actionable, and categorized data queries. Max 10."
    )


# -------------------- ANALYTICS SCHEMAS --------------------
class calculate_mutual_information_tool_schema(BaseModel):
    """
    Calculates the mutual information (from Shannon information theory) between a target variable and one or more feature variables.
    Use this to determine how much information the features provide about the target variable, capturing both linear and non-linear dependencies.
    """
    TABLE_NAME: Optional[Union[str, List[str]]] = Field(
        ..., 
        description="The exact SQL-safe table name(s) to query, e.g., '\"sandbox\".\"acquisition_data_v3\"', '\"sandbox\".\"dbs_marketing_sync\"', or '\"sandbox\".\"subcount_data_synced\"'."
    )

    dataframe_id: Optional[str] = Field(
        default=None,
        description="The ID of a dataset saved to memory in a previous step (e.g., 'df_a1b2c3'). Use this INSTEAD of TABLE_NAME if the data was already queried, cleaned, or aggregated."
    )
    
    target_variable: str = Field(
        ..., 
        description="The exact column name of the target variable."
    )
    
    feature_variables: List[str] = Field(
        ..., 
        description="A list of exact column names for the feature variables to evaluate against the target."
    )
    
    target_type: Literal["continuous", "discrete"] = Field(
        default="continuous", 
        description="Specify 'continuous' if the target variable is numerical (e.g., spend, MRR), or 'discrete' if it is categorical (e.g., churned yes/no, segment)."
    )


class run_kmeans_clustering_tool_schema(BaseModel):
    """
    Performs K-Means clustering to group data into distinct segments based on feature similarities. 
    Use this to discover customer segments, group similar behaviors, or identify natural groupings in the data.
    """
    TABLE_NAME: Optional[Union[str, List[str]]] = Field(
        ..., 
        description="The exact SQL-safe table name(s) to query, e.g., '\"sandbox\".\"acquisition_data_v3\"', '\"sandbox\".\"dbs_marketing_sync\"', or '\"sandbox\".\"subcount_data_synced\"'."
    )

    dataframe_id: Optional[str] = Field(
        default=None,
        description="The ID of a dataset saved to memory in a previous step (e.g., 'df_a1b2c3'). Use this INSTEAD of TABLE_NAME if the data was already queried, cleaned, or aggregated."
    )
    
    feature_variables: List[str] = Field(
        ..., 
        description="A list of exact column names to use for clustering."
    )
    
    n_clusters: Optional[int] = Field(
        default=3, 
        description="The number of clusters (k) to create. Default is 3."
    )


class run_ols_regression_tool_schema(BaseModel):
    """
    Performs an Ordinary Least Squares (OLS) multiple regression. Cannot perform non-linear regression. 
    Use this when the user asks to analyze the relationship, correlation, or impact of multiple independent numerical variables on a dependent target variable.
    
    CRITICAL MULTI-TABLE RULE: If combining tables with multiple rows per month (like marketing spend and subcounts), do NOT use TABLE_NAME. Instead, use execute_sql_query_tool first to aggregate (SUM) the data by month and filter correctly, then pass the resulting dataframe_id to this tool.
    """
    TABLE_NAME: Optional[Union[str, List[str]]] = Field(
        ..., 
        description="The exact SQL-safe table name(s) to query."
    )

    dataframe_id: Optional[str] = Field(
        default=None,
        description="The ID of a dataset saved to memory in a previous step (e.g., 'df_a1b2c3'). Use this INSTEAD of TABLE_NAME if the data was already queried, cleaned, or aggregated."
    )
    
    where_clause: Optional[str] = Field(
        default=None,
        description="Optional PostgreSQL WHERE clause to filter the data before running the regression (e.g., '\"Metric\" = ''Gross Adds'''). Exclude the 'WHERE' keyword."
    )
    
    dependent_variable: str = Field(
        ..., 
        description="The exact column name of the target numerical variable to predict (the Y variable)."
    )
    
    independent_variables: List[str] = Field(
        ..., 
        description="A list of exact column names for the numerical predictor variables (the X variables)."
    )


class run_pca_tool_schema(BaseModel):
    """
    Performs Principal Component Analysis (PCA) to reduce dimensionality and find the underlying variance/patterns in a set of features. 
    Use this to identify which combinations of variables explain the most variance in the dataset.
    """
    TABLE_NAME: Optional[Union[str, List[str]]] = Field(
        ..., 
        description="The exact SQL-safe table name(s) to query, e.g., '\"sandbox\".\"acquisition_data_v3\"', '\"sandbox\".\"dbs_marketing_sync\"', or '\"sandbox\".\"subcount_data_synced\"'."
    )

    dataframe_id: Optional[str] = Field(
        default=None,
        description="The ID of a dataset saved to memory in a previous step (e.g., 'df_a1b2c3'). Use this INSTEAD of TABLE_NAME if the data was already queried, cleaned, or aggregated."
    )
    
    feature_variables: List[str] = Field(
        ..., 
        description="A list of exact column names to include in the PCA."
    )
    
    n_components: Optional[int] = Field(
        default=None, 
        description="The number of principal components to compute. If omitted, computes components for all features."
    )


class run_neural_network_tool_schema(BaseModel):
    """
    Trains a Multi-Layer Perceptron (MLP) Neural Network for complex non-linear regression or classification.
    Use this for advanced predictive modeling when basic regression or Random Forest is insufficient.

    Returns train + test metrics, a training loss curve, actual-vs-predicted and residual charts
    (regression), a confusion matrix (classification), permutation-based feature importance,
    and architecture feedback (overfitting / convergence warnings).
    Optionally predicts on a new data point when predict_on is supplied.
    """
    TABLE_NAME: Optional[Union[str, List[str]]] = Field(
        default=None,
        description="The exact SQL-safe table name(s) to query, e.g., '\"sandbox\".\"acquisition_data_v3\"'. Omit if passing dataframe_id."
    )
    dataframe_id: Optional[str] = Field(
        default=None,
        description="The ID of a dataset saved to memory in a previous step (e.g., 'df_a1b2c3'). Use this INSTEAD of TABLE_NAME if data was already queried or aggregated."
    )
    target_variable: str = Field(
        ...,
        description="The exact column name of the target variable to predict (e.g., 'npv', 'sac', 'Tactic')."
    )
    feature_variables: List[str] = Field(
        ...,
        description="A list of exact column names for the predictor variables."
    )
    task_type: Literal["regression", "classification"] = Field(
        ...,
        description="'regression' if the target is a continuous number (e.g., npv, sac). 'classification' if the target is a category (e.g., Sales_Channel, Activation_Plan)."
    )
    hidden_layer_sizes: Optional[List[int]] = Field(
        default=[100, 50],
        description=(
            "Neuron count per hidden layer, e.g. [100, 50] = two layers with 100 and 50 neurons. "
            "Larger values increase model capacity but slow training and risk overfitting on small datasets. "
            "Default [100, 50] is a sensible starting point for most problems."
        )
    )
    max_iter: Optional[int] = Field(
        default=500,
        description="Maximum training epochs. The tool uses early stopping, so training may end earlier. Increase if the result text warns that max_iter was reached."
    )
    predict_on: Optional[dict] = Field(
        default=None,
        description=(
            "Optional. A dictionary of feature_name → value for a single new data point you want "
            "to predict on after training. Must include every feature in feature_variables. "
            "Example: {\"Marketing\": -450.0, \"Sales_Channel\": \"Direct\", \"Geobucket\": 3}. "
            "The prediction will be appended to the result text."
        )
    )

class run_optimization_tool_schema(BaseModel):
    """
    Runs general-purpose linear programming optimization (scipy.optimize.linprog) to maximize or
    minimize a custom objective function subject to linear constraints.
    Use this for abstract resource allocation problems where you supply your own coefficients.
    For SAC / marketing budget allocation problems using DISH historical data, prefer
    run_sac_optimization_tool instead — it computes coefficients automatically from the data.
    """
    objective_coefficients: List[float] = Field(
        ...,
        description=(
            "The coefficients of the objective function, one per decision variable "
            "(e.g., [npv_per_dollar_tactic1, npv_per_dollar_tactic2]). "
            "Set maximize=True rather than manually negating values."
        )
    )
    maximize: Optional[bool] = Field(
        default=False,
        description="Set to True to maximize the objective. Default is False (minimize)."
    )
    variable_names: Optional[List[str]] = Field(
        default=None,
        description=(
            "Human-readable labels for each decision variable, in the same order as "
            "objective_coefficients (e.g., ['Direct Mail', 'Digital Search', 'DRTV']). "
            "When provided, all output tables and text use these names instead of 'Variable_1' etc."
        )
    )
    inequality_constraints_matrix: Optional[List[List[float]]] = Field(
        default=None,
        description=(
            "Left-hand side coefficients for ≤ inequality constraints (A_ub). "
            "Each inner list is one constraint row, with one coefficient per decision variable. "
            "Example — total budget ≤ 1,000,000 across 3 tactics: [[1.0, 1.0, 1.0]]."
        )
    )
    inequality_constraints_bounds: Optional[List[float]] = Field(
        default=None,
        description="Right-hand side limits for each inequality constraint row (b_ub)."
    )
    equality_constraints_matrix: Optional[List[List[float]]] = Field(
        default=None,
        description="Left-hand side coefficients for equality constraints (A_eq)."
    )
    equality_constraints_bounds: Optional[List[float]] = Field(
        default=None,
        description="Right-hand side values for each equality constraint (b_eq)."
    )
    bounds: Optional[List[List[Optional[float]]]] = Field(
        default=None,
        description=(
            "Per-variable [min, max] bounds. Use null for no bound. "
            "Example for 3 variables with a $0 floor and no ceiling: "
            "[[0, null], [0, null], [0, null]]."
        )
    )


class run_sac_optimization_tool_schema(BaseModel):
    """
    Data-aware SAC / marketing budget optimizer for DISH TV acquisition economics.
    Automatically queries acquisition_data_v3 and dbs_marketing_sync to compute
    historical efficiency metrics (NPV per activation, cost per activation) for each
    marketing tactic, then solves a linear program to find the spend allocation that
    maximises total projected NPV (or minimises total SAC) within the given budget.

    Use this whenever the user asks questions like:
      - "How should we allocate our $5M marketing budget?"
      - "What is the optimal SAC allocation across tactics?"
      - "Which marketing channels give us the best NPV per dollar?"
      - "Optimize our spend mix to maximize subscriber NPV"
      - "What's the most efficient way to acquire X new subscribers?"

    Returns a plain-English recommendation with tactic-level spend amounts, projected
    activations, projected NPV, historical benchmarks, sensitivity analysis on the
    budget constraint, and a bar chart — all labelled with real tactic names.
    """
    total_budget: float = Field(
        ...,
        description=(
            "The total marketing budget available to allocate across tactics, in dollars. "
            "Example: 5000000.0 for a $5M budget."
        )
    )
    objective: Optional[str] = Field(
        default="npv",
        description=(
            "What to optimize for. "
            "'npv' (default) — maximise total projected net present value across all activations. "
            "'sac' — minimise total subscriber acquisition cost."
        )
    )
    tactic_filters: Optional[List[str]] = Field(
        default=None,
        description=(
            "Optional list of specific marketing tactic names to include. "
            "If omitted, all tactics with sufficient historical data are considered. "
            "Valid tactic names match the 'Tactic' column in acquisition_data_v3: "
            "'direct mail', 'digital', 'TV', 'print', 'OOH/out of home', 'radio', etc. "
            "Use exact casing as it appears in the data."
        )
    )
    min_spend_by_tactic: Optional[dict] = Field(
        default=None,
        description=(
            "Optional minimum spend floor (in dollars) for specific tactics. "
            "Only include tactics that need a floor — omitted tactics default to $0 minimum. "
            "Example: {'direct mail': 200000, 'digital': 500000}"
        )
    )
    max_spend_by_tactic: Optional[dict] = Field(
        default=None,
        description=(
            "Optional maximum spend ceiling (in dollars) for specific tactics. "
            "Omit tactics that have no cap. "
            "Example: {'TV': 1000000}"
        )
    )
    target_activations: Optional[float] = Field(
        default=None,
        description=(
            "Optional exact number of new subscriber activations to hit. "
            "When set, the optimizer adds an equality constraint so the projected "
            "activation count equals this value exactly. "
            "Leave null to let the optimizer find the activation count naturally."
        )
    )
    start_year: Optional[int] = Field(
        default=None,
        description=(
            "Optional start year (inclusive) for the historical data window used to "
            "compute efficiency benchmarks. Example: 2022. "
            "If omitted, all available history is used."
        )
    )
    end_year: Optional[int] = Field(
        default=None,
        description=(
            "Optional end year (inclusive) for the historical data window. Example: 2025."
        )
    )


class run_random_forest_tool_schema(BaseModel):
    """
    Trains a Random Forest machine learning model to predict a target variable based on multiple features. 
    Use this to find non-linear relationships, classify outcomes, or determine the importance/impact of various features.
    """
    TABLE_NAME: Optional[Union[str, List[str]]] = Field(
        ..., 
        description="The exact SQL-safe table name(s) to query, e.g., '\"sandbox\".\"acquisition_data_v3\"', '\"sandbox\".\"dbs_marketing_sync\"', or '\"sandbox\".\"subcount_data_synced\"'."
    )

    dataframe_id: Optional[str] = Field(
        default=None,
        description="The ID of a dataset saved to memory in a previous step (e.g., 'df_a1b2c3'). Use this INSTEAD of TABLE_NAME if the data was already queried, cleaned, or aggregated."
    )
    
    target_variable: str = Field(
        ..., 
        description="The exact column name of the target variable to predict."
    )
    
    feature_variables: List[str] = Field(
        ..., 
        description="A list of exact column names for the predictor variables."
    )
    
    task_type: Literal["regression", "classification"] = Field(
        ..., 
        description="Specify 'regression' if the target variable is numerical/continuous, or 'classification' if the target is categorical/discrete."
    )
    
    n_estimators: Optional[int] = Field(
        default=100, 
        description="The number of trees in the forest (default is 100)."
    )


class run_forecasting_tool_schema(BaseModel):
    """
    Performs Holt-Winters Exponential Smoothing time series forecasting. Automatically resolves
    the correct year/month column names for any table registered in TABLE_DIMENSIONS
    (e.g., acquisition_data_v3, dbs_marketing_sync, subcount_data_synced).
    Use this when the user asks to predict or forecast future values based on historical trends.

    NOTE: For tables with multiple metric rows per month (like subcount_data_synced), use where_clause 
    to filter to a single Metric and Row_Type, or filter via a prior execute_sql_query_tool call and pass the dataframe_id.
    
    MODEL PARAMETER GUIDE:
    - trend / seasonal: use 'add' (additive) when seasonal swings are roughly constant in size
      over time. Use 'mul' (multiplicative) when swings grow proportionally with the level
      of the series (e.g., a metric that doubles each year and whose seasonal spikes also double).
    - seasonal_periods: set to 12 for monthly data (default), 4 for quarterly, 52 for weekly.
    """
    TABLE_NAME: Optional[Union[str, List[str]]] = Field(
        default=None,
        description=(
            "The exact SQL-safe table name(s) to query, e.g., "
            "'\"sandbox\".\"acquisition_data_v3\"', "
            "'\"sandbox\".\"dbs_marketing_sync\"', or "
            "'\"sandbox\".\"subcount_data_synced\"'. "
            "Omit if passing dataframe_id instead."
        )
    )

    dataframe_id: Optional[str] = Field(
        default=None,
        description=(
            "The ID of a dataset saved to memory in a previous step (e.g., 'df_a1b2c3'). "
            "Use this INSTEAD of TABLE_NAME if the data was already queried, filtered, or aggregated "
            "(e.g., after isolating a single Metric from subcount_data_synced)."
        )
    )
    
    where_clause: Optional[str] = Field(
        default=None,
        description="Optional PostgreSQL WHERE clause to filter the data before running the forecast (e.g., '\"Metric\" = ''Gross Adds'''). Exclude the 'WHERE' keyword."
    )

    value_column: str = Field(
        ...,
        description=(
            "The exact column name of the numerical variable to forecast "
            "(e.g., 'amount', 'mcf', 'sac', 'temp_Id')."
        )
    )

    aggregation: Optional[Literal["SUM", "AVG", "COUNT"]] = Field(
        default="SUM",
        description=(
            "The aggregation function applied to value_column per month before fitting the model. "
            "Use SUM for totals (e.g., total spend), AVG for averages, COUNT for volume. Default is SUM."
        )
    )

    steps: Optional[int] = Field(
        default=6,
        description="The number of future periods (months) to forecast ahead. Default is 6."
    )

    trend: Optional[Literal["add", "mul"]] = Field(
        default="add",
        description=(
            "The trend component type. "
            "'add' (additive) — use when the trend grows/shrinks by a roughly constant amount each period. "
            "'mul' (multiplicative) — use when the trend grows/shrinks proportionally (exponentially). "
            "Default is 'add'."
        )
    )

    seasonal: Optional[Literal["add", "mul"]] = Field(
        default="add",
        description=(
            "The seasonal component type. "
            "'add' (additive) — use when seasonal fluctuations are roughly constant regardless of the series level. "
            "'mul' (multiplicative) — use when seasonal swings scale with the level of the series. "
            "Default is 'add'."
        )
    )

    seasonal_periods: Optional[int] = Field(
        default=12,
        description=(
            "The number of periods in one full seasonal cycle. "
            "12 for monthly data (default), 4 for quarterly, 52 for weekly."
        )
    )


class ScenarioChange(BaseModel):
    """A single hypothetical change to a feature variable."""
    column_name: str = Field(
        ..., 
        description="The exact name of the feature column to modify (e.g., 'Marketing_Spend')."
    )
    new_value: float = Field(
        ..., 
        description="The new hypothetical numerical value for this column (e.g., 50000.0)."
    )
class run_scenario_planning_tool_schema(BaseModel):
    """
    Performs statistical what-if scenario planning and simulations using OLS regression. 
    Use this tool whenever the user asks:
    - What would happen to a target variable (Z) if a feature (X) changes by a certain percentage or to a specific value.
    - Questions containing phrases like "what if", "assume X is", "increase/decrease by X%", or "hold Y constant".
    This tool automatically computes baseline averages, applies the hypothetical changes, holds specified control variables constant at their historical means, and returns expected predictions with 95% confidence intervals.
    """
    TABLE_NAME: Optional[Union[str, List[str]]] = Field(
        ..., 
        description="The exact SQL-safe table name(s) to query, e.g., '\"sandbox\".\"acquisition_data_v3\"', '\"sandbox\".\"dbs_marketing_sync\"', or '\"sandbox\".\"subcount_data_synced\"'."
    )

    dataframe_id: Optional[str] = Field(
        default=None,
        description="The ID of a dataset saved to memory in a previous step (e.g., 'df_a1b2c3'). Use this INSTEAD of TABLE_NAME if the data was already queried, cleaned, or aggregated."
    )
    
    target_variable: str = Field(
        ..., 
        description="The exact column name of the target variable to predict (the Z variable)."
    )
    
    feature_variables: List[str] = Field(
        ..., 
        description="A list of all relevant predictor columns to include in the model (both the variables being changed AND the variables being held constant)."
    )
    
    # Updated: Replaced dict[str, float] with List[ScenarioChange] to satisfy strict JSON Schema rules
    scenario_changes: List[ScenarioChange] = Field(
        ..., 
        description="A list of specific feature columns and their new hypothetical values."
    )
    
    confidence_level: Optional[float] = Field(
        default=0.95, 
        description="The statistical confidence level for the prediction interval (default is 0.95 for a 95% interval)."
    )


# -------------------- TRANSFORMATIONS SCHEMAS --------------------
class execute_sql_query_tool_schema(BaseModel):
    """
    Queries the Databricks database using PostgreSQL syntax. Because this is PostgreSQL, you MUST wrap all column names in double quotes to preserve exact capitalization. 
    Do NOT use this tool if a more specific tool is available for the user's request:
    - Use calculate_unit_economics_tool for CPA or CLV questions.
    - Use run_scenario_planning_tool for ANY what-if analysis, simulating changes to variables, or holding variables constant.
    - Use regression/forecasting/clustering tools for modeling.
    Do NOT attempt to write complex SQL window functions, regressions, or simulations manually if a tool exists for it.
    IMPORTANT — joining subcount_data_synced: this table has ~18 rows per month (one per Metric). 
    You MUST filter by "Metric" and/or "Row_Type" BEFORE or WITHIN any join to avoid row multiplication. 
    For example: JOIN ... ON year/month AND "subcount_data_synced"."Metric" = 'Ending Period Subscribers'.
    Never join this table without a Metric or Row_Type filter in the ON or WHERE clause.
    if you get a 'column does not exist' error, make sure to go back and re-read the data dictionary JSON schema before rewriting your query.
    """
    sql_query: str = Field(
        ..., 
        description="The raw PostgreSQL query to execute."
    )


class join_dataframes_tool_schema(BaseModel):
    """
    Joins (merges) two previously saved in-memory DataFrames together based on shared columns.
    Use this when you need to combine data from two different tables that you have already queried.
    """
    left_dataframe_id: str = Field(
        ..., 
        description="The ID of the first (left) dataset saved to memory (e.g., 'df_a1b2c3')."
    )
    right_dataframe_id: str = Field(
        ..., 
        description="The ID of the second (right) dataset saved to memory."
    )
    how: Literal["inner", "left", "right", "outer", "cross"] = Field(
        default="inner",
        description="The type of merge to be performed. 'inner' is standard."
    )
    left_on: List[str] = Field(
        ...,
        description="A list of exact column names from the left DataFrame to join on."
    )
    right_on: List[str] = Field(
        ...,
        description="A list of exact column names from the right DataFrame to join on."
    )


class pivot_dataframe_tool_schema(BaseModel):
    """
    Reshapes an in-memory DataFrame from long format to wide format using pandas pivot_table.
    Use this when long-format metrics (like subcount or P&L rows) need to become columns side-by-side.
    """
    dataframe_id: str = Field(
        ..., 
        description="The ID of the dataset saved in memory (e.g., 'df_a1b2c3')."
    )
    index_columns: List[str] = Field(
        ..., 
        description="Column name(s) to use as the row headers (e.g., ['Year', 'Month'])."
    )
    pivot_column: str = Field(
        ..., 
        description="The column whose unique values will become the new table headers (e.g., 'Metric')."
    )
    value_column: str = Field(
        ..., 
        description="The numerical column providing values for the new table cells (e.g., 'Amount')."
    )
    aggregation: Literal["SUM", "AVG", "COUNT", "MAX", "MIN"] = Field(
        default="SUM",
        description="How to aggregate duplicate entries if multiple rows share index/pivot keys. Default is SUM."
    )


class calculate_cpa_tool_schema(BaseModel):
    """
    Used to calculate marketing cost per acquistion (CPA) ratio.
    calculates three ratios: overall cpa, residential cpa, and residential non-caliber cpa.
    use this whenever the user asks about cpas, sac per add, cost per acquisition, marketing efficiency, or other related terms.
    """
    marketing_where_clause: Optional[str] = Field(
        default=None, 
        description="Optional. A PostgreSQL WHERE clause to filter the marketing spend table (e.g., '\"account\" = ''611010'''). Exclude the 'WHERE' keyword."
    )
    
    subscriber_where_clause: Optional[str] = Field(
        default=None, 
        description="A PostgreSQL WHERE clause to filter the subscriber table ('\"Metric\" = 'Gross Adds'). Exclude the 'WHERE' keyword."
    )


class calculate_ratio_tool_schema(BaseModel):
    """
    Calculates a monthly ratio between any two numeric columns across one or two tables or saved dataframes.
    Returns a DataFrame with year, month, the two source values, and the computed ratio
    (numerator / denominator) for every period where both values are present.
    """
    numerator_column: str = Field(
        ...,
        description="The exact column name to use as the numerator of the ratio (e.g., 'Total_Spend')."
    )
    numerator_table: Optional[str] = Field(
        default=None,
        description="The exact SQL-safe table name containing the numerator column (e.g., '\"sandbox\".\"dbs_marketing_sync\"'). Use this if pulling directly from the database."
    )
    numerator_dataframe_id: Optional[str] = Field(
        default=None,
        description="The ID of a dataset saved to memory in a previous step containing the numerator (e.g., 'df_a1b2c3'). Use this INSTEAD of numerator_table if data is already in memory."
    )
    denominator_column: str = Field(
        ...,
        description="The exact column name to use as the denominator of the ratio (e.g., 'Activations')."
    )
    denominator_table: Optional[str] = Field(
        default=None,
        description="The exact SQL-safe table name containing the denominator column."
    )
    denominator_dataframe_id: Optional[str] = Field(
        default=None,
        description="The ID of a dataset saved to memory containing the denominator."
    )
    where_clause: Optional[str] = Field(
        default=None,
        description="Optional. A PostgreSQL WHERE clause applied when fetching columns from a database (e.g., '\"year\" = 2024'). Exclude the 'WHERE' keyword."
    )
    numerator_aggregation: Optional[str] = Field(
        default="SUM",
        description="SQL aggregation to apply to the numerator column when grouping by month. One of SUM, AVG, COUNT. Defaults to SUM."
    )
    denominator_aggregation: Optional[str] = Field(
        default="SUM",
        description="SQL aggregation to apply to the denominator column when grouping by month. One of SUM, AVG, COUNT. Defaults to SUM."
    )


class execute_python_tool_schema(BaseModel):
    """
    Executes raw Python code generated by the LLM in a secure, sandboxed environment.
    Use this as a fallback for complex analytics, custom data manipulation, or advanced mathematical operations that pre-built tools cannot handle.

    DATA LOADING RULES — read carefully:
    - Single table: 'df' is a plain pandas DataFrame containing up to 100,000 rows from that table.
    - Multiple tables: 'df' is a Python LIST of DataFrames, one per table in the order provided
      (e.g. df[0] = first table, df[1] = second table). Each table is loaded independently —
      NO pre-join is performed. Use pandas (pd.merge) to join them yourself inside the code.

    The code is executed with access to pandas (pd) and numpy (np).
    To return data to the LLM, assign the final text output to 'result_text' and any resulting DataFrame to 'result_df'.
    """
    TABLE_NAME: Optional[Union[str, List[str]]] = Field(
        ..., 
        description="The exact SQL-safe table name(s) to query and load into the 'df' variable, e.g., '\"sandbox\".\"acquisition_data_v3\"'."
    )

    dataframe_id: Optional[str] = Field(
        default=None,
        description="The ID of a dataset saved to memory in a previous step. Use this INSTEAD of TABLE_NAME if the data was already queried."
    )
    
    code: str = Field(
        ..., 
        description=(
            "The Python code to execute. Must be valid Python. "
            "Only use the pre-loaded pandas (pd) and numpy (np) libraries. "
            "Do NOT import outside libraries like os, sys, subprocess, requests, or database drivers. "
            "Do NOT attempt to run SQL mutation commands. "
            "Use the pre-loaded 'df' variable."
        )
    )


#----------------------------VISUALS SCHEMAS----------------------------
class generate_barchart_tool_schema(BaseModel):
    """
    Generates a bar chart to compare aggregated numerical values across categorical groups or time periods.
    Automatically handles pre-aggregation (SUM, AVG, COUNT) to ensure clean visualizations.
    """
    TABLE_NAME: Optional[Union[str, List[str]]] = Field(
        ..., 
        description="The exact SQL-safe table name(s) to query, e.g., '\"sandbox\".\"acquisition_data_v3\"', '\"sandbox\".\"dbs_marketing_sync\"', or '\"sandbox\".\"subcount_data_synced\"'."
    )

    dataframe_id: Optional[str] = Field(
        default=None,
        description="The ID of a dataset saved to memory in a previous step (e.g., 'df_a1b2c3'). Use this INSTEAD of TABLE_NAME if the data was already queried, cleaned, or aggregated."
    )
    x_column: str = Field(
        ..., 
        description="The exact column name for the X-axis (usually categorical or dates)."
    )
    y_column: str = Field(
        ..., 
        description="The exact column name for the Y-axis numerical value to measure."
    )
    category_column: Optional[str] = Field(
        default=None, 
        description="Optional column name to group or color-code side-by-side bars."
    )
    where_clause: Optional[str] = Field(
        default=None, 
        description="Optional PostgreSQL WHERE clause to filter data before plotting."
    )
    aggregation: Optional[Literal["SUM", "AVG", "COUNT", "MAX", "MIN", "NONE"]] = Field(
        default="SUM",
        description="The aggregation function applied to the Y-axis variable per X-axis group. Default is SUM."
    )


class generate_histogram_tool_schema(BaseModel):
    """
    Generates a histogram with an executive box-plot marginal to visualize data distributions, spread, and outliers.
    """
    TABLE_NAME: Optional[Union[str, List[str]]] = Field(
        ..., 
        description="The exact SQL-safe table name(s) to query, e.g., '\"sandbox\".\"acquisition_data_v3\"', '\"sandbox\".\"dbs_marketing_sync\"', or '\"sandbox\".\"subcount_data_synced\"'."
    )

    dataframe_id: Optional[str] = Field(
        default=None,
        description="The ID of a dataset saved to memory in a previous step (e.g., 'df_a1b2c3'). Use this INSTEAD of TABLE_NAME if the data was already queried, cleaned, or aggregated."
    )
    x_column: str = Field(
        ..., 
        description="The exact column name of the numerical variable to analyze."
    )
    n_bins: Optional[int] = Field(
        default=None, 
        description="Optional. The number of frequency bins to divide the data into."
    )
    category_column: Optional[str] = Field(
        default=None, 
        description="Optional column name to overlay multiple distribution cohorts."
    )
    where_clause: Optional[str] = Field(
        default=None, 
        description="Optional PostgreSQL WHERE clause."
    )


class generate_linechart_tool_schema(BaseModel):
    """
    Generates a continuous line chart to visualize trends over time or sequences.
    Automatically groups duplicate timestamps and sorts chronologically to prevent erratic line jumps.
    """
    TABLE_NAME: Optional[Union[str, List[str]]] = Field(
        ..., 
        description="The exact SQL-safe table name(s) to query, e.g., '\"sandbox\".\"acquisition_data_v3\"', '\"sandbox\".\"dbs_marketing_sync\"', or '\"sandbox\".\"subcount_data_synced\"'."
    )

    dataframe_id: Optional[str] = Field(
        default=None,
        description="The ID of a dataset saved to memory in a previous step (e.g., 'df_a1b2c3'). Use this INSTEAD of TABLE_NAME if the data was already queried, cleaned, or aggregated."
    )
    x_column: str = Field(
        ..., 
        description="The exact column name for the X-axis time, date, or sequential sequence."
    )
    y_column: str = Field(
        ..., 
        description="The exact column name for the Y-axis numerical variable."
    )
    category_column: Optional[str] = Field(
        default=None, 
        description="Optional column name to plot multiple colored trend lines simultaneously."
    )
    where_clause: Optional[str] = Field(
        default=None, 
        description="Optional PostgreSQL WHERE clause."
    )
    aggregation: Optional[Literal["SUM", "AVG", "COUNT", "MAX", "MIN", "NONE"]] = Field(
        default="SUM",
        description="The aggregation applied if multiple records share the same X-axis timestamp. Default is SUM."
    )


class generate_scatterplot_tool_schema(BaseModel):
    """
    Generates an interactive scatterplot to explore relationships between two numerical variables.
    Supports querying a single table or automatically joining multiple tables.
    """
    TABLE_NAME: Optional[Union[str, List[str]]] = Field(
        ..., 
        description="The exact SQL-safe table name(s) to query, e.g., '\"sandbox\".\"acquisition_data_v3\"', '\"sandbox\".\"dbs_marketing_sync\"', or '\"sandbox\".\"subcount_data_synced\"'."
    )

    dataframe_id: Optional[str] = Field(
        default=None,
        description="The ID of a dataset saved to memory in a previous step (e.g., 'df_a1b2c3'). Use this INSTEAD of TABLE_NAME if the data was already queried, cleaned, or aggregated."
    )
    x_column: str = Field(
        ..., 
        description="The exact column name for the X-axis numerical variable."
    )
    y_column: str = Field(
        ..., 
        description="The exact column name for the Y-axis numerical variable."
    )
    category_column: Optional[str] = Field(
        default=None, 
        description="Optional column name to color-code and segment the scatter points."
    )
    where_clause: Optional[str] = Field(
        default=None, 
        description="Optional PostgreSQL WHERE clause (exclude the 'WHERE' keyword)."
    )
    include_trendline: Optional[bool] = Field(
        default=False,
        description="Set to True to overlay an Ordinary Least Squares (OLS) trendline."
    )