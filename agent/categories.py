CATEGORY_REGISTRY: dict[str, dict] = {
    "STATISTICAL_MODELING": {
        "rule": "linear relationships, impact analysis, or dimensionality/segmentation work.",
        "tools": {
            "run_ols_regression_tool": "linear relationships / impact of X on Y.",
            "run_pca_tool": "dimensionality reduction / variance decomposition.",
            "run_kmeans_clustering_tool": "segmentation / natural groupings.",
            "calculate_mutual_information_tool": "mutual information and information theory."
        }
    },
    "ML_MODELING": {
        "rule": "non-linear predictive modeling where the goal is to predict a target variable or measure feature importance.",
        "tools": {
            "run_random_forest_tool": "non-linear prediction / feature importance.",
            "run_neural_network_tool": "complex non-linear modeling.",
        }
    },
    "SAC_OPTIMIZATION": {
        "rule": (
            "budget allocation, spend optimization, finding the best mix of marketing tactics, "
            "maximizing NPV or minimizing SAC under a budget constraint, or any question about "
            "how to distribute a marketing or acquisition budget. "
            "Use run_sac_optimization_tool when the question involves DISH historical data. "
            "Use run_optimization_tool for abstract LP problems where the user supplies their own numbers."
        ),
        "tools": {
            "run_sac_optimization_tool": (
                "data-driven optimal marketing budget allocation using historical DISH acquisition economics "
                "(NPV per activation, cost per activation by tactic). "
                "Handles total budget ceiling, per-tactic min/max spend floors, and optional activation targets."
            ),
            "run_optimization_tool": (
                "general-purpose linear programming for abstract optimization problems "
                "where the user provides their own objective coefficients and constraints."
            ),
        }
    },
    "FORECASTING_AND_SCENARIOS": {
        "rule": "predicting future values or answering hypothetical what-if style questions",
        "tools": {
            "run_forecasting_tool": "time-series prediction of future values.",
            "run_scenario_planning_tool": "what-if analysis, simulate variable changes, confidence intervals."
        }
    },

    "SQL_RETRIEVAL": {
        "rule": "simple data lookup, filtering, counting, summing, or averaging with no modeling, no chart, and no cross-table unit economics.",
        "tools": {
            "execute_sql_query_tool": "simple lookup, filter, or aggregation (SUM/AVG/COUNT/GROUP BY)."
        }
    },
    "DATA_TRANSFORMATION": {
        "rule": "pivot, combine, join, or merge data from two different sources.",
        "tools": {
            "join_dataframes_tool": "merges two previously saved dataframes together on specific columns.",
            "pivot_dataframe_tool": "reshape long-form tables into wide side by side format."
        }
    },
    "RATIO_ANALYSIS": {
        "rule": "any question about CPA (cost per acquisition) or any ratio analysis that needs to be calculated",
        "tools": {
            "calculate_cpa_tool": "CPA, marketing efficiency.",
            "calculate_ratio_tool": "monthly ratio or rate between two metrics over time."
        }
    },
    "CUSTOM_PYTHON": {
        "rule": "complex multi-step analytics that combine multiple tables or operations no single tool handles.",
        "tools": {
            "execute_python_tool": "multi-step or cross-table analysis no single tool can handle."
        }
    },
    
    "VISUALIZATION": {
        "rule": "explicitly asks for a chart, graph, or plot.",
        "tools": {
            "generate_barchart_tool": "compare across categories/time.",
            "generate_linechart_tool": "trends over time.",
            "generate_scatterplot_tool": "relationship between two numeric variables.",
            "generate_histogram_tool": "distribution / outliers."
        }
    },
}

# Auto-derive CATEGORY_TOOLS directly from the registry
CATEGORY_TOOLS: dict[str, list[str]] = {
    cat: list(data["tools"].keys()) 
    for cat, data in CATEGORY_REGISTRY.items()
}