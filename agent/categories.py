# ─── CATEGORY_REGISTRY ───────────────────────────────────────────────────────
#
# This is the authoritative source for:
#   • category routing rules  ("rule" key)
#   • which tools belong to each category  (keys of "tools" dict)
#
# The short tool description strings below are PLACEHOLDERS only.
# At runtime, agent/tool_descriptions.py replaces them with the first sentence
# of each tool's class docstring in agent/schemas.py.
#
# To change a tool's routing description → edit its class docstring in schemas.py.
# To change which tools belong to a category → edit the "tools" keys here.
# To change a category's selection rule → edit the "rule" string here.
#
CATEGORY_REGISTRY: dict[str, dict] = {
    "STATISTICAL_MODELING": {
        "rule": "linear relationships, impact analysis, or dimensionality/segmentation work.",
        "tools": {
            "run_ols_regression_tool":           "(see schemas.py docstring)",
            "run_pca_tool":                      "(see schemas.py docstring)",
            "run_kmeans_clustering_tool":        "(see schemas.py docstring)",
            "calculate_mutual_information_tool": "(see schemas.py docstring)",
        }
    },
    "ML_MODELING": {
        "rule": "non-linear predictive modeling where the goal is to predict a target variable or measure feature importance.",
        "tools": {
            "run_random_forest_tool":  "(see schemas.py docstring)",
            "run_neural_network_tool": "(see schemas.py docstring)",
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
            "run_sac_optimization_tool": "(see schemas.py docstring)",
            "run_optimization_tool":     "(see schemas.py docstring)",
        }
    },
    "FORECASTING_AND_SCENARIOS": {
        "rule": "predicting future values or answering hypothetical what-if style questions",
        "tools": {
            "run_forecasting_tool":       "(see schemas.py docstring)",
            "run_scenario_planning_tool": "(see schemas.py docstring)",
        }
    },
    "SQL_RETRIEVAL": {
        "rule": "simple data lookup, filtering, counting, summing, or averaging with no modeling, no chart, and no cross-table unit economics.",
        "tools": {
            "execute_sql_query_tool": "(see schemas.py docstring)",
        }
    },
    "DATA_TRANSFORMATION": {
        "rule": "pivot, combine, join, or merge data from two different sources.",
        "tools": {
            "join_dataframes_tool":  "(see schemas.py docstring)",
            "pivot_dataframe_tool":  "(see schemas.py docstring)",
        }
    },
    "RATIO_ANALYSIS": {
        "rule": "any question about CPA (cost per acquisition) or any ratio analysis that needs to be calculated",
        "tools": {
            "calculate_cpa_tool":   "(see schemas.py docstring)",
            "calculate_ratio_tool": "(see schemas.py docstring)",
        }
    },
    "CUSTOM_PYTHON": {
        "rule": "complex multi-step analytics that combine multiple tables or operations no single tool handles.",
        "tools": {
            "execute_python_tool": "(see schemas.py docstring)",
        }
    },
    "VISUALIZATION": {
        "rule": "explicitly asks for a chart, graph, or plot.",
        "tools": {
            "generate_barchart_tool":    "(see schemas.py docstring)",
            "generate_linechart_tool":   "(see schemas.py docstring)",
            "generate_scatterplot_tool": "(see schemas.py docstring)",
            "generate_histogram_tool":   "(see schemas.py docstring)",
        }
    },
}

# Auto-derive CATEGORY_TOOLS directly from the registry
CATEGORY_TOOLS: dict[str, list[str]] = {
    cat: list(data["tools"].keys()) 
    for cat, data in CATEGORY_REGISTRY.items()
}