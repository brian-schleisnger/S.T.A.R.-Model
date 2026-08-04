"""
tool_descriptions.py — Single assembly point for tool routing metadata.

Import chain (no cycles):
    categories.py   (leaf — defines CATEGORY_REGISTRY structure with placeholder descriptions)
        ↓
    schemas.py      (imports categories for SubQuestion field description)
        ↓
    tool_descriptions.py  (imports both; replaces placeholder descriptions with
                           first sentences derived from each schema class's docstring)
        ↓
    loop.py         (imports ENRICHED_CATEGORY_REGISTRY from here instead of categories)

To update a tool's short routing description: edit its class docstring in schemas.py.
The one-liner used in routing prompts is auto-derived from the first sentence of that docstring.
"""

import textwrap
from typing import Dict

import agent.schemas as schemas
from agent.categories import CATEGORY_REGISTRY


def _first_sentence(cls) -> str:
    """Extracts and normalises the first sentence of a class docstring."""
    doc = textwrap.dedent(cls.__doc__ or "").strip()
    # Split on the first period that ends a sentence (followed by whitespace or end-of-string)
    sentence = doc.split(".")[0].strip()
    return sentence if sentence else cls.__name__


# Map every tool name used in CATEGORY_REGISTRY to its schema class in schemas.py.
# When you add a new tool: add its schema class to schemas.py and add one line here.
_TOOL_SCHEMA_MAP: Dict[str, type] = {
    # Statistical Modeling
    "run_ols_regression_tool":              schemas.run_ols_regression_tool,
    "run_pca_tool":                         schemas.run_pca_tool,
    "run_kmeans_clustering_tool":           schemas.run_kmeans_clustering_tool,
    "calculate_mutual_information_tool":    schemas.calculate_mutual_information_tool,
    # ML Modeling
    "run_random_forest_tool":               schemas.run_random_forest_tool,
    "run_neural_network_tool":              schemas.run_neural_network_tool,
    # SAC Optimization
    "run_sac_optimization_tool":            schemas.run_sac_optimization_tool,
    "run_optimization_tool":               schemas.run_optimization_tool,
    # Forecasting & Scenarios
    "run_forecasting_tool":                 schemas.run_forecasting_tool,
    "run_scenario_planning_tool":           schemas.run_scenario_planning_tool,
    # SQL Retrieval
    "execute_sql_query_tool":              schemas.execute_sql_query_tool,
    # Data Transformation
    "join_dataframes_tool":                schemas.join_dataframes_tool,
    "pivot_dataframe_tool":                schemas.pivot_dataframe_tool,
    # Ratio Analysis
    "calculate_cpa_tool":                  schemas.calculate_cpa_tool,
    "calculate_ratio_tool":                schemas.calculate_ratio_tool,
    # Custom Python
    "execute_python_tool":                 schemas.execute_python_tool,
    # Visualization
    "generate_barchart_tool":              schemas.generate_barchart_tool,
    "generate_linechart_tool":             schemas.generate_linechart_tool,
    "generate_scatterplot_tool":           schemas.generate_scatterplot_tool,
    "generate_histogram_tool":             schemas.generate_histogram_tool,
}


def _build_enriched_registry() -> dict:
    """
    Returns a copy of CATEGORY_REGISTRY where every tool's description string
    is replaced with the first sentence of the corresponding schema class docstring.

    Tools not found in _TOOL_SCHEMA_MAP retain their original description as a
    fallback so the registry stays functional even if the map falls behind.
    """
    enriched = {}
    for cat, data in CATEGORY_REGISTRY.items():
        enriched_tools = {}
        for tool_name, fallback_desc in data["tools"].items():
            schema_cls = _TOOL_SCHEMA_MAP.get(tool_name)
            enriched_tools[tool_name] = (
                _first_sentence(schema_cls) if schema_cls else fallback_desc
            )
        enriched[cat] = {**data, "tools": enriched_tools}
    return enriched


# The enriched registry is built once at import time.
# Import this instead of CATEGORY_REGISTRY wherever tool descriptions are rendered.
ENRICHED_CATEGORY_REGISTRY: dict = _build_enriched_registry()
