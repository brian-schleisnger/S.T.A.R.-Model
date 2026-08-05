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

from agent.categories import CATEGORY_REGISTRY
from toolkit import TOOL_DISPATCHER


def _first_sentence(cls) -> str:
    """Extracts and normalises the first sentence of a class docstring."""
    doc = textwrap.dedent(cls.__doc__ or "").strip()
    # Split on the first period that ends a sentence (followed by whitespace or end-of-string)
    sentence = doc.split(".")[0].strip()
    return sentence if sentence else cls.__name__


def _build_enriched_registry() -> dict:
    enriched = {}
    for cat, data in CATEGORY_REGISTRY.items():
        enriched_tools = {}
        for tool_name, fallback_desc in data["tools"].items():
            # Safely get the tuple from the dispatcher, or None if not found
            dispatcher_entry = TOOL_DISPATCHER.get(tool_name)
            schema_cls = dispatcher_entry[1] if dispatcher_entry else None
            
            enriched_tools[tool_name] = (
                _first_sentence(schema_cls) if schema_cls else fallback_desc
            )
        enriched[cat] = {**data, "tools": enriched_tools}
    return enriched


# The enriched registry is built once at import time.
# Import this instead of CATEGORY_REGISTRY wherever tool descriptions are rendered.
ENRICHED_CATEGORY_REGISTRY: dict = _build_enriched_registry()
