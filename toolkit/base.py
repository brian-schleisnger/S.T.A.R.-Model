import copy
import json
import os
from pathlib import Path
import ssl
import logging
from functools import lru_cache


from databricks.sdk import WorkspaceClient
import instructor
import mlflow
from openai import OpenAI
import pandas as pd
import pg8000
from pydantic import BaseModel
import sqlalchemy as sa

from .validators import validate_read_only_sql, _extract_text_content, SecurityViolationError
from agent.context import SessionContext

logger = logging.getLogger(__name__)


# ─── Configuration ───────────────────────────────────────────────

# !! ADD YOUR MODEL ENDPOINTS HERE !!
# Keys are the display names shown in the UI dropdown.
# Values are the Databricks serving endpoint names passed to the OpenAI-compatible API.
AVAILABLE_MODELS: dict[str, str] = {
    "GPT 5.4 Nano (Low)": "system.ai.gpt-5-4-nano",
    "Gemini Flash Lite (Low-Medium)": "system.ai.gemini-3-1-flash-lite",
    "Gemini Flash (Medium)": "system.ai.gemini-3-5-flash",
    "Claude Sonet (Medium-High)": "system.ai.claude-sonnet-5",
    "Claude Opus (High)": "system.ai.claude-opus-4-8"
}

#the cost in dbus per 1 million tokens. ex: gpt5.4 nano costs 2.857 dbus per 1 million input tokens, and 17.857 dbus per 1 million output tokens
#pulled fom databrick's model endpoints page
MODEL_DBUS = pd.DataFrame({"Model": ["system.ai.gpt-5-4-nano","system.ai.gemini-3-1-flash-lite","system.ai.gemini-3-5-flash","system.ai.claude-sonnet-5","system.ai.claude-opus-4-8"],
                           "inputs": [2.857,6.428,21.2485,28.5714,71.429],
                           "Outputs": [17.857,38.572,128.571,142.857,357.143]})

#databricks AI's are estimated to cost about 7 cents per dbu
#this is online and just an estimate
DBU_COST = .07

def calculate_cost(model_endpoint: str, input_tokens: int, output_tokens: int) -> float:
    """
    Estimates the dollar cost of a query given token counts and the active model endpoint.

    Looks up the model's DBU rates per million tokens from MODEL_DBUS, multiplies by
    DBU_COST ($/DBU), and returns the total estimated cost in dollars.
    Returns 0.0 if the model is not found in MODEL_DBUS.
    """
    row = MODEL_DBUS[MODEL_DBUS["Model"] == model_endpoint]
    if row.empty:
        return 0.0
    input_dbus_per_m = row["inputs"].iloc[0]
    output_dbus_per_m = row["Outputs"].iloc[0]
    input_cost = (input_tokens / 1_000_000) * input_dbus_per_m * DBU_COST
    output_cost = (output_tokens / 1_000_000) * output_dbus_per_m * DBU_COST
    return input_cost + output_cost

        
DATABRICKS_HOST = os.environ.get('DATABRICKS_HOST', '').rstrip('/')
PGHOST = os.environ.get("PGHOST")
PGDATABASE = "databricks_postgres"


@lru_cache(maxsize=1)
def get_workspace_client() -> WorkspaceClient:
    """
    Returns the singleton Databricks WorkspaceClient, initialized lazily on first use.
    Using @lru_cache means the SDK only authenticates once per process, but the module
    can be safely imported in environments without Databricks credentials (e.g. tests,
    local dev) without raising an error at import time.
    """
    return WorkspaceClient()


def get_databricks_host() -> str:
    """Returns the Databricks host URL, derived lazily from the SDK client config."""
    return get_workspace_client().config.host


# Single source of truth for the AI gateway base URL.
# Evaluated lazily so it doesn't trigger SDK auth at import time.
def _get_ai_base_url() -> str:
    return f"{get_databricks_host()}/ai-gateway/mlflow/v1"


def get_auth_token() -> str:
    """Dynamically fetches a fresh token from the Databricks SDK on every call."""
    auth_headers = get_workspace_client().config.authenticate()
    return auth_headers["Authorization"].split(" ")[1]

def _make_fresh_openai_client() -> OpenAI:
    """
    Single factory for all OpenAI client construction.
    Always fetches a fresh token so short-lived Databricks tokens never go stale.
    Used by both DynamicOpenAIClient and llm_call so token rotation and the
    base URL are defined in exactly one place.
    """
    return OpenAI(
        api_key=get_auth_token(),
        base_url=_get_ai_base_url(),
        max_retries=0
    )

class DynamicOpenAIClient:
    """
    Proxy wrapper that ensures every API call uses a freshly rotated token.
    Intercepts .chat.completions.create to strip 'strict: True' from tool
    definitions, which Claude rejects.
    """
    def __getattr__(self, name):
        """
        Intercepts all attribute access. Builds a fresh OpenAI client on every
        call (token rotation), then returns either a ChatProxy (for .chat) or
        the raw attribute from the underlying client for everything else.
        """
        client = _make_fresh_openai_client()

        if name == "chat":
            class ChatProxy:
                """Wraps the OpenAI chat namespace to allow pre-call sanitization."""
                @property
                def completions(self):
                    """Returns a CompletionsProxy that strips model-incompatible fields."""
                    class CompletionsProxy:
                        """Sanitizes tool definitions before forwarding to the real client."""
                        def create(self, *args, **kwargs):
                            model = kwargs.get("model", "")

                            # Claude and Gemini both reject 'strict: True' in tool definitions
                            if ("claude" in model.lower() or "gemini" in model.lower()) and "tools" in kwargs:
                                safe_tools = copy.deepcopy(kwargs["tools"])
                                for tool in safe_tools:
                                    if "function" in tool and "strict" in tool["function"]:
                                        del tool["function"]["strict"]
                                kwargs["tools"] = safe_tools

                            return client.chat.completions.create(*args, **kwargs)
                    return CompletionsProxy()
            return ChatProxy()

        return getattr(client, name)

# Singleton proxy — imported by loop.py and cache.py for all tool-routing,
# synthesis, and embedding calls.
raw_client = DynamicOpenAIClient()

mlflow.openai.autolog()

# ─── Conformed Dimension Mapping ─────────────────────────────────
# Keys ('year', 'month', 'day') are standardized conceptual dimensions.
# Values are the actual column names used inside that specific table.
TABLE_DIMENSIONS = {
    '"sandbox"."dbs_marketing_sync"': {
        "year": "Year",
        "month": "Month",
    },
    '"sandbox"."acquisition_data_v3"': {
        "year": "Activation_Year",
        "month": "Activation_Month",
    },
    '"sandbox"."subcount_data_synced"': {
        "year": "Year",
        "month": "Month",
    },
    '"sandbox"."sales_data_sync"': {
        "year": "year",
        "month": "month",
    },
    '"sandbox"."dbspl_sync"': {
        "year": "Year",
        "month": "Month",
    }
}

# ─── Schema Context ──────────────────────────────────────────────
SCHEMA_CONFIG = {
    '"sandbox"."acquisition_data_v3"': "acquisition_data_dictionary.json",
    '"sandbox"."dbs_marketing_sync"': "marketing_spend_dictionary.json",
    '"sandbox"."subcount_data_synced"': "subscriber_count_dictionary.json",
    '"sandbox"."sales_data_sync"': "sales_dictionary.json",
    '"sandbox"."dbspl_sync"': "dish_pl_dictionary.json"
}
'''
ALIASES = {
    '"sandbox"."acquisition_data_v3"': ['MOonthly cash flow', 'sac', 'subscribers', 'per-customer', 'economic', 'mcf', 'npv', 'subscriber acquisition cost', 'lifetime value', 'clv'],
    '"sandbox"."dbs_marketing_sync"': ['marketing', 'spend', 'budget', 'cpa', 'tactic', 'digital', 'tv', 'cost per acquisition', 'ad spend', 'campaign', 'media', 'advertising'],
    '"sandbox"."subcount_data_synced"': ['subscribers', 'subscriber count', 'subscriber balance', 'gross adds', 'net adds', 'disconnects', 'churn rate', 'beginning subscribers', 'ending subscribers', 'local retail', 'sales partner', 'national retail', 'telco activations', 'indirect activations', 'direct activations', 'commercial activations', 'subscriber growth', 'subscriber base'],
    '"sandbox"."sales_data_sync"': ['sales','buyers remorse','brm', 'calls','selling'],
    '"sandbox"."dbspl_sync"': ['p&l','p/l','income statement','i/s','profit and loss','arpu','cogs','totals','revenue','cogs','operating income','oibda']
}
#'''

DATA_DICTIONARY = {}

# Step up from /toolkit/base.py to the root project folder
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

for table_name, file_name in SCHEMA_CONFIG.items():
    dict_path = PROJECT_ROOT / "dictionaries" / file_name
    try:
        with dict_path.open("r", encoding="utf-8") as f:
            schema_data = json.load(f)
            schema_data["table_name"] = table_name
            DATA_DICTIONARY[table_name] = schema_data
    except Exception as e:
        logger.error(f"Error loading schema {file_name}: {e}")


# ─── Helper Functions ────────────────────────────────────────────
# Replace @st.cache_resource with:
@lru_cache(maxsize=1)
def get_db_engine():
    """
    Caches the SQLAlchemy engine pool, but delegates physical connection 
    creation to a dynamic function so Postgres always gets a fresh token.
    """
    ssl_context = ssl.create_default_context()
    
    def get_fresh_connection():
        """Creates a new pg8000 connection with a freshly fetched auth token."""
        fresh_token = get_auth_token()
        current_user = get_workspace_client().current_user.me().user_name
        return pg8000.connect(
            user=current_user,
            password=fresh_token,
            host=PGHOST,
            port=5432,
            database=PGDATABASE,
            ssl_context=ssl_context
        )
    
    # We pass an empty URL and bind the dynamic connection creator
    return sa.create_engine("postgresql+pg8000://", creator=get_fresh_connection)


@mlflow.trace(name="run_sql_query")
def run_sql_query(query: str) -> pd.DataFrame:
    """Executes a raw SQL string against the Postgres engine and returns a DataFrame after validating safety."""
    validate_read_only_sql(query)
    engine = get_db_engine()
    with engine.connect() as conn:
        return pd.read_sql(sa.text(query), conn)

def _is_gpt_model(model_name: str) -> bool:
    """Returns True if the endpoint name indicates a GPT model (OpenAI-native tool-calling)."""
    return "gpt" in model_name.lower()



def llm_call(
    messages: list, 
    response_model: BaseModel = None, 
    tools: list = None, 
    timeout: int = 20, 
    model_name: str = None, 
    context: SessionContext = None
):
    """
    Unified LLM call supporting both structured JSON outputs (via instructor/schema)
    and native chat completions (with or without tools).
    """
    resolved_model = model_name or (context.active_model if context else "system.ai.gpt-5-4-nano")

    # --- 1. Standard Chat/Tool Completion (No Pydantic Model) ---
    if not response_model:
        kwargs = {"model": resolved_model, "messages": messages, "timeout": timeout}
        if tools:
            kwargs["tools"] = tools
            
        raw_res = raw_client.chat.completions.create(**kwargs)
        track_tokens(raw_res, context)
        return raw_res

    # --- 2. Structured Output (Pydantic Model Provided) ---
    if _is_gpt_model(resolved_model):
        fresh_client = _make_fresh_openai_client()
        fresh_instructor_client = instructor.from_openai(fresh_client)
        model_res, raw_res = fresh_instructor_client.chat.completions.create_with_completion(
            model=resolved_model,
            messages=messages,
            response_model=response_model,
            max_retries=3
        )
        track_tokens(raw_res, context) 
        return model_res
    else:
        schema_str = json.dumps(response_model.model_json_schema(), indent=2)
        json_instruction = (
            f"\n\nYou MUST respond with a single valid JSON object that strictly conforms "
            f"to this schema and nothing else — no markdown, no explanation:\n{schema_str}"
        )

        augmented_messages = list(messages)
        system_prompt_index = next(
            (i for i, msg in enumerate(augmented_messages) if msg.get("role") == "system"), -1
        )
        if system_prompt_index != -1:
            augmented_messages[system_prompt_index] = dict(augmented_messages[system_prompt_index])
            augmented_messages[system_prompt_index]["content"] += json_instruction
        else:
            augmented_messages = [{"role": "system", "content": json_instruction.strip()}] + augmented_messages

        last_exc = None
        for attempt in range(3):
            raw_res = raw_client.chat.completions.create(
                model=resolved_model,
                messages=augmented_messages,
                timeout=timeout
            )
            track_tokens(raw_res, context) 
            text = _extract_text_content(raw_res.choices[0].message).strip()

            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()

            try:
                return response_model.model_validate_json(text)
            except Exception as exc:
                last_exc = exc
                augmented_messages = augmented_messages + [
                    {"role": "assistant", "content": text},
                    {"role": "user", "content": f"That response failed validation: {exc}. Please return only valid JSON."}
                ]

        raise last_exc

def track_tokens(response, context: SessionContext):
    """
    Extracts token usage from a live API response and calculates cost based on 
    the active model saved in the session context.
    """
    if not context:
        return

    if hasattr(response, "usage") and response.usage:
        input_t = getattr(response.usage, "input_tokens", 0) or getattr(response.usage, "prompt_tokens", 0)
        output_t = getattr(response.usage, "output_tokens", 0) or getattr(response.usage, "completion_tokens", 0)
        total_t = getattr(response.usage, "total_tokens", 0) or (input_t + output_t)

        context.input_tokens += input_t
        context.output_tokens += output_t
        context.total_tokens += total_t

        # Uses context.active_model instead of global ModelConfig.ACTIVE_MODEL
        context.estimated_cost += calculate_cost(
            context.active_model, input_t, output_t
        )

def get_join_clause(table_a: str, table_b: str) -> str:
    """
    Dynamically generates a SQL ON clause by intersecting the shared 
    conformed dimensions between two tables.
    """
    dims_a = TABLE_DIMENSIONS.get(table_a)
    dims_b = TABLE_DIMENSIONS.get(table_b)
    
    if not dims_a or not dims_b:
        raise ValueError(f"One or both tables not found in TABLE_DIMENSIONS: {table_a}, {table_b}")
        
    # Find the overlapping dimensional concepts (e.g., {'year', 'month'})
    shared_concepts = set(dims_a.keys()) & set(dims_b.keys())
    
    if not shared_concepts:
        raise ValueError(f"No shared dimensions found between {table_a} and {table_b} to join on.")
        
    join_conditions = []
    for concept in shared_concepts:
        col_a = dims_a[concept]
        col_b = dims_b[concept]
        # Generates: "table_a"."col_a" = "table_b"."col_b"
        join_conditions.append(f'{table_a}."{col_a}" = {table_b}."{col_b}"')
        
    return " AND ".join(join_conditions)
