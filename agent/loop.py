import concurrent.futures
import inspect
import json
import time
import traceback
from typing import Any, Dict, List, Tuple

import mlflow
import pandas as pd
import plotly.graph_objects as go
from pydantic import BaseModel

from agent.cache import agent_cache
from agent.categories import CATEGORY_REGISTRY, CATEGORY_TOOLS
from agent.context import SessionContext
from agent.schemas import DecomposedQuestions
from toolkit import TOOLS, TOOL_DISPATCHER
from toolkit.base import DATA_DICTIONARY, _extract_text_content, llm_call


Text_history_token_Limit = 50000
Text_history_chat_limit = 4
raw_output_token_limit = 20000
compression_target_rate = 0.5
Tool_timeout = 90
Llm_timeout = 25
Max_retries = 3


# ─── 1. Context & Schema Helpers ─────────────────────────────────────────
def filter_schema(user_prompt: str, run_log: List[str] = None, context: SessionContext = None) -> dict:
    """
    Uses an LLM call to select which tables from DATA_DICTIONARY are needed to
    answer the user's prompt. Omit verbose reference arrays to save tokens.
    """
    def _build_table_summary(table_name: str, table_data: dict) -> dict:
        meta = table_data.get("table_metadata", {})
        actual_table_name = meta.get("table_name", table_name)
        description = meta.get("description", "")
        special_rules = meta.get("special_rules", [])

        raw_columns = table_data.get("columns", [])
        columns = []
        for col in raw_columns:
            columns.append({
                "column": col.get("name", ""),
                "description": col.get("description", "")
            })

        return {
            "table_name": actual_table_name,
            "description": description,
            "special_rules": special_rules,
            "columns": columns
        }

    full_schema_payload = {
        table_name: _build_table_summary(table_name, table_data)
        for table_name, table_data in DATA_DICTIONARY.items()
    }

    prompt = f"""You are a data architect helping route a user's question to the correct database tables.

            User question: "{user_prompt}"

            Below is the complete schema for every available table, including each table's description,
            column names, descriptions, and special rules:

            {json.dumps(full_schema_payload, indent=2)}

            Your task:
            - Read the user's question carefully.
            - Select ONLY the tables whose data is actually needed to answer it.
            - If a question touches revenue, costs, ARPU, OIBDA, or P&L line items → include 'dbspl_sync'.
            - If a question touches subscriber counts, gross/net adds, or churn → include 'subcount_data_synced'.
            - If a question touches marketing spend, tactics, or budgets → include 'dbs_marketing_sync'.
            - If a question touches per-subscriber economics, sales cahnnels, activation plans, packages, cash flow, SAC, NPV, or activation data → include 'acquisition_data_v3'.
            - If a question touches sales, calls, or buyers remorse → include 'sales_data_sync'.
            - When in doubt about whether a table is needed, include it rather than exclude it.
            - Return an empty list only if the question is completely unrelated to any data (e.g. a greeting).

            Return ONLY a JSON object in this exact format — no markdown, no explanation:
            {{"required_tables": ["<exact table name>", ...]}}"""

    msgs = [{"role": "user", "content": prompt}]

    active_model = context.active_model if context else "system.ai.gpt-5-4-nano"

    class SchemaSelection(BaseModel):
        required_tables: List[str]

    try:
        parsed_result = llm_call(
            msgs, 
            response_model=SchemaSelection, 
            model_name=active_model, 
            context=context)
            
        filtered_dict = {}
        for t in parsed_result.required_tables:
            t_clean = t.replace('"', '').replace("'", "").strip().lower()
            if "." in t_clean:
                t_clean = t_clean.split(".")[-1]
                
            matched = False
            for key, data in DATA_DICTIONARY.items():
                if key.strip().lower() == t_clean:
                    filtered_dict[key] = data
                    matched = True
                    break
            
            if not matched:
                for key, data in DATA_DICTIONARY.items():
                    meta_name = data.get("table_metadata", {}).get("table_name", "")
                    meta_clean = meta_name.replace('"', '').replace("'", "").strip().lower()
                    if "." in meta_clean:
                        meta_clean = meta_clean.split(".")[-1]
                        
                    if meta_clean == t_clean:
                        filtered_dict[key] = data
                        break

        if not filtered_dict:
            if run_log is not None:
                run_log.append("Schema filtering returned no tables — defaulting to full schema.")
            return DATA_DICTIONARY

        if run_log is not None:
            run_log.append(f"Schema filtering selected: {list(filtered_dict.keys())}")

        return filtered_dict

    except Exception as e:
        if run_log is not None:
            run_log.append(f"Schema filtering failed ({type(e).__name__}: {str(e)}). Defaulting to full schema.")
        return DATA_DICTIONARY


def decompose_question(user_prompt: str, 
                       schema: dict, 
                       history: List[dict], 
                       run_log: List[str], 
                       context_optimizer, 
                       context: SessionContext = None
                       ) -> List[str]:
    """Breaks the user's prompt into specific data questions using chat history."""
    history_text = context_optimizer.format_history_for_prompt(history, max_tokens=Text_history_token_Limit)
    dataframe_memory = context.get_memory_summary()
    
    prompt = f"""You are a data strategist. Break the user's broad request down into specific, actionable sub-questions, and assign each one the correct category.
    
    Available Data Schema: {json.dumps(schema)}
    
    Recent Conversation History: {history_text}
    
    User Request: {user_prompt}

    RULES:
    1. Only generate data queries if the user is explicitly asking for data analysis, metrics, or insights.
    2. Generate at most ten sub-questions.
    3. If the user is asking a general question, greeting you, or asking about your capabilities, return the user's exact prompt as a single item with category SQL_RETRIEVAL and do NOT generate data queries.
    4. Use the 'Recent Conversation History' to resolve pronouns and missing context. Every sub-question must be fully self-contained.
    5. Do NOT break a single statistical model (Regression, Random Forest, ARIMA, etc.) into separate sub-questions for each metric. Group all requirements for one model into ONE sub-question.
    6. Think and plan your questions sequentially. example: if a user asks for a visualization, you might need one sub-question to pull one data, one sub-question to pull the toehr data, one sub-question to merge the sources together, and ond sub-question to produce the vidualization."""
    
    msgs = [{"role": "user", "content": prompt}]
    
    try:
        parsed_result = llm_call(
            msgs, 
            response_model=DecomposedQuestions, 
            context=context)
        
        return parsed_result.questions
    except Exception as e:
        error_msg = f"Decomposition failed ({type(e).__name__}: {str(e)}). Falling back to raw prompt."
        run_log.append(error_msg)
        return [user_prompt]
    

# ─── 2. Tool Execution & Routing Engine ──────────────────────────────────
def build_tool_selection_prompt(category_hint: str, relevant_schema: dict, context: SessionContext = None) -> str:
    """Builds the system prompt for the tool-selection LLM call."""
    prompt_lines = [
        "You are a tool-selection assistant. Your only job is to call the right tool for the sub-question below.\n",
        f"Category: {category_hint}",
        "Available tools for this category:\n"
    ]
    
    # Generate category tool list dynamically from the registry
    for cat, data in CATEGORY_REGISTRY.items():
        tool_descriptions = [f"{tool_name} ({desc})" for tool_name, desc in data["tools"].items()]
        prompt_lines.append(f"{cat:<22} → {', '.join(tool_descriptions)}")

    # Inject cross-turn memory summary so the LLM can reuse previously pulled data
    memory_summary = context.get_memory_summary() if context else "No data currently in memory from previous turns."
    prompt_lines.extend([
        f"\nDATA MEMORY: {memory_summary}",
        "If a dataframe_id above matches what this sub-question needs, pass it directly instead of re-querying the database.\n",
        f"Use this exact schema for all column names: {json.dumps(relevant_schema)}"
    ])
    
    return "\n".join(prompt_lines)


def execute_tool_call(tool_call: Dict[str, Any], attempt: int, run_log: List[str], df_memory: Any, context: SessionContext = None) -> Tuple[str, bool, List[Any]]:
    """Handles parsing, Pydantic validation, and execution of a single tool call with timeout."""
    tool_name = tool_call["function"]["name"]
    extracted_objects = []
    
    try:
        raw_args = json.loads(tool_call["function"]["arguments"])
        if tool_name in TOOL_DISPATCHER:
            _, validator = TOOL_DISPATCHER[tool_name]
            validated_args_model = validator(**raw_args)
            clean_args = validated_args_model.model_dump()
        else:
            clean_args = raw_args
    except Exception as e:
        error_msg = f"Validation Error on '{tool_name}': {str(e)}"
        run_log.append(error_msg)
        return error_msg, True, []

    log_entry = f"Attempt {attempt+1}: Agent selected {tool_name}"
    if tool_name == "execute_python_tool" and "code" in clean_args:
        log_entry += f"\n\nPython Code:\n{clean_args['code']}"
    elif tool_name == "execute_sql_query_tool" and "sql_query" in clean_args:
        log_entry += f"\n\nSQL Query:\n{clean_args['sql_query']}"
    else:
        formatted_args = json.dumps(clean_args, indent=2).replace('\\n', '\n')
        log_entry += f" with args:\n{formatted_args}"
        
    run_log.append(log_entry)
    
    if tool_name not in TOOL_DISPATCHER:
        error_msg = f"Error: Tool '{tool_name}' does not exist in TOOL_DISPATCHER."
        run_log.append(error_msg)
        return error_msg, True, []

    func, _ = TOOL_DISPATCHER[tool_name]
    try:
        if "df_memory" in inspect.signature(func).parameters:
            clean_args["df_memory"] = df_memory

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func, **clean_args)
            try:
                result = future.result(timeout=Tool_timeout)
            except concurrent.futures.TimeoutError:
                error_msg = f"Error: Tool '{tool_name}' timed out after {Tool_timeout} seconds. Execution aborted."
                run_log.append(error_msg)
                return error_msg, True, []
        
        if isinstance(result, dict):
            if result.get("status") == "error":
                output_text = result.get("message", "Tool failed internally.")
                run_log.append(f"Tool '{tool_name}' returned error status: {output_text}")
                return output_text, True, []
            
            output_text = result.get("text", "Tool executed successfully.")
            
            if result.get("data") is not None and isinstance(result["data"], pd.DataFrame):
                df_id = df_memory.save_df(result["data"])
                output_text += f"\n[System Note: Data saved to Python memory with ID: {df_id}]"
                extracted_objects.append(result["data"])

                # Register this DataFrame in the persistent cross-turn memory summary
                if context is not None:
                    # Build a short human-readable label from the tool name and its key args
                    label_parts = [tool_name.replace("_tool", "")]
                    for key in ("TABLE_NAME", "target_variable", "x_column", "dataframe_id"):
                        val = clean_args.get(key)
                        if val:
                            label_parts.append(str(val) if not isinstance(val, list) else ", ".join(val))
                            break
                    context.register_df(df_id, " | ".join(label_parts))
                
            for key in ["model", "figure"]:
                if result.get(key) is not None:
                    extracted_objects.append(result[key])
            # Some tools (e.g. run_random_forest_tool) return multiple charts under
            # a "figures" list key.  Drain it here so every figure reaches the UI.
            # Use id()-based dedup — Plotly Figure.__eq__ returns a Figure, not a bool,
            # so "not in" would trigger "object has no len()" via truthiness evaluation.
            existing_ids = {id(obj) for obj in extracted_objects}
            for extra_fig in result.get("figures", []) or []:
                if extra_fig is not None and id(extra_fig) not in existing_ids:
                    extracted_objects.append(extra_fig)
                    existing_ids.add(id(extra_fig))
        else:
            output_text = str(result)
            
        error_signatures = ["Error:", "Error executing", "Exception:", "Failed:"]
        has_error = any(sig in output_text for sig in error_signatures)
        
        if has_error:
            run_log.append(f"Tool '{tool_name}' execution flagged issue: {output_text}")
            
        return output_text, has_error, extracted_objects

    except Exception as e:
        error_msg = f"Exception executing tool '{tool_name}': {str(e)}"
        run_log.append(error_msg)
        run_log.append(traceback.format_exc())
        return error_msg, True, []


def execute_tool_routing(sub_questions: List[Any], relevant_schema: dict, chat_history: List[dict], context: SessionContext, run_log: List[str]) -> Tuple[List[str], List[Any], Dict[str, float]]:
    """
    Iterates through decomposed sub-questions, routing them to the correct LLM tools,
    handling retries, and capturing the resultant data and latencies.
    """
    df_memory = context.df_memory
    raw_outputs = []
    current_turn_dfs = []
    routing_latencies = {}
    
    t0_tools = time.perf_counter()

    active_model = context.active_model if context else "system.ai.gpt-5-4-nano"
    
    for idx, sq_obj in enumerate(sub_questions):
        t0_sq = time.perf_counter()
        if isinstance(sq_obj, str):
            sq_text = sq_obj
            category_hint = "SQL_RETRIEVAL"
        elif isinstance(sq_obj, dict):
            sq_text = sq_obj.get("question", str(sq_obj))
            category_hint = sq_obj.get("target_category", "SQL_RETRIEVAL")
        else:
            sq_text = getattr(sq_obj, "question", str(sq_obj))
            category_hint = getattr(sq_obj, "target_category", "SQL_RETRIEVAL")

        prompt = build_tool_selection_prompt(category_hint, relevant_schema, context)
        
        system_content = prompt
        if raw_outputs:
            system_content += f"\n\nContext from previous sub-questions analyzed just now: {raw_outputs}"

        msgs = [{"role": "system", "content": system_content}]
        
        if chat_history:
            clean_history = [
                {"role": m["role"], "content": m.get("content", "")}
                for m in chat_history[-Text_history_chat_limit:]
                if m["role"] != "system"
            ]
            msgs.extend(clean_history)
            
        msgs.append({"role": "user", "content": sq_text})

        for attempt in range(Max_retries):
            allowed_names = CATEGORY_TOOLS.get(category_hint)

            if allowed_names is not None:
                allowed_names = allowed_names.copy()
                if "execute_sql_query_tool" not in allowed_names:
                    allowed_names.append("execute_sql_query_tool")

            active_tools = (
                [t for t in TOOLS if t["function"]["name"] in allowed_names]
                if allowed_names else TOOLS
            )

            response = llm_call(
                messages=msgs, 
                tools=active_tools, 
                timeout=Llm_timeout, 
                model_name=active_model, 
                context=context
            )

            assistant_msg = response.choices[0].message.model_dump(exclude_none=True)
            
            # The agent breaks the loop successfully ONLY when it answers with text (no tool calls)
            if not assistant_msg.get("tool_calls"):
                raw_outputs.append(f"Sub-question: {sq_text}\nAnswer: {_extract_text_content(response.choices[0].message)}")
                break
            
            msgs.append(assistant_msg)
            has_turn_error = False
            
            for tool_call in assistant_msg["tool_calls"]:
                call_id = tool_call.get("id", "call_id")
                tool_name = tool_call["function"]["name"]
                
                output_text, has_error, extracted_objects = execute_tool_call(tool_call, attempt, run_log, df_memory, context)
                
                if has_error:
                    has_turn_error = True
                else:
                    current_turn_dfs.extend(extracted_objects)
                    # Append successful outputs dynamically
                    raw_outputs.append(f"Sub-question: {sq_text}\nTool Used: {tool_name}\nData: {output_text}")
                    
                msgs.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": tool_name,
                    "content": output_text
                })
                
            # Log catastrophic failure only if the final attempt also failed
            if attempt == Max_retries - 1 and has_turn_error:
                raw_outputs.append(f"Sub-question: {sq_text}\nFailed after {Max_retries} attempts.")
        
        routing_latencies[f"  ↳ Tool Exec {idx + 1}"] = round(time.perf_counter() - t0_sq, 2)

    routing_latencies["2. Tool Routing & Execution"] = round(time.perf_counter() - t0_tools, 2)
    return raw_outputs, current_turn_dfs, routing_latencies


# ─── 3. Final Synthesis ──────────────────────────────────────────────────
def synthesize_final_response(user_prompt: str, raw_outputs: List[str], relevant_schema: dict, chat_history: List[dict], context: SessionContext, run_log: List[str]) -> str:
    """
    Compresses tool outputs if necessary and synthesizes the raw data back 
    into a business-friendly final response for the user.
    """
    context_optimizer = context.context_optimizer
    raw_outputs_str = str(raw_outputs)
    
    if context_optimizer.count_tokens(raw_outputs_str) > raw_output_token_limit:
        raw_outputs_str = context_optimizer.compress_text(
            raw_outputs_str, 
            target_rate=compression_target_rate,
            context_instruction="Preserve all numerical values, metric names, and tool error messages."
        )

    synthesis_prompt = f"""You are a data insights assistant writing for business leaders with limited statistical knowledge. 
    User's Original Prompt: {user_prompt}
    Raw Data Extracted across all tools: {raw_outputs_str}
    Relevant Schema: {json.dumps(relevant_schema)}
    
    Synthesize the raw data into a clear, business-friendly summary answering the original prompt.
    If any tools failed or returned errors in the raw data, briefly mention what analysis could not be completed and why, alongside the successful insights.
    Do not try and do math. If the user asked for a yearly total and you received monthly totals for the year, provide the monthly totals without attempting to sum them yourself.
    Do not write code or json in your final answer. 
    Don't ever say the word forecast, call it a computer projection if needed. 
    if the data returned isn't related to the user's prompt, or the user's prompt was unspecific, share what you do have and ask questions to clarify the user's intent.
    
    CRITICAL FORMATTING RULE: 
    Do not use LaTeX formatting for regular text. When mentioning currency, you MUST escape the dollar sign (e.g., \\$10M) so it does not accidentally trigger markdown math blocks."""
    
    clean_messages = [{"role": m["role"], "content": m.get("content", "")} for m in chat_history]
    final_msgs = clean_messages + [{"role": "user", "content": synthesis_prompt}]

    active_model = context.active_model if context else "system.ai.gpt-5-4-nano"
    
    try:
        response = llm_call(
            messages=final_msgs, 
            timeout=Llm_timeout, 
            model_name=active_model, 
            context=context
        )
        
        final_text = _extract_text_content(response.choices[0].message)
    except Exception as e:
        error_trace = traceback.format_exc()
        run_log.append(f"Synthesis Model Failed ({type(e).__name__}): {e}\n{error_trace}")
        final_text = f"**⚠️ Synthesis Failed:** The synthesis model encountered an error.\n\n**Error ({type(e).__name__}):** {e}\n\n**Raw Extracted Data:**\n```text\n{raw_outputs_str[:2000]}...\n```"
        
    return final_text


# ─── 4. Main Agent Orchestrator ──────────────────────────────────────────
@mlflow.trace(name="run_agent_loop")
def run_agent_loop(user_prompt: str, chat_history: List[dict], context: SessionContext) -> Dict[str, Any]:
    """
    The main orchestrator chaining the workflow together across multiple tools.
    Decoupled from UI: Takes history and context in, returns structured dictionary out.
    """
    run_log: List[str] = []
    step_latencies: Dict[str, float] = {}

    # Do NOT clear df_memory here — DataFrames persist across turns so the LLM
    # can reference data from previous prompts via dataframe_id.
    t_start_total = time.perf_counter()

    start_input_tokens = context.input_tokens
    start_output_tokens = context.output_tokens
    start_total_tokens = context.total_tokens

    with mlflow.start_run(run_name="Agent_Interaction"):
        mlflow.log_param("user_prompt", user_prompt)
        
        # ─── 1. SEMANTIC CACHE INTERCEPT ───
        t0 = time.perf_counter()
        cached_result = agent_cache.check_cache(user_prompt)
        step_latencies["Cache Check"] = round(time.perf_counter() - t0, 2)
        
        if cached_result:
            step_latencies["Total Execution"] = round(time.perf_counter() - t_start_total, 2)
            run_log.append(f"⚡ Served from Semantic Cache. Matched Prompt: '{cached_result['matched_prompt']}' ({cached_result['similarity']*100:.1f}% similarity)")
            
            mlflow.log_metrics({
                "latency_cache_check_sec": step_latencies["Cache Check"],
                "latency_total_sec": step_latencies["Total Execution"],
                "cache_hit": 1
            })
            
            return {
                "final_text": cached_result["content"],
                "dfs": cached_result["dfs"],
                "figures": cached_result["figures"],
                "run_log": run_log,
                "step_latencies": step_latencies,
                "is_cached": True,
                "cache_info": cached_result
            }
    
        mlflow.log_metric("cache_hit", 0)

        # ─── 2. DECOMPOSITION & ROUTING ───
        t0 = time.perf_counter()
        relevant_schema = filter_schema(user_prompt, run_log=run_log, context=context)
        sub_questions = decompose_question(user_prompt, relevant_schema, chat_history, run_log, context.context_optimizer, context=context)
        step_latencies["1. Decomposition"] = round(time.perf_counter() - t0, 2)
        run_log.append(f"Sub-questions identified: {sub_questions}")
        
        # ─── 3. TOOL EXECUTION ───
        raw_outputs, current_turn_dfs, routing_latencies = execute_tool_routing(
            sub_questions, relevant_schema, chat_history, context, run_log
        )
        step_latencies.update(routing_latencies)

        # ─── 4. FINAL SYNTHESIS ───
        t0 = time.perf_counter()
        final_text = synthesize_final_response(
            user_prompt, raw_outputs, relevant_schema, chat_history, context, run_log
        )
        step_latencies["3. Final Synthesis"] = round(time.perf_counter() - t0, 2)
        
        step_latencies["Total Execution"] = round(time.perf_counter() - t_start_total, 2)
        
        turn_figures = [item for item in current_turn_dfs if isinstance(item, go.Figure)]
        turn_dfs = [item for item in current_turn_dfs if isinstance(item, pd.DataFrame)]

        # ─── 5. MLFLOW TELEMETRY LOGGING ───
        turn_input_tokens = context.input_tokens - start_input_tokens
        turn_output_tokens = context.output_tokens - start_output_tokens
        turn_total_tokens = context.total_tokens - start_total_tokens

        mlflow.log_metrics({
            "turn_input_tokens": turn_input_tokens,
            "turn_output_tokens": turn_output_tokens,
            "turn_total_tokens": turn_total_tokens,
            "latency_1_decomposition_sec": step_latencies.get("1. Decomposition", 0.0),
            "latency_2_tools_sec": step_latencies.get("2. Tool Routing & Execution", 0.0),
            "latency_3_synthesis_sec": step_latencies.get("3. Final Synthesis", 0.0),
            "latency_total_sec": step_latencies["Total Execution"]
        })

        agent_cache.save_to_cache(
            user_prompt=user_prompt,
            final_text=final_text,
            dfs=turn_dfs,
            figures=turn_figures
        )

        return {
            "final_text": final_text,
            "dfs": turn_dfs,
            "figures": turn_figures,
            "run_log": run_log,
            "step_latencies": step_latencies,
            "is_cached": False
        }