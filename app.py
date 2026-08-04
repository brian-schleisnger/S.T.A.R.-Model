import hashlib
import importlib
import io
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import traceback

import streamlit.components.v1 as components

logger = logging.getLogger(__name__)

from databricks.sdk import WorkspaceClient
import mlflow
import openpyxl
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Dataset Agent",
    page_icon=str(Path(__file__).parent / "logo1.png"),
    layout="wide",
)

# ─── CONFIGURATION CONSTANTS ───────────────────────────────────────────────
TIKTOKEN_ENCODING_URL = os.environ.get(
    "TIKTOKEN_ENCODING_URL", 
    "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken"
)
WORKSPACE_TIKTOKEN_PATH = os.environ.get(
    "WORKSPACE_TIKTOKEN_PATH", 
    "/Shared/star-stuff/o200k_base.tiktoken"
)
TORCH_CPU_WHEEL_NAME = os.environ.get(
    "TORCH_CPU_WHEEL_NAME", 
    "torch-2.4.0+cpu-cp311-cp311-linux_x86_64.whl"
)
WORKSPACE_WHL_DIR = os.environ.get(
    "WORKSPACE_WHL_DIR", 
    "/Shared/star-stuff"
)
MLFLOW_EXPERIMENT_PATH = os.environ.get(
    "MLFLOW_EXPERIMENT_PATH", 
    "/Workspace/star-stuff/"
)

# ─── 1. ENVIRONMENT BOOTSTRAPPING (CACHED) ───────────────────────────────
@st.cache_resource
def set_environment() -> None:
    """
    Runs offline caching, PyTorch CPU workarounds, and wheel installations exactly ONCE 
    per server lifecycle, preventing Streamlit from re-running them on every UI interaction.
    """
    print("Initializing environment bootstrap...")
    
    # --- A. TIKTOKEN OFFLINE CACHE SETUP ---
    cache_dir = os.path.join(tempfile.gettempdir(), "tiktoken_cache")
    os.environ["TIKTOKEN_CACHE_DIR"] = cache_dir
    os.makedirs(cache_dir, exist_ok=True)

    tiktoken_url = TIKTOKEN_ENCODING_URL
    url_hash = hashlib.sha1(tiktoken_url.encode()).hexdigest()
    tiktoken_cache_path = os.path.join(cache_dir, url_hash)

    if not os.path.exists(tiktoken_cache_path):
        print("Downloading offline tiktoken vocabulary from Workspace via SDK...")
        w = WorkspaceClient()
        with w.workspace.download(WORKSPACE_TIKTOKEN_PATH) as response:
            with open(tiktoken_cache_path, "wb") as outfile:
                shutil.copyfileobj(response, outfile)
    else:
        print("Tiktoken offline cache already present!")

    # --- B. PYTORCH CPU WORKAROUND & MODULE FLUSHING ---
    try:
        import torch
        print("PyTorch is already installed and working cleanly!")
    except (ImportError, OSError, ValueError) as e:
        print(f"PyTorch missing or broken C++ CUDA dependencies ({type(e).__name__}). Starting clean CPU setup...")

        wheel_name = TORCH_CPU_WHEEL_NAME
        wheel_path = os.path.join(tempfile.gettempdir(), wheel_name)

        if not os.path.exists(wheel_path):
            print("Connecting to Databricks Workspace via SDK...")
            w = WorkspaceClient()
            workspace_path = f"{WORKSPACE_WHL_DIR}/{wheel_name}"
            print(f" -> Downloading CPU-only PyTorch from {workspace_path}...")

            with w.workspace.download(workspace_path) as response:
                with open(wheel_path, "wb") as outfile:
                    shutil.copyfileobj(response, outfile)
        else:
            print(f"Found existing {wheel_path} on disk. Skipping download...")

        print("Force-installing PyTorch CPU into active virtual environment...")
        subprocess.check_call(["pip", "install", wheel_path, "--no-deps", "--force-reinstall"])

        print("Flushing module cache so Python sees the fresh CPU installation...")
        for mod in list(sys.modules.keys()):
            if mod.startswith("torch"):
                del sys.modules[mod]
        importlib.invalidate_caches()

    # --- C. GIT REPO DEPENDENT PACKAGES ---
    print("Installing dependent packages from Git repo...")
    for pkg in [
        "whls/accelerate-1.14.0-py3-none-any.whl",
        "whls/llmlingua-0.2.2-py3-none-any.whl",
    ]:
        subprocess.check_call(["pip", "install", pkg, "--no-deps"])
        
    print("Environment bootstrap complete!")

# Execute bootstrap immediately before importing heavy ML/agent modules
set_environment()


# ─── 2. AGENT & TOOLKIT IMPORTS ──────────────────────────────────────────
from agent.loop import run_agent_loop
from agent.cache import agent_cache
from agent.context import SessionContext
from toolkit.base import AVAILABLE_MODELS


# ─── 3. GLOBAL CONFIGURATION & UI HELPERS ────────────────────────────────
# Set MLflow experiment once globally so it doesn't fire API calls on every chat turn
mlflow.set_experiment(MLFLOW_EXPERIMENT_PATH)

def load_css() -> None:
    """Reads custom CSS from style.css co-located with app.py and injects it."""
    css_path = Path(__file__).parent / "style.css"
    try:
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("⚠️ style.css not found. Proceeding with default styling.")

def scroll_to_bottom() -> None:
    """
    Injects a tiny JS snippet that scrolls the main content area to the bottom.
    Prevents Streamlit's default behavior of jumping back to the top of the page
    on every st.rerun() triggered by button clicks, ratings, re-runs, etc.
    The iframe height is 0 so it takes up no visible space.
    """
    components.html(
        """
        <script>
            // Walk up from this iframe to find Streamlit's main scrollable container
            // and scroll it to the bottom so the latest message stays in view.
            (function () {
                const scrollToBottom = () => {
                    // The main app content lives in the parent window
                    const doc = window.parent.document;
                    // Streamlit wraps everything in a div with overflow:auto
                    const scroller = doc.querySelector('[data-testid="stAppViewBlockContainer"]')
                                  || doc.querySelector('.main .block-container')
                                  || doc.querySelector('.main');
                    if (scroller) {
                        scroller.scrollTop = scroller.scrollHeight;
                    } else {
                        window.parent.scrollTo(0, window.parent.document.body.scrollHeight);
                    }
                };
                // Small delay lets Streamlit finish rendering new elements first
                setTimeout(scrollToBottom, 100);
            })();
        </script>
        """,
        height=0,
    )


def create_excel_buffer(data_list: list) -> bytes:
    """Extracts DataFrames from the agent's output, strips timezones, and writes them to an Excel buffer."""
    buffer = io.BytesIO()
    
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        sheet_counter = 1
        has_data = False
        
        for item in data_list:
            if isinstance(item, pd.DataFrame):
                df = item.copy()
                
                # 1. Strip timezone from standard pandas 'datetimetz' columns
                for col in df.select_dtypes(include=['datetimetz']).columns:
                    df[col] = df[col].dt.tz_localize(None)
                
                # 2. Fallback check for any generic datetime dtypes that still hold tz metadata
                for col in df.columns:
                    if pd.api.types.is_datetime64_any_dtype(df[col]) and getattr(df[col].dt, 'tz', None) is not None:
                        df[col] = df[col].dt.tz_localize(None)
                        
                # Write each cleaned DataFrame to its own tab
                df.to_excel(writer, index=False, sheet_name=f"Result_{sheet_counter}")
                sheet_counter += 1
                has_data = True
                
        # Fallback if the agent only returned models/text but no tabular data
        if not has_data:
            pd.DataFrame({"Message": ["No tabular data available for this query."]}).to_excel(writer, index=False, sheet_name="No Data")
            
    return buffer.getvalue()

# ─── 4. SESSION STATE INITIALIZATION ─────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "context" not in st.session_state:
    st.session_state.context = SessionContext()
if "last_step_latencies" not in st.session_state:
    st.session_state.last_step_latencies = {}
if "rerun_prompt" not in st.session_state:
    st.session_state.rerun_prompt = None
if "rerun_msg_index" not in st.session_state:
    st.session_state.rerun_msg_index = None
if "pending_rating" not in st.session_state:
    # Stores (msg_index, thumbs_up: bool) for ratings submitted this turn,
    # processed at the top of the next rerun before history is rendered.
    st.session_state.pending_rating = None

# Apply CSS after session state so any st.warning() from load_css renders correctly
load_css()

# ─── 4b. PROCESS PENDING RATINGS ─────────────────────────────────────────
# Ratings are set by button clicks and stored in session state so they survive
# the st.rerun() that follows. We commit them to the DB here, at the top of the
# next render pass, before the chat history loop runs.
if st.session_state.pending_rating is not None:
    _rating_index, _thumbs_up = st.session_state.pending_rating
    st.session_state.pending_rating = None

    _rated_msg = st.session_state.messages[_rating_index]
    _rated_prompt = _rated_msg.get("prompt", "")
    _rated_msg["rating"] = "up" if _thumbs_up else "down"

    if _rated_prompt:
        try:
            agent_cache.rate_cache_entry(_rated_prompt, _thumbs_up)
        except Exception as _e:
            logger.warning(f"Rating commit failed: {_e}")


# ─── 5. SIDEBAR ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 Model Performance")
    st.divider()

    # ── Model Selection ──
    model_options = list(AVAILABLE_MODELS.keys())
    
    # Optional: Find the index of the currently active model so the dropdown remembers the choice
    current_endpoint = st.session_state.context.active_model
    default_index = 0
    for i, (display_name, endpoint) in enumerate(AVAILABLE_MODELS.items()):
        if endpoint == current_endpoint:
            default_index = i
            break

    selected_model = st.selectbox(
        "Active Model",
        options=model_options if model_options else ["No models configured"],
        index=default_index,
        help="Select the model used for all reasoning steps.",
        disabled=not model_options,
    )
    if model_options:
        # Save the endpoint directly to the user's isolated session context
        st.session_state.context.active_model = AVAILABLE_MODELS[selected_model]

    st.divider()

    # ── Estimated Cost ──
    # Displayed directly from session state context
    st.metric(label="💰 Est. Session Cost", value=f"${st.session_state.context.estimated_cost:.4f}")

    st.divider()

    # ── Step Latencies ──
    st.markdown("#### ⏱️ Last Turn Latency")
    latencies = st.session_state.get("last_step_latencies", {})
    if latencies:
        total_time = latencies.get("Total Execution", 0.0)
        st.metric(label="Total", value=f"{total_time:.2f}s")
        for step_name, duration in latencies.items():
            if step_name != "Total Execution":
                st.markdown(
                    f"<div style='font-size:0.85em; margin-bottom:2px;'>"
                    f"<b>{step_name}</b>: {duration:.2f}s</div>",
                    unsafe_allow_html=True,
                )
    else:
        st.caption("No query executed yet.")

    st.divider()

    # ── Clear Chat ──
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.context = SessionContext()  # Reset the entire context
        st.session_state.last_step_latencies = {}
        st.rerun()


# ─── 6. MAIN AREA: WELCOME SCREEN (ALWAYS VISIBLE) ──────────────────────
st.markdown("## S.T.A.R. (Subscriber Trends & Acquisition Reference) Model")
st.markdown("Connected data sources:")

# Wrapper div so card CSS only applies here, not to every stContainer in the app
st.markdown('<div class="welcome-cards">', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

with col1:
    with st.container(border=True):
        st.markdown("#### Marketing Spend <span class='status-badge'>● Connected</span>", unsafe_allow_html=True)
        st.markdown("**Separated Monthly by Tactic and Sub-Tactic**")
        st.caption("Source: FCG (01/2021 - 05/2026)")

    with st.container(border=True):
        st.markdown("#### Sales <span class='status-badge'>● Connected</span>", unsafe_allow_html=True)
        st.markdown("**Daily calls, sales, direct adds, and buyer's remorse**")
        st.caption("Source: B.I. & Performance Optimization team (01/2019 - 06/2026)")

with col2:
    with st.container(border=True):
        st.markdown("#### Subscriber Totals <span class='status-badge'>● Connected</span>", unsafe_allow_html=True)
        st.markdown("**Monthly subscriber counts, adds by channel, deactivations, and churn rate**")
        st.caption("Source: OneStream (01/2018 - 06/2026)")

    with st.container(border=True):
        st.markdown("#### Financials <span class='status-badge'>● Connected</span>", unsafe_allow_html=True)
        st.markdown("**Dish business unit monthly P&L statements**")
        st.caption("Source: OneStream (01/2018 - 06/2026)")

with col3:
    with st.container(border=True):
        st.markdown("#### Individual Subscriber <span class='status-badge'>● Connected</span>", unsafe_allow_html=True)
        st.markdown("**Per-customer demographics, sales channel, package, sac attributions, and estimated future values**")
        st.caption("Source: FS2 Economic Data (10/2018 - 03/2026)")

st.markdown("</div>", unsafe_allow_html=True)


# ─── 7. CHAT HISTORY RENDERING ───────────────────────────────────────────
for i, msg in enumerate(st.session_state.messages):
    if msg["role"] in ["user", "assistant"] and msg.get("content"):
        with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🌐"):
            st.markdown(msg["content"])

            if msg.get("figures"):
                for j, fig in enumerate(msg["figures"]):
                    if isinstance(fig, go.Figure):
                        fig.update_layout(
                            height=400, 
                            colorway=["#105e62", "#b2d8d8", "#000000"], 
                            paper_bgcolor='rgba(0,0,0,0)', 
                            plot_bgcolor='rgba(0,0,0,0)'
                        )
                        st.plotly_chart(fig, use_container_width=True, key=f"fig_{i}_{j}")
                    
            if msg["role"] == "assistant" and (msg.get("dfs") or msg.get("run_log")):
                st.markdown("---")
                act_col1, act_col2, act_col3, act_col4, act_col5 = st.columns(5)
                
                with act_col1:
                    if msg.get("dfs"):
                        try:
                            excel_data = create_excel_buffer(msg["dfs"])
                            st.download_button(
                                label="📥 Export Data",
                                data=excel_data,
                                file_name=f"agent_data_export_{i}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                key=f"download_hist_{i}",
                                use_container_width=True
                            )
                        except Exception as e:
                            st.warning(f"⚠️ Excel export unavailable: {e}")
                
                with act_col2:
                    if msg.get("run_log"):
                        with st.expander("🧠 View Agent Execution Trace", expanded=False):
                            for step_num, log in enumerate(msg["run_log"], 1):
                                st.markdown(f"**Step {step_num}:**")
                                clean_log = log.replace('\\n', '\n')
                                st.code(clean_log, language="python" if "def " in clean_log or "import pandas" in clean_log else "sql" if "SELECT " in clean_log else "text", wrap_lines=True)

                with act_col3:
                    if i > 0 and st.session_state.messages[i - 1]["role"] == "user":
                        if st.button("🔄 Re-run", key=f"rerun_{i}", use_container_width=True):
                            st.session_state.rerun_prompt = st.session_state.messages[i - 1]["content"]
                            st.session_state.rerun_msg_index = i
                            st.rerun()

                # Thumbs up / thumbs down rating — only shown for assistant messages
                # that have an associated prompt stored (i.e. came from the agent loop).
                existing_rating = msg.get("rating")  # "up", "down", or None
                with act_col4:
                    if existing_rating == "up":
                        st.markdown(
                            '<div class="rating-submitted rating-up">👍 Helpful</div>',
                            unsafe_allow_html=True,
                        )
                    elif existing_rating != "down" and msg.get("prompt"):
                        if st.button("� Helpful", key=f"thumbs_up_{i}", use_container_width=True):
                            st.session_state.pending_rating = (i, True)
                            st.rerun()

                with act_col5:
                    if existing_rating == "down":
                        st.markdown(
                            '<div class="rating-submitted rating-down">👎 Not helpful</div>',
                            unsafe_allow_html=True,
                        )
                    elif existing_rating != "up" and msg.get("prompt"):
                        if st.button("👎 Not helpful", key=f"thumbs_down_{i}", use_container_width=True):
                            st.session_state.pending_rating = (i, False)
                            st.rerun()

# ─── 8. RE-RUN HANDLER ───────────────────────────────────────────────────
if st.session_state.rerun_prompt is not None:
    rerun_prompt = st.session_state.rerun_prompt
    rerun_index = st.session_state.rerun_msg_index

    st.session_state.rerun_prompt = None
    st.session_state.rerun_msg_index = None

    from agent.cache import agent_cache as _cache
    _cache.mark_as_rerun(rerun_prompt)

    history_before = st.session_state.messages[: rerun_index - 1]

    with st.chat_message("assistant", avatar="🌐"):
        try:
            with st.spinner("Re-running..."):
                # Pass the context object here
                result = run_agent_loop(rerun_prompt, history_before, st.session_state.context)

            st.session_state.last_step_latencies = result.get("step_latencies", {})

            st.session_state.messages[rerun_index] = {
                "role": "assistant",
                "content": result["final_text"],
                "figures": result["figures"],
                "dfs": result["dfs"],
                "run_log": result["run_log"],
                "prompt": rerun_prompt,
                "rating": None,
            }
            st.rerun()

        except Exception as e:
            logger.error(f"Re-run execution failed: {e}", exc_info=True)
            st.error(f"Re-run Error ({type(e).__name__}): {e}")
            with st.expander("Show Traceback"):
                st.code(traceback.format_exc(), language="python")

# ─── 9. CHAT INPUT & EXECUTION ───────────────────────────────────────────
if prompt := st.chat_input("Ask a question about the data..."):
    # 1. Immediately save the user's prompt to state so it survives any reruns
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # 2. Display the user's prompt
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # 3. Run the agent and save the assistant's response
    with st.chat_message("assistant", avatar="🌐"):
        try:
            with st.spinner("Analyzing..."):
                # Pass the context object here
                result = run_agent_loop(prompt, st.session_state.messages, st.session_state.context)
            
            st.session_state.last_step_latencies = result.get("step_latencies", {})
            
            # 4. Only append the assistant's response here
            st.session_state.messages.append({
                "role": "assistant", 
                "content": result["final_text"],
                "figures": result["figures"],
                "dfs": result["dfs"],
                "run_log": result["run_log"],
                "prompt": prompt,
                "rating": None,
            })
            
            st.rerun()
                    
        except Exception as e:
            logger.error(f"Agent Orchestration Error: {e}", exc_info=True)
            st.error(f"Agent Orchestration Error ({type(e).__name__}): {e}")
            with st.expander("Show Traceback"):
                st.code(traceback.format_exc(), language="python")

# ─── 10. SCROLL ANCHOR ───────────────────────────────────────────────────
# Keep the viewport at the bottom after every rerun so button clicks and new
# responses don't jerk the page back to the top.
scroll_to_bottom()
