from dataclasses import dataclass, field
from typing import Dict, Set
from agent.memory import DataFrameMemory, ContextOptimizer

@dataclass
class SessionContext:
    """
    Holds all stateful information and memory objects required by the agent during a run.
    By storing active_model here, we ensure multi-tenant session isolation.
    """
    # Active LLM Model Selection (Isolated per user session)
    active_model: str = "system.ai.gpt-5-4-nano"
    
    # Token Tracking
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    
    # Cost Tracking
    estimated_cost: float = 0.0
    
    # Memory and Optimization Instances
    df_memory: DataFrameMemory = field(default_factory=DataFrameMemory)
    context_optimizer: ContextOptimizer = field(default_factory=ContextOptimizer)

    # Cross-turn DataFrame registry: maps df_id → human-readable label so the LLM
    # can reference data from previous prompts without re-querying the database.
    # Populated by run_agent_loop after each successful tool execution.
    persistent_df_labels: Dict[str, str] = field(default_factory=dict)

    def register_df(self, df_id: str, label: str) -> None:
        """Records a saved DataFrame ID with a short descriptive label for cross-turn reuse."""
        self.persistent_df_labels[df_id] = label

    def get_memory_summary(self) -> str:
        """
        Returns a formatted summary of all DataFrames currently in memory.
        Injected into the tool-selection prompt so the LLM knows what it can reuse.
        """
        if not self.persistent_df_labels:
            return "No data currently in memory from previous turns."
        lines = ["DataFrames available from previous turns (pass as dataframe_id to avoid re-querying):"]
        for df_id, label in self.persistent_df_labels.items():
            df = self.df_memory.get_df(df_id)
            shape_info = f"{df.shape[0]:,} rows × {df.shape[1]} cols" if df is not None else "evicted"
            lines.append(f"  • {df_id} — {label} ({shape_info})")
        return "\n".join(lines)