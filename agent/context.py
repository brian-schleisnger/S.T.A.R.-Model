from dataclasses import dataclass, field
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