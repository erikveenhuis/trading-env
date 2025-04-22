"""Configuration management for the trading environment."""
from dataclasses import dataclass
from typing import Literal, Optional

@dataclass
class TradingEnvConfig:
    """Configuration for the trading environment."""
    
    # Data configuration
    window_size: int = 60
    
    # Trading parameters
    initial_balance: float = 10000.0
    transaction_fee: float = 0.001  # 0.1% fee
    
    # Reward parameters
    reward_pnl_scale: float = 1.0
    reward_cost_scale: float = 0.1
    invalid_action_penalty: float = -1.0
    
    # Environment parameters
    render_mode: Optional[Literal["human", "terminal", "visual"]] = None
    
    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {self.window_size}")
        if self.initial_balance <= 0:
            raise ValueError(f"initial_balance must be > 0, got {self.initial_balance}")
        if not 0 <= self.transaction_fee < 1:
            raise ValueError(f"transaction_fee must be in [0,1), got {self.transaction_fee}")
        if self.reward_pnl_scale <= 0:
            raise ValueError(f"reward_pnl_scale must be > 0, got {self.reward_pnl_scale}")
        if self.reward_cost_scale < 0:
            raise ValueError(f"reward_cost_scale must be >= 0, got {self.reward_cost_scale}") 