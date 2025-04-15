"""
Core trading environment implementation.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import logging

# Configure ONLY the TradingEnv logger for DEBUG messages
logger = logging.getLogger("trading_env")  # Updated logger name


class TradingEnv(gym.Env):
    """
    A trading environment for reinforcement learning.

    Features:
    - 60 normalized OHLCV observations
    - Action space: Discrete: 0=Hold, 1=Buy25%, 2=Buy50%, 3=Buy100%, 4=Sell25%, 5=Sell50%, 6=Sell100%
    - Account state information (position and balance)
    """

    def __init__(
        self,
        data_path: str,
        reward_pnl_scale: float,
        reward_cost_scale: float,
        initial_balance: float,
        transaction_fee: float,
        window_size: int,
    ):
        super(TradingEnv, self).__init__()

        logger.info(
            f"Initializing TradingEnv with window_size={window_size}, initial_balance={initial_balance}, transaction_fee={transaction_fee}"
        )

        self.data_path = data_path
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        self.reward_pnl_scale = reward_pnl_scale
        self.reward_cost_scale = reward_cost_scale

        # ... existing code ... 