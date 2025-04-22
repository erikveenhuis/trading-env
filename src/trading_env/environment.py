from typing import Dict, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.envs.registration import register
import pandas as pd

from trading_env.config import TradingEnvConfig
from trading_env.data import MarketData, MarketDataProcessor, get_observation_at_step
from trading_env.trading import TradingLogic, PortfolioState
from trading_env.visualization import TradingVisualizer

class TradingEnv(gym.Env):
    """
    A trading environment for reinforcement learning.

    Features:
    - Normalized OHLCV observations with configurable window size
    - Discrete action space: 0=Hold, 1=Buy25%, 2=Buy50%, 3=Buy100%, 4=Sell25%, 5=Sell50%, 6=Sell100%
    - Account state information (position and balance)
    - Configurable transaction fees and initial balance
    - Customizable reward function based on PnL and trading costs
    """
    metadata = {'render_modes': ['human', 'terminal', 'visual'], 'render_fps': 1}

    def __init__(self, config: TradingEnvConfig, render_mode: Optional[str] = None) -> None:
        """Initialize the trading environment.
        
        Args:
            config: Environment configuration
            render_mode: The render mode to use
        """
        super().__init__()
        
        # Store configuration
        self.config = config
        self.render_mode = render_mode or config.render_mode
        
        # Initialize components
        self.data_processor = MarketDataProcessor(window_size=config.window_size)
        self.trading_logic = TradingLogic(transaction_fee=config.transaction_fee)
        
        # Load and process market data
        self.market_data = self.data_processor.load_and_process_data(config.data_path)
        
        # Initialize visualizer if needed
        self.visualizer = None
        if self.render_mode == "visual":
            self.visualizer = TradingVisualizer(self.market_data)
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(7)  # 0=Hold, 1=Buy25%, 2=Buy50%, 3=Buy100%, 4=Sell25%, 5=Sell50%, 6=Sell100%
        self.observation_space = spaces.Dict({
            "market_data": spaces.Box(
                low=0,  # Normalized features are in [0,1]
                high=1,
                shape=(config.window_size, self.market_data.num_features),
                dtype=np.float32
            ),
            "account_state": spaces.Box(
                low=0,  # Normalized position and balance are in [0,1]
                high=1,
                shape=(2,),
                dtype=np.float32
            ),
        })
        
        # Initialize state variables
        self.state = None
        self.portfolio_state = None
        self.reset()

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict]:
        """Reset the environment to initial state.
        
        Args:
            seed: Random seed
            options: Additional options (unused)
            
        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)
        
        # Reset state
        self.state = {
            'current_step': 0,
        }
        
        # Reset portfolio state
        self.portfolio_state = PortfolioState(
            balance=self.config.initial_balance,
            position=0.0,
            position_price=0.0,
            total_transaction_cost=0.0,
        )
        
        # Get initial observation and info
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info

    def _get_info(self) -> Dict[str, Union[int, float]]:
        """Get additional information about the current state."""
        current_price = self.market_data.close_prices[self.state['current_step']]
        portfolio_value = self.portfolio_state.portfolio_value(current_price)
        
        return {
            "step": self.state['current_step'],
            "price": current_price,
            "balance": self.portfolio_state.balance,
            "position": self.portfolio_state.position,
            "portfolio_value": portfolio_value,
            "transaction_cost": self.portfolio_state.total_transaction_cost,
            "action": None,  # Will be set in step method
            "step_transaction_cost": 0.0,  # Will be set in step method
        }

    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict]:
        """Execute one step in the environment.
        
        Args:
            action: Trading action to execute (0=Hold, 1=Buy25%, 2=Buy50%, 3=Buy100%, 4=Sell25%, 5=Sell50%, 6=Sell100%)
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        if self.state['current_step'] >= self.market_data.data_length:
            raise RuntimeError("Episode is done, call reset() first")
            
        # Get current price
        current_price = self.market_data.close_prices[self.state['current_step']]
        
        # Store old portfolio state for reward calculation
        old_portfolio_state = self.portfolio_state
        
        # Map action to type and value
        if action == 0:  # Hold
            action_type = 0
            action_value = 0.0
        elif 1 <= action <= 3:  # Buy
            action_type = 1
            action_value = {1: 0.25, 2: 0.50, 3: 1.00}[action]
        else:  # Sell
            action_type = 2
            action_value = {4: 0.25, 5: 0.50, 6: 1.00}[action]
        
        # Apply trading action
        new_portfolio_state = self.trading_logic.apply_trade(
            portfolio_state=self.portfolio_state,
            current_price=current_price,
            action=action_type,
            action_value=action_value,
        )
        
        # Check if action was valid
        is_valid = new_portfolio_state != self.portfolio_state
        
        # Update portfolio state
        self.portfolio_state = new_portfolio_state
        
        # Calculate reward
        reward = float(self.trading_logic.calculate_reward(
            portfolio_state=old_portfolio_state,
            new_portfolio_state=self.portfolio_state,
            current_price=current_price,
            is_valid=is_valid,
        ))
        
        # Get observation and info
        observation = self._get_observation()
        info = self._get_info()
        info["action"] = action
        info["step_transaction_cost"] = (
            self.portfolio_state.total_transaction_cost - old_portfolio_state.total_transaction_cost
        )
        
        # Check termination
        terminated = self._check_termination(self.portfolio_state.portfolio_value(current_price))
        
        # Advance step
        self.state['current_step'] += 1
        
        return observation, reward, terminated, False, info

    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get the current observation."""
        current_price = self.market_data.close_prices[self.state['current_step']]
        return get_observation_at_step(
            market_data=self.market_data,
            step=self.state['current_step'],
            position=self.portfolio_state.position,
            balance=self.portfolio_state.balance,
            initial_balance=self.config.initial_balance,
            current_price=current_price,
        )

    def _check_termination(self, current_portfolio_value: float) -> bool:
        """Check if the episode should terminate."""
        # End if we've reached the end of data
        if self.state['current_step'] >= self.market_data.data_length - 1:
            return True
            
        # End if portfolio value is too low
        if current_portfolio_value < self.config.initial_balance * 0.01:
            return True
            
        return False

    def render(self) -> None:
        """Render the environment."""
        if self.render_mode is None:
            return
            
        info = self._get_info()
        if self.render_mode == "human":
            print(f"Step: {info['step']}\n"
                  f"Price: ${info['price']:.2f}\n"
                  f"Balance: ${info['balance']:.2f}\n"
                  f"Position: {info['position']:.4f}\n"
                  f"Portfolio Value: ${info['portfolio_value']:.2f}\n"
                  f"Total Transaction Cost: ${info['transaction_cost']:.2f}"
                  + (f"\nLast Action: {info['action']}" if info['action'] is not None else "")
                  + f"\n{'-' * 50}")
        elif self.render_mode == "terminal":
            print(f"Step {info['step']}: Price=${info['price']:.2f}, PV=${info['portfolio_value']:.2f}")
        elif self.render_mode == "visual" and self.visualizer is not None:
            self.visualizer.update(info)

    def close(self) -> None:
        """Clean up environment resources."""
        if self.visualizer is not None:
            self.visualizer.close()
