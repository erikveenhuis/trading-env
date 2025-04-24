"""Trading logic for the trading environment."""
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class PortfolioState:
    """Container for portfolio state."""
    
    balance: float
    position: float
    position_price: float
    total_transaction_cost: float
    
    def portfolio_value(self, current_price: float) -> float:
        """Calculate total portfolio value at current price."""
        return max(0, self.balance + self.position * current_price)


class TradingLogic:
    """Handle trading logic and portfolio management."""
    
    def __init__(self, transaction_fee: float,
                 reward_scale: float,
                 invalid_action_penalty: float) -> None:
        """Initialize trading logic.

        Args:
            transaction_fee: Transaction fee as a fraction of trade value
            reward_scale: Scaling factor for PnL in rewards
            invalid_action_penalty: Penalty for invalid actions
        """
        self.transaction_fee = transaction_fee
        self.reward_scale = reward_scale
        self.invalid_action_penalty = invalid_action_penalty
    
    def handle_buy(
        self,
        portfolio_state: PortfolioState,
        current_price: float,
        action_value: float,
    ) -> Tuple[bool, PortfolioState]:
        """Handle buy action logic.
        
        Args:
            portfolio_state: Current portfolio state
            current_price: Current asset price
            action_value: Fraction of balance to use for buying (0 to 1)
            
        Returns:
            Tuple of (is_valid, new_portfolio_state)
        """
        if action_value > 1.0 or action_value < 0.0:
            return False, portfolio_state
            
        if portfolio_state.balance <= 1e-9 or current_price <= 1e-20:
            return False, portfolio_state
            
        # Calculate gross transaction value and fee
        gross_transaction_value_cash = portfolio_state.balance * action_value
        transaction_cost = gross_transaction_value_cash * self.transaction_fee
        
        # Calculate net transaction value and resulting position change
        net_transaction_value_cash = gross_transaction_value_cash - transaction_cost
        position_change = net_transaction_value_cash / current_price
        
        # Update state
        new_balance = portfolio_state.balance - gross_transaction_value_cash
        new_position = portfolio_state.position + position_change
        new_total_cost = portfolio_state.total_transaction_cost + transaction_cost
        
        # Calculate position price as weighted average (cost basis)
        if portfolio_state.position <= 1e-9:
            # If no existing position, just use current price
            new_position_price = current_price
        else:
            # Calculate weighted average based on old and new position sizes
            new_position_price = (
                (portfolio_state.position * portfolio_state.position_price) +
                (position_change * current_price)
            ) / new_position
        
        return True, PortfolioState(
            balance=new_balance,
            position=new_position,
            position_price=new_position_price,
            total_transaction_cost=new_total_cost,
        )
    
    def handle_sell(
        self,
        portfolio_state: PortfolioState,
        current_price: float,
        action_value: float,
    ) -> Tuple[bool, PortfolioState]:
        """Handle sell action logic.
        
        Args:
            portfolio_state: Current portfolio state
            current_price: Current asset price
            action_value: Fraction of position to sell (0 to 1)
            
        Returns:
            Tuple of (is_valid, new_portfolio_state)
        """
        if action_value > 1.0 or action_value < 0.0:
            return False, portfolio_state
            
        if portfolio_state.position <= 1e-9:
            return False, portfolio_state
            
        # Calculate shares to sell and gross transaction value
        sell_amount_shares = portfolio_state.position * action_value
        gross_transaction_value_cash = sell_amount_shares * current_price
        
        # Calculate fee and net transaction value (cash received)
        transaction_cost = gross_transaction_value_cash * self.transaction_fee
        net_transaction_value_cash = gross_transaction_value_cash - transaction_cost
        
        if net_transaction_value_cash < 0:
            return False, portfolio_state
            
        # Update state
        new_balance = portfolio_state.balance + net_transaction_value_cash  # Add net amount
        new_position = portfolio_state.position - sell_amount_shares
        new_total_cost = portfolio_state.total_transaction_cost + transaction_cost
        
        return True, PortfolioState(
            balance=new_balance,
            position=new_position,
            position_price=portfolio_state.position_price if new_position > 1e-9 else 0.0,
            total_transaction_cost=new_total_cost,
        )
    
    def apply_trade(
        self,
        portfolio_state: PortfolioState,
        current_price: float,
        action: int,
        action_value: float,
    ) -> PortfolioState:
        """Apply a trading action.
        
        Args:
            portfolio_state: Current portfolio state
            current_price: Current asset price
            action: Action type (0=Hold, 1=Buy, 2=Sell)
            action_value: Action value (0 to 1)
            
        Returns:
            New portfolio state
        """
        if action == 0:  # Hold
            return portfolio_state
            
        if action == 1:  # Buy
            is_valid, new_state = self.handle_buy(portfolio_state, current_price, action_value)
            return new_state if is_valid else portfolio_state
            
        if action == 2:  # Sell
            is_valid, new_state = self.handle_sell(portfolio_state, current_price, action_value)
            return new_state if is_valid else portfolio_state
            
        return portfolio_state
    
    def calculate_reward(
        self,
        prev_portfolio_value: float,
        cur_portfolio_value: float,
        is_valid: bool,
    ) -> float:
        """Calculate reward based on portfolio value change.

        Args:
            prev_portfolio_value: Portfolio value at the end of the previous step.
            cur_portfolio_value: Portfolio value at the end of the current step.
            is_valid: Whether the action taken in the current step was valid.

        Returns:
            Log return scaled reward or penalty for invalid action.
        """
        if not is_valid:
            return self.invalid_action_penalty

        # Add small epsilon to avoid division by zero or log(0)
        if prev_portfolio_value <= 1e-12 or cur_portfolio_value <= 1e-12:
             return 0.0

        # Calculate log return based on previous vs current value
        log_return = np.log(cur_portfolio_value / prev_portfolio_value)

        # Calculate and return the final reward (scaled log return)
        reward = log_return * self.reward_scale

        return reward 