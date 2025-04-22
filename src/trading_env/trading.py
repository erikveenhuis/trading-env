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
    
    def __init__(self, transaction_fee: float) -> None:
        """Initialize trading logic.
        
        Args:
            transaction_fee: Transaction fee as a fraction of trade value
        """
        self.transaction_fee = transaction_fee
        self.invalid_action_penalty = -1.0
    
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
        
        return True, PortfolioState(
            balance=new_balance,
            position=new_position,
            position_price=current_price,
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
        portfolio_state: PortfolioState,
        new_portfolio_state: PortfolioState,
        current_price: float,
        is_valid: bool,
    ) -> float:
        """Calculate reward for the current step.
        
        Args:
            portfolio_state: Portfolio state before action
            new_portfolio_state: Portfolio state after action
            current_price: Current asset price
            is_valid: Whether the action was valid
            
        Returns:
            Calculated reward
        """
        if not is_valid:
            return self.invalid_action_penalty
            
        # Calculate realized PnL
        realized_pnl = new_portfolio_state.balance - portfolio_state.balance
        
        # Calculate unrealized PnL
        old_unrealized = portfolio_state.position * (current_price - portfolio_state.position_price)
        new_unrealized = new_portfolio_state.position * (current_price - new_portfolio_state.position_price)
        unrealized_pnl_change = new_unrealized - old_unrealized
        
        # Total PnL is realized + change in unrealized
        total_pnl = realized_pnl + unrealized_pnl_change
        
        # Return total PnL as reward
        return total_pnl 