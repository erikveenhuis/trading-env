"""Tests for the trading logic module."""
import pytest
import numpy as np

from trading_env.trading import PortfolioState, TradingLogic


@pytest.fixture
def portfolio_state():
    """Create a portfolio state instance."""
    return PortfolioState(
        balance=10000.0,
        position=0.0,
        position_price=0.0,
        total_transaction_cost=0.0,
    )


@pytest.fixture
def trading_logic():
    """Create a trading logic instance."""
    return TradingLogic(transaction_fee=0.001)


def test_portfolio_state_initialization(portfolio_state):
    """Test portfolio state initialization."""
    assert portfolio_state.balance == 10000.0
    assert portfolio_state.position == 0.0
    assert portfolio_state.position_price == 0.0
    assert portfolio_state.total_transaction_cost == 0.0


def test_portfolio_value(portfolio_state):
    """Test portfolio value calculation."""
    # Initial state
    assert portfolio_state.portfolio_value(current_price=100.0) == 10000.0
    
    # With position
    portfolio_state.position = 2.0
    portfolio_state.position_price = 100.0
    assert portfolio_state.portfolio_value(current_price=120.0) == 10240.0  # 10000 + 2 * (120-100)


def test_handle_buy(trading_logic, portfolio_state):
    """Test buy action handling."""
    current_price = 100.0
    action_value = 0.5  # Buy 50% of available balance
    
    # Execute buy
    is_valid, new_state = trading_logic.handle_buy(
        portfolio_state=portfolio_state,
        current_price=current_price,
        action_value=action_value,
    )
    
    assert is_valid
    assert new_state.balance == pytest.approx(5000.0, rel=1e-10)  # 10000 * 0.5
    assert new_state.position == pytest.approx(49.95, rel=1e-10)  # (10000 * 0.5 * (1 - 0.001)) / 100
    assert new_state.position_price == current_price
    assert new_state.total_transaction_cost == pytest.approx(5.0, rel=1e-10)  # 5000 * 0.001


def test_handle_sell(trading_logic, portfolio_state):
    """Test sell action handling."""
    # Setup initial position
    portfolio_state.position = 1.0
    portfolio_state.position_price = 90.0
    current_price = 100.0
    action_value = 0.5  # Sell 50% of position
    
    # Execute sell
    is_valid, new_state = trading_logic.handle_sell(
        portfolio_state=portfolio_state,
        current_price=current_price,
        action_value=action_value,
    )
    
    assert is_valid
    expected_sell_value = 50.0  # 100 * 0.5
    expected_transaction_cost = expected_sell_value * 0.001
    
    assert new_state.balance == pytest.approx(10000.0 + expected_sell_value - expected_transaction_cost, rel=1e-10)
    assert new_state.position == pytest.approx(0.5, rel=1e-10)
    assert new_state.position_price == 90.0  # Unchanged
    assert new_state.total_transaction_cost == pytest.approx(expected_transaction_cost, rel=1e-10)


def test_invalid_actions(trading_logic, portfolio_state):
    """Test invalid action handling."""
    current_price = 100.0
    
    # Invalid buy (action_value > 1)
    is_valid, _ = trading_logic.handle_buy(
        portfolio_state=portfolio_state,
        current_price=current_price,
        action_value=1.5,
    )
    assert not is_valid
    
    # Invalid sell (no position)
    is_valid, _ = trading_logic.handle_sell(
        portfolio_state=portfolio_state,
        current_price=current_price,
        action_value=0.5,
    )
    assert not is_valid


def test_apply_trade(trading_logic, portfolio_state):
    """Test trade application."""
    current_price = 100.0
    
    # Test hold action
    new_state = trading_logic.apply_trade(
        portfolio_state=portfolio_state,
        current_price=current_price,
        action=0,  # Hold
        action_value=0.0,
    )
    assert new_state == portfolio_state  # Should be unchanged
    
    # Test buy action
    new_state = trading_logic.apply_trade(
        portfolio_state=portfolio_state,
        current_price=current_price,
        action=1,  # Buy
        action_value=0.5,
    )
    assert new_state.balance < portfolio_state.balance
    assert new_state.position > portfolio_state.position
    
    # Test sell action (after setting up a position)
    portfolio_state.position = 1.0
    portfolio_state.position_price = 90.0
    new_state = trading_logic.apply_trade(
        portfolio_state=portfolio_state,
        current_price=current_price,
        action=2,  # Sell
        action_value=0.5,
    )
    assert new_state.balance > portfolio_state.balance
    assert new_state.position < portfolio_state.position


def test_calculate_reward(trading_logic, portfolio_state):
    """Test reward calculation."""
    current_price = 100.0
    
    # Test reward for hold action
    reward = trading_logic.calculate_reward(
        portfolio_state=portfolio_state,
        new_portfolio_state=portfolio_state,
        current_price=current_price,
        is_valid=True,
    )
    assert reward == 0.0  # No change in portfolio value
    
    # Test reward for invalid action
    reward = trading_logic.calculate_reward(
        portfolio_state=portfolio_state,
        new_portfolio_state=portfolio_state,
        current_price=current_price,
        is_valid=False,
    )
    assert reward == -1.0  # Penalty for invalid action
    
    # Test reward for profitable trade
    portfolio_state.position = 1.0
    portfolio_state.position_price = 90.0
    new_state = PortfolioState(
        balance=10100.0,
        position=0.0,
        position_price=0.0,
        total_transaction_cost=0.1,
    )
    reward = trading_logic.calculate_reward(
        portfolio_state=portfolio_state,
        new_portfolio_state=new_state,
        current_price=current_price,
        is_valid=True,
    )
    assert reward > 0.0  # Positive reward for profitable trade 