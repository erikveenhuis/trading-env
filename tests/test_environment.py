"""Tests for the trading environment module."""
import pytest
import numpy as np
import pandas as pd
import gymnasium as gym

from trading_env.environment import TradingEnv
from trading_env.config import TradingEnvConfig


@pytest.fixture
def sample_data():
    """Create sample market data for testing."""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='h')
    data = pd.DataFrame({
        'open': np.random.uniform(90, 110, 100),
        'high': np.random.uniform(100, 120, 100),
        'low': np.random.uniform(80, 100, 100),
        'close': np.random.uniform(90, 110, 100),
        'volume': np.random.uniform(1000, 2000, 100),
    }, index=dates)
    return data


@pytest.fixture
def trading_env(sample_data, tmp_path):
    """Create a trading environment instance."""
    # Save sample data to temporary file
    data_path = tmp_path / "test_data.csv"
    sample_data.to_csv(data_path)
    
    config = TradingEnvConfig(
        data_path=str(data_path),
        window_size=10,
        initial_balance=10000.0,
        transaction_fee=0.001,
        reward_pnl_scale=1.0,
        reward_cost_scale=0.1,
        render_mode=None,
    )
    
    env = TradingEnv(config=config)
    return env


def test_environment_initialization(trading_env):
    """Test environment initialization."""
    assert isinstance(trading_env, gym.Env)
    assert trading_env.config.window_size == 10
    assert trading_env.config.initial_balance == 10000.0
    assert trading_env.config.transaction_fee == 0.001
    assert trading_env.config.reward_pnl_scale == 1.0
    assert trading_env.config.reward_cost_scale == 0.1


def test_action_space(trading_env):
    """Test action space configuration."""
    assert isinstance(trading_env.action_space, gym.spaces.Discrete)
    assert trading_env.action_space.n == 7  # Hold, Buy25%, Buy50%, Buy100%, Sell25%, Sell50%, Sell100%


def test_observation_space(trading_env):
    """Test observation space configuration."""
    assert isinstance(trading_env.observation_space, gym.spaces.Dict)
    
    # Market data features
    assert isinstance(trading_env.observation_space['market_data'], gym.spaces.Box)
    assert trading_env.observation_space['market_data'].shape == (10, 5)  # window_size x features
    
    # Account state features
    assert isinstance(trading_env.observation_space['account_state'], gym.spaces.Box)
    assert trading_env.observation_space['account_state'].shape == (2,)  # position, balance


def test_reset(trading_env):
    """Test environment reset."""
    observation, info = trading_env.reset()
    
    # Check observation structure
    assert isinstance(observation, dict)
    assert 'market_data' in observation
    assert 'account_state' in observation
    
    # Check observation shapes
    assert observation['market_data'].shape == (10, 5)  # window_size x features
    assert observation['account_state'].shape == (2,)  # position, balance
    
    # Check initial portfolio state
    assert info['balance'] == 10000.0  # Initial balance
    assert info['position'] == 0.0  # Initial position


def test_step(trading_env):
    """Test environment step."""
    trading_env.reset()
    
    # Test hold action
    observation, reward, terminated, truncated, info = trading_env.step(0)
    assert not terminated
    assert isinstance(reward, float)
    assert isinstance(observation, dict)
    assert isinstance(info, dict)
    
    # Test buy action
    observation, reward, terminated, truncated, info = trading_env.step(1)  # Buy 25%
    assert not terminated
    assert isinstance(reward, float)
    assert info['balance'] < 10000.0  # Balance should decrease
    assert info['position'] > 0.0  # Position should increase


def test_invalid_actions(trading_env):
    """Test invalid action handling."""
    trading_env.reset()
    
    # Test selling with no position
    observation, reward, terminated, truncated, info = trading_env.step(4)  # Sell 25%
    assert reward == trading_env.config.invalid_action_penalty
    assert info['balance'] == 10000.0  # Balance should be unchanged
    assert info['position'] == 0.0  # Position should be unchanged


def test_episode_termination(trading_env):
    """Test episode termination conditions."""
    trading_env.reset()
    
    # Run until termination
    terminated = False
    step_count = 0
    while not terminated and step_count < 1000:
        _, _, terminated, truncated, _ = trading_env.step(0)  # Hold action
        step_count += 1
    
    assert terminated  # Episode should terminate eventually 