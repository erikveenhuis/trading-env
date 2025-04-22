# Trading Environment

A Gymnasium-based trading environment for reinforcement learning, featuring:

- Real-time visualization of trading performance
- Configurable trading parameters (fees, initial balance, etc.)
- Support for both discrete and continuous action spaces
- Realistic trading logic with transaction costs
- Dark mode visualization

## Installation

```bash
pip install trading-env
```

## Quick Start

```python
import gymnasium as gym
from trading_env.config import TradingEnvConfig

# Register and create environment
gym.register(id='trading-env-v0', entry_point='trading_env.environment:TradingEnv')
env = gym.make('trading-env-v0', 
    config=TradingEnvConfig(
        data_path='path/to/your/data.csv',
        window_size=60,
        initial_balance=10000.0,
        transaction_fee=0.001,
        reward_pnl_scale=1.0,
        reward_cost_scale=0.1,
        render_mode='visual'
    )
)

# Run a random agent
observation, info = env.reset()
env.render()

while True:
    action = env.action_space.sample()  # Random action
    observation, reward, terminated, truncated, info = env.step(action)
    
    if not (terminated or truncated):
        env.render()
    
    if terminated or truncated:
        print(f"\nEpisode finished!")
        print(f"Final portfolio value: ${info['portfolio_value']:.2f}")
        print(f"Total transaction cost: ${info['transaction_cost']:.2f}")
        break

env.close()
```

## Features

### Environment
- Normalized OHLCV observations with configurable window size
- Discrete action space: 0=Hold, 1=Buy25%, 2=Buy50%, 3=Buy100%, 4=Sell25%, 5=Sell50%, 6=Sell100%
- Account state information (position and balance)
- Configurable transaction fees and initial balance
- Customizable reward function based on PnL and trading costs

### Visualization
- Real-time plotting of price, position, portfolio value, and returns
- Dark mode support
- Buy/sell markers on price chart
- Performance metrics display

## Data Format

The environment expects CSV data with the following columns:
- `timestamp`: Unix timestamp
- `open`: Opening price
- `high`: Highest price
- `low`: Lowest price
- `close`: Closing price
- `volume`: Trading volume

## License

MIT License - see LICENSE file for details 