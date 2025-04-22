"""Demo script that runs a random agent in the trading environment."""
import os
import gymnasium as gym
from gymnasium.utils.env_checker import check_env
from trading_env.config import TradingEnvConfig

def main():
    # Get project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(project_root, 'tests', 'fixtures', '2017-06-27_BTC-USD.csv')
    
    # Register and create environment
    gym.register(id='trading-env-v0', entry_point='trading_env.environment:TradingEnv')
    env = gym.make('trading-env-v0', 
        data_path=data_path,
        config=TradingEnvConfig(
            window_size=60,
            initial_balance=10000.0,
            transaction_fee=0.001,
            reward_pnl_scale=1.0,
            reward_cost_scale=0.1,
            render_mode='visual'
        )
    )
    
    print("Checking environment validity...")
    check_env(env.unwrapped)
    print("Environment check passed!")
    
    observation, info = env.reset()
    env.render()
    
    step = 0
    while True:
        step += 1
        action = env.action_space.sample()  # Random action
        observation, reward, terminated, truncated, info = env.step(action)
        
        if not (terminated or truncated):
            env.render()
        
        if terminated or truncated:
            print(f"\nEpisode finished after {step} steps!")
            print(f"Final portfolio value: ${info['portfolio_value']:.2f}")
            print(f"Total transaction cost: ${info['transaction_cost']:.2f}")
            break
    
    env.close()

if __name__ == "__main__":
    main() 