import pandas as pd
import numpy as np
from imitationlearning.gymnasium_env.envs.driver_monaco import LeclercMonaco

# Load the data
merged_data = pd.read_csv('merged_data_clean.csv')
print(f"Loaded {len(merged_data)} rows of data")

# Create environment
env = LeclercMonaco(merged_data)

# Test reset
print("Testing environment reset...")
obs, info = env.reset()
print(f"Observation shape: {obs.shape}")
print(f"Observation type: {type(obs)}")
print(f"Observation space: {env.observation_space}")
print(f"Action space: {env.action_space}")
print(f"Sample observation: {obs}")

# Test one step
print("\nTesting environment step...")
action = env._get_target_actions()  # Use expert action
print(f"Action shape: {action.shape}")
print(f"Action: {action}")

obs, reward, terminated, truncated, info = env.step(action)
print(f"New observation shape: {obs.shape}")
print(f"Reward: {reward}")
print(f"Terminated: {terminated}")
print(f"Truncated: {truncated}")

print("Environment test completed successfully!")
