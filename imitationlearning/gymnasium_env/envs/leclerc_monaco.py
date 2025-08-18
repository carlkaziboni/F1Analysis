import gymnasium as gym
from gymnasium import spaces
import numpy as np

class LeclercMonaco(gym.Env):

    def __init__(self, telemetry_df):
        self.telemetry = telemetry_df
        self.current_index = 0
        self.max_steps = len(telemetry_df) - 1

        self.action_columns = ['throttle', 'brake', 'n_gear']
        self.observation_columns = [col for col in telemetry_df.columns if col not in self.action_columns]

        # Remove throttle, brake and gear changes
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(self.observation_columns),), dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=np.array([0,0,-1]), high=np.array([100,100,1]), dtype=np.float32
        )


    def _get_obs(self):
        # Return all columns except throttle, brake, and n_gear
        return self.telemetry.iloc[self.current_index][self.observation_columns].values.astype(np.float32)

    def _get_info(self):
        return {
            'actual_actions': {       # The real driver actions for comparison
                'throttle': self.telemetry.iloc[self.current_index]['throttle'],
                'brake': self.telemetry.iloc[self.current_index]['brake'], 
                'gear': self.telemetry.iloc[self.current_index]['n_gear']
            },
            'lap_progress': self.current_index / len(self.telemetry)
        }


    def _get_target_actions(self):
        # Return the throttle, brake, and n_gear values
        return self.telemetry.iloc[self.current_index][self.action_columns].values.astype(np.float32)
        

    def reset(self, seed=None, options=None):
        self.current_index = 0
        observation = self._get_obs()
        info = self._get_info()
        return observation, info
    
    def _calculate_reward(self, action):
        target_actions = self._get_target_actions()

        action_diff = np.linalg.norm(action - target_actions)
        reward = -action_diff

        return reward

    def step(self, action):
        """Take one step in the environment"""
        # Validate action
        if not self.action_space.contains(action):
            raise ValueError(f"Action {action} is not within action space")
        
        # Calculate reward
        reward = self._calculate_reward(action)
        
        # Move to next timestep
        self.current_index += 1
        
        # Check if episode is done
        terminated = self.current_index >= self.max_steps
        truncated = False
        
        # Get next observation and info
        if not terminated:
            observation = self._get_obs()
        else:
            # Return final observation
            observation = self._get_obs()
            
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info

