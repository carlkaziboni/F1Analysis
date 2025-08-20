import gymnasium as gym
from gymnasium import spaces
import numpy as np

class DriverMonaco(gym.Env):

    def __init__(self, telemetry_df):
        self.telemetry = telemetry_df[
            (telemetry_df['x'] != 0) & (telemetry_df['y'] != 0)
        ].reset_index(drop=True)
        self.current_index = 0
        self.max_steps = len(telemetry_df) - 1

        self.action_columns = ['throttle', 'brake', 'n_gear']
        exclude_columns = self.action_columns + ['date', 'driver_number', 'meeting_key', 
                                            'session_key', 'driver_number_loc', 
                                            'meeting_key_loc', 'session_key_loc']
        self.observation_columns = [col for col in telemetry_df.columns if col not in exclude_columns]

        print(f"Observation features: {self.observation_columns}")
        print(f"Dataset size after cleaning: {len(self.telemetry)} rows")

        # Remove throttle, brake and gear changes
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(self.observation_columns),), dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=np.array([0,0,0]), high=np.array([100,100,7]), dtype=np.float32
        )


    def _get_obs(self):
        # Return all columns except throttle, brake, and n_gear
        obs = self.telemetry.iloc[self.current_index][self.observation_columns].values.astype(np.float32)
        
        # Handle any NaN values by replacing with 0
        obs = np.nan_to_num(obs, nan=0.0)
        
        # Ensure the observation matches the expected shape
        if len(obs) != len(self.observation_columns):
            raise ValueError(f"Observation shape mismatch: got {len(obs)}, expected {len(self.observation_columns)}")
            
        return obs

    def _get_info(self):
        """Return auxiliary information for debugging/analysis"""
        current_row = self.telemetry.iloc[self.current_index]
        return {
            'timestamp': current_row['date'],
            'lap_progress': self.current_index / self.max_steps,
            'step_number': self.current_index,
            'position': {
                'x': current_row['x'],
                'y': current_row['y'], 
                'z': current_row['z']
            },
            'actual_actions': {
                'throttle': current_row['throttle'],
                'brake': current_row['brake'], 
                'gear': current_row['n_gear']
            },
            'car_state': {
                'speed': current_row['speed'],
                'rpm': current_row['rpm'],
                'drs': current_row['drs']
            }
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
        """Calculate reward for imitation learning"""
        target_actions = self._get_target_actions()
        
        # Weighted reward - different importance for different actions
        throttle_diff = abs(action[0] - target_actions[0]) / 100.0  # Normalize throttle
        brake_diff = abs(action[1] - target_actions[1]) / 100.0     # Normalize brake  
        gear_diff = abs(action[2] - target_actions[2]) / 8.0        # Normalize gear
        
        # Weight throttle/brake more heavily than gear changes
        weighted_error = 0.4 * throttle_diff + 0.4 * brake_diff + 0.2 * gear_diff
        reward = -weighted_error  # Negative error (closer = better)
        
        return reward
    
    def _calculate_track_features(self):
        """Calculate additional track-based features"""
        if self.current_index < len(self.telemetry) - 10:  # Look ahead 10 steps
            current_pos = np.array([self.telemetry.iloc[self.current_index]['x'], 
                                  self.telemetry.iloc[self.current_index]['y']])
            future_pos = np.array([self.telemetry.iloc[self.current_index + 10]['x'], 
                                 self.telemetry.iloc[self.current_index + 10]['y']])
            
            # Calculate upcoming track curvature (very basic)
            direction_vector = future_pos - current_pos
            curvature = np.linalg.norm(direction_vector)  # Simplified curvature measure
            
            return curvature
        return 0.0    

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

