import pandas as pd
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from gymnasium_env.envs.driver_monaco import DriverMonaco
import torch as th

class ExpertDemonstrationPolicy:
    """A wrapper to make expert demonstrations compatible with imitation library"""
    
    def __init__(self, env):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        
    def predict(self, observation, state=None, episode_start=None, deterministic=True):
        """Predict action for given observation"""
        # The imitation library expects this exact signature
        if isinstance(observation, np.ndarray) and len(observation.shape) == 1:
            # Single observation
            action = self.env._get_target_actions()
            return action, state
        else:
            # Batch of observations
            batch_size = observation.shape[0] if hasattr(observation, 'shape') else 1
            actions = []
            for i in range(batch_size):
                action = self.env._get_target_actions()
                actions.append(action)
            return np.array(actions), state

class DeterministicExpertPolicy:
    """Simple deterministic expert policy for rollouts"""
    
    def __init__(self, env):
        self.env = env
        self.current_step = 0
        
    def __call__(self, obs):
        """Call method for rollout collection"""
        action = self.env._get_target_actions()
        return action

def create_dummy_sb3_policy(env):
    """Create a dummy SB3 policy that can be used with imitation library"""
    
    class ExpertSB3Policy(ActorCriticPolicy):
        def __init__(self, observation_space, action_space, lr_schedule, env_ref=None):
            super().__init__(
                observation_space,
                action_space, 
                lr_schedule,
                net_arch=[64, 64],
                activation_fn=th.nn.ReLU,
                features_extractor_class=None,
                features_extractor_kwargs=None,
                normalize_images=True,
                optimizer_class=th.optim.Adam,
                optimizer_kwargs=None,
            )
            self.env_ref = env_ref
        
        def predict(self, observation, state=None, episode_start=None, deterministic=True):
            # Return expert action
            if self.env_ref is not None:
                action = self.env_ref._get_target_actions()
                return action, state
            return np.zeros(self.action_space.shape), state
    
    # Create a dummy learning rate schedule
    def lr_schedule(progress_remaining):
        return 0.001
    
    policy = ExpertSB3Policy(env.observation_space, env.action_space, lr_schedule, env)
    return policy

def load_and_train_driver_imitation(driver_name, csv_file):
    """Load data and train driver behavioral cloning model using imitation library"""
    
    print(f"Loading {driver_name}'s merged telemetry data...")
    
    # Load cleaned data
    merged_data = pd.read_csv(csv_file)
    print(f"Loaded {len(merged_data)} rows of {driver_name} data")
    
    # Create DriverMonaco environment
    print(f"Creating DriverMonaco environment for {driver_name}...")
    env = DriverMonaco(merged_data)
    
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Test the environment first
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial observation: {obs}")
    
    # Create expert demonstrations manually
    print("Creating expert demonstrations...")
    
    observations = []
    actions = []
    rewards = []
    dones = []
    infos = []
    
    # Collect full trajectory
    obs, info = env.reset()
    
    for step in range(min(1000, env.max_steps)):  # Collect up to 1000 steps
        action = env._get_target_actions()
        next_obs, reward, terminated, truncated, next_info = env.step(action)
        
        observations.append(obs.copy())
        actions.append(action.copy())
        rewards.append(reward)
        dones.append(terminated or truncated)
        infos.append(info)
        
        obs = next_obs
        info = next_info
        
        if terminated or truncated:
            print(f"Episode ended at step {step}")
            break
    
    # Add the final observation
    observations.append(obs.copy())
    
    print(f"Collected {len(actions)} demonstration steps with {len(observations)} observations")
    
    # Convert to trajectory format expected by imitation library
    from imitation.data.types import Trajectory
    
    trajectory = Trajectory(
        obs=np.array(observations),
        acts=np.array(actions),
        infos=np.array(infos),
        terminal=dones[-1]  # Only the last step should be terminal
    )
    
    trajectories = [trajectory]
    
    print(f"Created trajectory with {len(trajectory.obs)} steps")
    print(f"Observation shape: {trajectory.obs.shape}")
    print(f"Action shape: {trajectory.acts.shape}")
    
    # Train behavioral cloning model
    print(f"Training Behavioral Cloning model for {driver_name}...")
    
    try:
        bc_trainer = bc.BC(
            observation_space=env.observation_space,
            action_space=env.action_space,
            demonstrations=trajectories,
            rng=np.random.default_rng(42),
        )
        
        # Train for multiple epochs
        print(f"Starting training for {driver_name}...")
        bc_trainer.train(n_epochs=50)
        
        print(f"Training completed successfully for {driver_name}!")
        
        # Evaluate the learned policy
        print(f"\nEvaluating learned policy for {driver_name}...")
        evaluate_imitation_policy(bc_trainer, env, driver_name)
        
        return bc_trainer, env, driver_name
        
    except Exception as e:
        print(f"Error during BC training: {e}")
        print(f"Error type: {type(e)}")
        import traceback
        traceback.print_exc()
        return None, env, driver_name

def evaluate_imitation_policy(bc_trainer, env, driver_name, num_steps=100):
    """Evaluate how well the model learned the driver's driving style"""
    
    obs, info = env.reset()
    
    predicted_actions = []
    actual_actions = []
    rewards = []
    
    print("Running evaluation...")
    
    for step in range(min(num_steps, env.max_steps)):
        # Get predicted action from learned policy
        try:
            predicted_action, _ = bc_trainer.policy.predict(obs, deterministic=True)
            predicted_actions.append(predicted_action)
            
            # Get actual expert action
            actual_action = env._get_target_actions()
            actual_actions.append(actual_action)
            
            # Step environment with predicted action
            obs, reward, done, truncated, info = env.step(predicted_action)
            rewards.append(reward)
            
            if done or truncated:
                print(f"Evaluation episode ended at step {step}")
                break
                
        except Exception as e:
            print(f"Error during evaluation at step {step}: {e}")
            break
    
    if len(predicted_actions) > 0:
        predicted_actions = np.array(predicted_actions)
        actual_actions = np.array(actual_actions)
        
        # Calculate metrics
        action_names = ['Throttle', 'Brake', 'Gear']
        
        print(f"\n{driver_name} Evaluation Results ({len(predicted_actions)} steps):")
        print(f"Average Reward: {np.mean(rewards):.4f}")
        
        for i, action_name in enumerate(action_names):
            pred = predicted_actions[:, i]
            actual = actual_actions[:, i]
            
            mae = np.mean(np.abs(pred - actual))
            correlation = np.corrcoef(pred, actual)[0, 1] if len(set(actual)) > 1 else 0
            
            print(f"{action_name}:")
            print(f"  Mean Absolute Error: {mae:.3f}")
            print(f"  Correlation: {correlation:.3f}")
        
        return predicted_actions, actual_actions, rewards
    else:
        print("No successful evaluation steps completed")
        return [], [], []

if __name__ == "__main__":
    # Train both Leclerc's and Hamilton's behavioral cloning models using imitation library
    print("🏁 Starting Dual Driver Behavioral Cloning with Imitation Library")
    print("="*70)
    
    # Define data files for each driver
    drivers = {
        "Leclerc": "merged_data_clean.csv",
        "Hamilton": "hamilton_merged_data_clean.csv"
    }
    
    trained_models = {}
    
    for driver_name, csv_file in drivers.items():
        print(f"\n🏁 Training {driver_name} Model")
        print("-"*50)
        
        bc_trainer, env, driver = load_and_train_driver_imitation(driver_name, csv_file)
        
        if bc_trainer is not None:
            # Save the trained model with driver-specific name
            model_name = f"{driver_name.lower()}_imitation_bc"
            try:
                bc_trainer.policy.save(model_name)
                print(f"✅ {driver_name} model saved as '{model_name}'")
                trained_models[driver_name] = model_name
            except Exception as e:
                print(f"❌ Error saving {driver_name} model: {e}")
        else:
            print(f"❌ {driver_name} training failed")
    
    print("\n" + "="*70)
    print("🏁 Training Summary")
    print("="*70)
    
    if trained_models:
        print("✅ Successfully trained models:")
        for driver, model_name in trained_models.items():
            print(f"  • {driver}: {model_name}")
        
        print(f"\n🎯 Total models trained: {len(trained_models)}")
        print("🏁 Dual driver training complete!")
    else:
        print("❌ No models were successfully trained")
        
    print("="*70)
