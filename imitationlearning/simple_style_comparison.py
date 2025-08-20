import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gymnasium_env.envs.driver_monaco import DriverMonaco
from stable_baselines3.common.policies import ActorCriticPolicy

# Set matplotlib to use non-interactive backend
plt.switch_backend('Agg')

def compare_driving_styles():
    """Simple comparison of Leclerc vs Hamilton driving styles"""
    
    print("🏁 F1 Driving Style Analysis - Leclerc vs Hamilton")
    print("="*60)
    
    # Load data
    print("Loading driver data...")
    leclerc_data = pd.read_csv('merged_data_clean.csv')
    hamilton_data = pd.read_csv('hamilton_merged_data_clean.csv')
    
    print(f"Leclerc data: {len(leclerc_data)} records")
    print(f"Hamilton data: {len(hamilton_data)} records")
    
    # Load trained models
    print("Loading trained models...")
    try:
        leclerc_policy = ActorCriticPolicy.load("leclerc_imitation_bc")
        hamilton_policy = ActorCriticPolicy.load("hamilton_imitation_bc")
        print("✅ Models loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return
    
    # Create environments
    leclerc_env = DriverMonaco(leclerc_data)
    hamilton_env = DriverMonaco(hamilton_data)
    
    # Analyze expert data first (actual telemetry)
    print("\n📊 EXPERT DATA ANALYSIS")
    print("-"*40)
    analyze_expert_data(leclerc_data, hamilton_data)
    
    # Collect model predictions
    print("\n🤖 MODEL PREDICTIONS ANALYSIS")
    print("-"*40)
    
    num_steps = 500
    leclerc_actions = collect_model_actions(leclerc_policy, leclerc_env, num_steps)
    hamilton_actions = collect_model_actions(hamilton_policy, hamilton_env, num_steps)
    
    if len(leclerc_actions) > 0 and len(hamilton_actions) > 0:
        analyze_model_predictions(leclerc_actions, hamilton_actions)
        create_comparison_visualizations(leclerc_actions, hamilton_actions, leclerc_data, hamilton_data)
    else:
        print("❌ Failed to collect model predictions")
    
    print("\n🏁 Analysis Complete!")

def analyze_expert_data(leclerc_data, hamilton_data):
    """Analyze the actual expert telemetry data"""
    
    print("Analyzing actual telemetry data...")
    
    # Basic statistics
    leclerc_stats = {
        'throttle_mean': leclerc_data['throttle'].mean(),
        'throttle_std': leclerc_data['throttle'].std(),
        'brake_mean': leclerc_data['brake'].mean(),
        'brake_std': leclerc_data['brake'].std(),
        'gear_mean': leclerc_data['n_gear'].mean(),
        'speed_mean': leclerc_data['speed'].mean(),
        'speed_max': leclerc_data['speed'].max(),
    }
    
    hamilton_stats = {
        'throttle_mean': hamilton_data['throttle'].mean(),
        'throttle_std': hamilton_data['throttle'].std(),
        'brake_mean': hamilton_data['brake'].mean(),
        'brake_std': hamilton_data['brake'].std(),
        'gear_mean': hamilton_data['n_gear'].mean(),
        'speed_mean': hamilton_data['speed'].mean(),
        'speed_max': hamilton_data['speed'].max(),
    }
    
    print("Expert Telemetry Comparison:")
    print(f"                  Leclerc    Hamilton    Difference")
    print(f"Throttle (avg):   {leclerc_stats['throttle_mean']:6.1f}%    {hamilton_stats['throttle_mean']:6.1f}%    {leclerc_stats['throttle_mean'] - hamilton_stats['throttle_mean']:+6.1f}%")
    print(f"Brake (avg):      {leclerc_stats['brake_mean']:6.1f}%    {hamilton_stats['brake_mean']:6.1f}%    {leclerc_stats['brake_mean'] - hamilton_stats['brake_mean']:+6.1f}%")
    print(f"Speed (avg):      {leclerc_stats['speed_mean']:6.1f}     {hamilton_stats['speed_mean']:6.1f}     {leclerc_stats['speed_mean'] - hamilton_stats['speed_mean']:+6.1f}")
    print(f"Speed (max):      {leclerc_stats['speed_max']:6.1f}     {hamilton_stats['speed_max']:6.1f}     {leclerc_stats['speed_max'] - hamilton_stats['speed_max']:+6.1f}")
    print(f"Gear (avg):       {leclerc_stats['gear_mean']:6.2f}      {hamilton_stats['gear_mean']:6.2f}      {leclerc_stats['gear_mean'] - hamilton_stats['gear_mean']:+6.2f}")

def collect_model_actions(policy, env, num_steps):
    """Collect actions from a trained model"""
    print(f"Collecting {num_steps} model predictions...")
    
    try:
        obs, _ = env.reset()
        actions = []
        
        for step in range(min(num_steps, env.max_steps)):
            action, _ = policy.predict(obs, deterministic=True)
            actions.append(action)
            obs, _, done, truncated, _ = env.step(action)
            
            if done or truncated:
                print(f"Episode ended at step {step}")
                break
        
        print(f"Collected {len(actions)} actions")
        return np.array(actions)
        
    except Exception as e:
        print(f"Error collecting actions: {e}")
        return np.array([])

def analyze_model_predictions(leclerc_actions, hamilton_actions):
    """Analyze the model predictions"""
    
    print("Model Predictions Comparison:")
    print(f"                  Leclerc    Hamilton    Difference")
    
    # Throttle analysis
    lec_throttle = np.mean(leclerc_actions[:, 0])
    ham_throttle = np.mean(hamilton_actions[:, 0])
    print(f"Throttle (avg):   {lec_throttle:6.1f}%    {ham_throttle:6.1f}%    {lec_throttle - ham_throttle:+6.1f}%")
    
    # Brake analysis  
    lec_brake = np.mean(leclerc_actions[:, 1])
    ham_brake = np.mean(hamilton_actions[:, 1])
    print(f"Brake (avg):      {lec_brake:6.1f}%    {ham_brake:6.1f}%    {lec_brake - ham_brake:+6.1f}%")
    
    # Gear analysis
    lec_gear = np.mean(leclerc_actions[:, 2])
    ham_gear = np.mean(hamilton_actions[:, 2])
    print(f"Gear (avg):       {lec_gear:6.2f}      {ham_gear:6.2f}      {lec_gear - ham_gear:+6.2f}")
    
    # Variability analysis
    lec_throttle_std = np.std(leclerc_actions[:, 0])
    ham_throttle_std = np.std(hamilton_actions[:, 0])
    print(f"Throttle (std):   {lec_throttle_std:6.1f}     {ham_throttle_std:6.1f}     {lec_throttle_std - ham_throttle_std:+6.1f}")
    
    # Gear changes
    lec_gear_changes = len(np.where(np.diff(leclerc_actions[:, 2]) != 0)[0])
    ham_gear_changes = len(np.where(np.diff(hamilton_actions[:, 2]) != 0)[0])
    print(f"Gear changes:     {lec_gear_changes:6d}      {ham_gear_changes:6d}      {lec_gear_changes - ham_gear_changes:+6d}")

def create_comparison_visualizations(leclerc_actions, hamilton_actions, leclerc_data, hamilton_data):
    """Create comparison visualizations"""
    
    print("\nCreating visualizations...")
    
    # 1. Action distributions comparison
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    action_names = ['Throttle (%)', 'Brake (%)', 'Gear']
    
    for i, action_name in enumerate(action_names):
        # Model predictions
        axes[i, 0].hist(leclerc_actions[:, i], bins=30, alpha=0.7, label='Leclerc', color='red', density=True)
        axes[i, 0].hist(hamilton_actions[:, i], bins=30, alpha=0.7, label='Hamilton', color='blue', density=True)
        axes[i, 0].set_title(f'{action_name} - Model Predictions')
        axes[i, 0].set_xlabel(action_name)
        axes[i, 0].set_ylabel('Density')
        axes[i, 0].legend()
        axes[i, 0].grid(True, alpha=0.3)
        
        # Expert data
        if i == 0:  # throttle
            axes[i, 1].hist(leclerc_data['throttle'], bins=30, alpha=0.7, label='Leclerc', color='red', density=True)
            axes[i, 1].hist(hamilton_data['throttle'], bins=30, alpha=0.7, label='Hamilton', color='blue', density=True)
        elif i == 1:  # brake
            axes[i, 1].hist(leclerc_data['brake'], bins=30, alpha=0.7, label='Leclerc', color='red', density=True)
            axes[i, 1].hist(hamilton_data['brake'], bins=30, alpha=0.7, label='Hamilton', color='blue', density=True)
        else:  # gear
            axes[i, 1].hist(leclerc_data['n_gear'], bins=8, alpha=0.7, label='Leclerc', color='red', density=True)
            axes[i, 1].hist(hamilton_data['n_gear'], bins=8, alpha=0.7, label='Hamilton', color='blue', density=True)
        
        axes[i, 1].set_title(f'{action_name} - Expert Data')
        axes[i, 1].set_xlabel(action_name)
        axes[i, 1].set_ylabel('Density')
        axes[i, 1].legend()
        axes[i, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('driving_style_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: driving_style_comparison.png")
    plt.close()
    
    # 2. Racing line comparison
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Sample positions from the data for visualization
    sample_size = min(1000, len(leclerc_data), len(hamilton_data))
    lec_sample = leclerc_data.sample(n=sample_size, random_state=42)
    ham_sample = hamilton_data.sample(n=sample_size, random_state=42)
    
    # Plot racing lines
    scatter1 = ax.scatter(lec_sample['x'], lec_sample['y'], c=lec_sample['speed'], 
                         cmap='Reds', s=20, alpha=0.6, label='Leclerc')
    scatter2 = ax.scatter(ham_sample['x'], ham_sample['y'], c=ham_sample['speed'], 
                         cmap='Blues', s=20, alpha=0.6, label='Hamilton')
    
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_title('Monaco GP - Racing Line Comparison (Speed Colored)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    plt.colorbar(scatter1, ax=ax, label='Speed (km/h)', shrink=0.8)
    plt.tight_layout()
    plt.savefig('racing_line_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: racing_line_comparison.png")
    plt.close()
    
    # 3. Performance metrics comparison
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Throttle over time
    ax1.plot(leclerc_actions[:100, 0], label='Leclerc Model', color='red', alpha=0.8)
    ax1.plot(hamilton_actions[:100, 0], label='Hamilton Model', color='blue', alpha=0.8)
    ax1.set_title('Throttle Application (First 100 Steps)')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Throttle (%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Brake over time
    ax2.plot(leclerc_actions[:100, 1], label='Leclerc Model', color='red', alpha=0.8)
    ax2.plot(hamilton_actions[:100, 1], label='Hamilton Model', color='blue', alpha=0.8)
    ax2.set_title('Brake Application (First 100 Steps)')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Brake (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Box plots for throttle
    ax3.boxplot([leclerc_actions[:, 0], hamilton_actions[:, 0]], 
               labels=['Leclerc', 'Hamilton'], patch_artist=True,
               boxprops=dict(facecolor='lightblue', alpha=0.7))
    ax3.set_title('Throttle Distribution Comparison')
    ax3.set_ylabel('Throttle (%)')
    ax3.grid(True, alpha=0.3)
    
    # Box plots for brake
    ax4.boxplot([leclerc_actions[:, 1], hamilton_actions[:, 1]], 
               labels=['Leclerc', 'Hamilton'], patch_artist=True,
               boxprops=dict(facecolor='lightcoral', alpha=0.7))
    ax4.set_title('Brake Distribution Comparison')
    ax4.set_ylabel('Brake (%)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('performance_metrics.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: performance_metrics.png")
    plt.close()
    
    print(f"\n📊 INSIGHTS SUMMARY:")
    print("-"*30)
    
    # Generate insights
    lec_throttle_mean = np.mean(leclerc_actions[:, 0])
    ham_throttle_mean = np.mean(hamilton_actions[:, 0])
    
    if abs(lec_throttle_mean - ham_throttle_mean) > 2:
        if lec_throttle_mean > ham_throttle_mean:
            print(f"• Leclerc is more aggressive on throttle (+{lec_throttle_mean - ham_throttle_mean:.1f}%)")
        else:
            print(f"• Hamilton is more aggressive on throttle (+{ham_throttle_mean - lec_throttle_mean:.1f}%)")
    else:
        print("• Both drivers have similar throttle application")
    
    lec_brake_mean = np.mean(leclerc_actions[:, 1])
    ham_brake_mean = np.mean(hamilton_actions[:, 1])
    
    if abs(lec_brake_mean - ham_brake_mean) > 2:
        if lec_brake_mean > ham_brake_mean:
            print(f"• Leclerc brakes harder (+{lec_brake_mean - ham_brake_mean:.1f}%)")
        else:
            print(f"• Hamilton brakes harder (+{ham_brake_mean - lec_brake_mean:.1f}%)")
    else:
        print("• Both drivers have similar braking patterns")
    
    # Consistency analysis
    lec_throttle_std = np.std(leclerc_actions[:, 0])
    ham_throttle_std = np.std(hamilton_actions[:, 0])
    
    if lec_throttle_std < ham_throttle_std:
        print(f"• Leclerc has more consistent throttle control")
    else:
        print(f"• Hamilton has more consistent throttle control")

if __name__ == "__main__":
    compare_driving_styles()
