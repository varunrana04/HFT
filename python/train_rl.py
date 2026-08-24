import os
import sys

# Ensure stable-baselines3 is available (run 'pip install stable-baselines3 gymnasium' if needed)
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.callbacks import EvalCallback
except ImportError:
    print("[ERROR] stable-baselines3 not found. Please install it using: pip install stable-baselines3 gymnasium")
    sys.exit(1)

# Import our custom HFT environment
from rl_env import HFTTradingEnv

def main():
    print("=====================================================")
    print(" HFT Engine - RL Training Module")
    print("=====================================================")
    
    # Initialize Environment
    # We will use the synthetic data generator for demonstration
    # In production, pass data_path="path/to/historical_l2_data.csv"
    env = HFTTradingEnv(steps_per_episode=1000, ticks_per_step=100)
    
    # Verify environment follows Gym API
    print("[INFO] Checking Gym Environment compatibility...")
    check_env(env, warn=True)
    print("[INFO] Environment check passed.")
    
    # Define PPO Model
    # MlpPolicy uses a standard feed-forward neural network
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./tensorboard_logs/")
    
    print("[INFO] Starting PPO Training (10,000 timesteps)...")
    try:
        model.learn(total_timesteps=10000)
    except KeyboardInterrupt:
        print("[INFO] Training interrupted by user.")
        
    # Save the trained model
    os.makedirs("../models", exist_ok=True)
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'ppo_hft_agent'))
    model.save(model_path)
    print(f"[SUCCESS] Model saved to {model_path}.zip")
    
    # Quick Test
    print("[INFO] Running a quick test episode with the trained agent...")
    obs, _ = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        
    print(f"[TEST] Episode finished. Total Reward: {total_reward:.2f}")
    print(f"[TEST] Final Equity: {info['equity']:.2f}")

if __name__ == "__main__":
    main()
