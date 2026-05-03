"""
rl_agent.py — DQN Reinforcement Learning Agent Wrapper

Uses stable-baselines3 DQN to train an agent on the custom TradingEnv.
Supports model saving/loading for persistence (train once, predict many times).
"""

import os
import logging
import numpy as np
import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from rl_env import TradingEnv, COOLDOWN_PERIOD

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Default model save path (relative to project directory)
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, "rl_model")


class TrainingLogger(BaseCallback):
    """
    Custom callback to log training progress and collect metrics for UI display.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0.0
        self.current_episode_length = 0
        self.total_episodes = 0

    def _on_step(self) -> bool:
        self.current_episode_reward += self.locals.get('rewards', [0])[0]
        self.current_episode_length += 1

        # Check if episode ended
        dones = self.locals.get('dones', [False])
        if dones[0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            self.total_episodes += 1
            self.current_episode_reward = 0.0
            self.current_episode_length = 0

        return True

    def get_training_stats(self):
        """Returns training statistics for UI display."""
        if not self.episode_rewards:
            return {
                'total_episodes': 0,
                'avg_reward': 0.0,
                'best_reward': 0.0,
                'worst_reward': 0.0,
                'avg_episode_length': 0,
            }

        return {
            'total_episodes': self.total_episodes,
            'avg_reward': float(np.mean(self.episode_rewards)),
            'best_reward': float(np.max(self.episode_rewards)),
            'worst_reward': float(np.min(self.episode_rewards)),
            'avg_episode_length': float(np.mean(self.episode_lengths)),
        }


def train_rl_agent(df, feature_cols, total_timesteps=10000,
                   stop_loss=0.02, take_profit=0.03, trans_cost=0.001,
                   save_path=None):
    """
    Trains a DQN agent on the trading environment and saves the model.

    Args:
        df (pd.DataFrame): Feature-engineered dataframe with OHLCV + features.
        feature_cols (list): List of feature column names.
        total_timesteps (int): Number of training timesteps for DQN.
        stop_loss (float): Stop-loss threshold.
        take_profit (float): Take-profit threshold.
        trans_cost (float): Transaction cost per trade.
        save_path (str): Path to save the trained model (without .zip extension).

    Returns:
        DQN: Trained DQN model.
        dict: Training statistics (episodes, avg reward, etc.).
    """
    logging.info(f"Training DQN agent for {total_timesteps} timesteps...")

    # Create environment
    env = TradingEnv(
        df=df, feature_cols=feature_cols,
        stop_loss=stop_loss, take_profit=take_profit,
        trans_cost=trans_cost
    )

    # Create callback for logging
    logger_callback = TrainingLogger()

    # Create DQN model with tuned hyperparameters
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=128,
        gamma=0.99,
        exploration_fraction=0.3,
        exploration_final_eps=0.05,
        target_update_interval=500,
        verbose=0
    )

    # Train
    model.learn(total_timesteps=total_timesteps, callback=logger_callback)

    # Get training stats
    training_stats = logger_callback.get_training_stats()
    training_stats['total_timesteps'] = total_timesteps
    training_stats['algorithm'] = 'DQN'

    # Save model
    if save_path is None:
        save_path = DEFAULT_MODEL_PATH
    model.save(save_path)
    logging.info(f"Model saved to {save_path}.zip")

    logging.info(f"DQN Training complete. Episodes: {training_stats['total_episodes']}, "
                 f"Avg Reward: {training_stats['avg_reward']:.4f}")

    return model, training_stats


def load_rl_model(model_path=None):
    """
    Loads a previously saved DQN model from disk.

    Args:
        model_path (str): Path to the saved model (without .zip extension).

    Returns:
        DQN: Loaded DQN model, or None if file doesn't exist.
    """
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH

    zip_path = model_path + ".zip"
    if os.path.exists(zip_path):
        logging.info(f"Loading saved RL model from {zip_path}...")
        model = DQN.load(model_path)
        logging.info("RL model loaded successfully.")
        return model
    else:
        logging.warning(f"No saved model found at {zip_path}.")
        return None


def is_model_saved(model_path=None):
    """Check if a trained model exists on disk."""
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH
    return os.path.exists(model_path + ".zip")


def predict_rl_actions(model, df, feature_cols):
    """
    Generates action sequence for each bar using the trained DQN model.
    Includes position tracking, cooldown, and trend filter constraints.

    Args:
        model (DQN): Trained DQN model.
        df (pd.DataFrame): Feature-engineered dataframe.
        feature_cols (list): Feature column names.

    Returns:
        list: Action sequence (0=HOLD, 1=BUY, 2=SELL) for each bar.
    """
    actions = []
    in_position = False
    cooldown_remaining = 0

    for i in range(len(df)):
        # Build observation: market features + position state
        market_features = df[feature_cols].iloc[i].values.astype(np.float32)
        position_state = np.array([1.0 if in_position else 0.0], dtype=np.float32)
        obs = np.concatenate([market_features, position_state])

        # Predict action (deterministic for evaluation)
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)

        # Enforce valid constraints
        if action == 1 and in_position:
            action = 0  # Can't buy if already holding
        if action == 2 and not in_position:
            action = 0  # Can't sell if not holding
        if action == 1 and cooldown_remaining > 0:
            action = 0  # Can't buy during cooldown

        # Trend filter: BUY only in uptrend
        if action == 1 and 'SMA_10' in df.columns and 'SMA_50' in df.columns:
            if df['SMA_10'].iloc[i] <= df['SMA_50'].iloc[i]:
                action = 0

        # Decrement cooldown
        if cooldown_remaining > 0:
            cooldown_remaining -= 1

        # Update position tracking
        if action == 1:
            in_position = True
        elif action == 2:
            in_position = False
            cooldown_remaining = COOLDOWN_PERIOD

        actions.append(action)

    return actions


def get_rl_live_prediction(model, today_features, feature_cols, in_position=False):
    """
    Makes a single prediction for the current bar using the trained RL model.

    Args:
        model (DQN): Trained DQN model.
        today_features (pd.Series or np.array): Today's feature values.
        feature_cols (list): Feature column names.
        in_position (bool): Whether currently holding a position.

    Returns:
        dict: Action and confidence information.
    """
    if isinstance(today_features, pd.Series):
        market_features = today_features[feature_cols].values.astype(np.float32)
    else:
        market_features = np.array(today_features, dtype=np.float32)

    position_state = np.array([1.0 if in_position else 0.0], dtype=np.float32)
    obs = np.concatenate([market_features, position_state])

    action, _ = model.predict(obs, deterministic=True)
    action = int(action)

    # Enforce constraints
    if action == 1 and in_position:
        action = 0
    if action == 2 and not in_position:
        action = 0

    action_map = {0: "HOLD", 1: "BUY", 2: "SELL"}

    return {
        'action': action,
        'action_name': action_map[action],
    }


if __name__ == "__main__":
    from data_loader import load_data
    from features import engineer_features

    df = load_data("AAPL", "1y")
    df_feat, feat_cols = engineer_features(df)

    # Train and save
    model, stats = train_rl_agent(df_feat, feat_cols, total_timesteps=5000)
    print(f"\nTraining Stats: {stats}")
    print(f"Model saved: {is_model_saved()}")

    # Load and predict
    loaded_model = load_rl_model()
    if loaded_model:
        actions = predict_rl_actions(loaded_model, df_feat, feat_cols)
        print(f"\nAction distribution: HOLD={actions.count(0)}, BUY={actions.count(1)}, SELL={actions.count(2)}")

    print("RL Agent module ready.")
