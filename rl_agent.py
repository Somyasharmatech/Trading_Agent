"""
rl_agent.py — DQN Reinforcement Learning Agent Wrapper

Uses stable-baselines3 DQN to train an agent on the custom TradingEnv.
Provides training with logging, prediction, and live signal generation.
"""

import logging
import numpy as np
import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from rl_env import TradingEnv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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
                   stop_loss=0.02, take_profit=0.03, trans_cost=0.001):
    """
    Trains a DQN agent on the trading environment.

    Args:
        df (pd.DataFrame): Feature-engineered dataframe with OHLCV + features.
        feature_cols (list): List of feature column names.
        total_timesteps (int): Number of training timesteps for DQN.
        stop_loss (float): Stop-loss threshold.
        take_profit (float): Take-profit threshold.
        trans_cost (float): Transaction cost per trade.

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

    # Create DQN model
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        buffer_size=10000,
        learning_starts=500,
        batch_size=64,
        gamma=0.99,
        exploration_fraction=0.3,
        exploration_final_eps=0.05,
        verbose=0
    )

    # Train
    model.learn(total_timesteps=total_timesteps, callback=logger_callback)

    # Get training stats
    training_stats = logger_callback.get_training_stats()
    training_stats['total_timesteps'] = total_timesteps
    training_stats['algorithm'] = 'DQN'

    logging.info(f"DQN Training complete. Episodes: {training_stats['total_episodes']}, "
                 f"Avg Reward: {training_stats['avg_reward']:.4f}")

    return model, training_stats


def predict_rl_actions(model, df, feature_cols):
    """
    Generates action sequence for each bar using the trained DQN model.

    Args:
        model (DQN): Trained DQN model.
        df (pd.DataFrame): Feature-engineered dataframe.
        feature_cols (list): Feature column names.

    Returns:
        list: Action sequence (0=HOLD, 1=BUY, 2=SELL) for each bar.
    """
    actions = []
    in_position = False  # Track position state for observation

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

        # Update position tracking
        if action == 1:
            in_position = True
        elif action == 2:
            in_position = False

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

    model, stats = train_rl_agent(df_feat, feat_cols, total_timesteps=1000)
    print(f"\nTraining Stats: {stats}")

    actions = predict_rl_actions(model, df_feat, feat_cols)
    print(f"\nAction distribution: HOLD={actions.count(0)}, BUY={actions.count(1)}, SELL={actions.count(2)}")
    print("RL Agent module ready.")
