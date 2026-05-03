"""
rl_env.py — Custom Gymnasium Trading Environment for DQN Agent

State Space (8 features):
    Returns, Volatility_10, Body_Ratio, Upper_Wick, Lower_Wick,
    SMA_10_Ratio, SMA_50_Ratio, Position (0=flat, 1=long)

Action Space (3 discrete):
    0 = HOLD
    1 = BUY
    2 = SELL

Reward Function (PDF-aligned):
    reward = (PnL × RR_bonus) - overtrading_penalty - holding_penalty - transaction_cost
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import logging

class TradingEnv(gym.Env):
    """
    A custom Gymnasium environment for simulating stock trading.
    The agent learns to BUY, SELL, or HOLD based on engineered features.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, df, feature_cols, stop_loss=0.02, take_profit=0.03,
                 trans_cost=0.001, initial_capital=100000):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.trans_cost = trans_cost
        self.initial_capital = initial_capital

        self.n_steps = len(self.df)

        # Action space: 0=HOLD, 1=BUY, 2=SELL
        self.action_space = spaces.Discrete(3)

        # Observation space: 7 market features + 1 position state
        # Position state: 0 = no position, 1 = long position
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(len(feature_cols) + 1,),  # +1 for position state
            dtype=np.float32
        )

        # Internal state
        self.current_step = 0
        self.in_position = False
        self.entry_price = 0.0
        self.entry_step = 0
        self.capital = initial_capital
        self.trade_count = 0
        self.trades = []

    def _get_observation(self):
        """Returns current observation: market features + position state."""
        market_features = self.df[self.feature_cols].iloc[self.current_step].values.astype(np.float32)
        position_state = np.array([1.0 if self.in_position else 0.0], dtype=np.float32)
        return np.concatenate([market_features, position_state])

    def reset(self, seed=None, options=None):
        """Reset environment to initial state."""
        super().reset(seed=seed)
        self.current_step = 0
        self.in_position = False
        self.entry_price = 0.0
        self.entry_step = 0
        self.capital = self.initial_capital
        self.trade_count = 0
        self.trades = []
        return self._get_observation(), {}

    def _calculate_reward(self, pnl, holding_time, exit_reason):
        """
        PDF-aligned reward function.

        reward = (PnL × RR_bonus)
                 - overtrading_penalty
                 - holding_penalty
                 - transaction_cost
        """
        rr_factor = self.take_profit / self.stop_loss  # 1.5

        # Bonus if take-profit hit, reduced bonus otherwise
        if exit_reason == "Take-Profit":
            rr_bonus = rr_factor
        else:
            rr_bonus = 0.5

        reward = (pnl * rr_bonus) \
                 - (0.001 * self.trade_count) \
                 - (0.0005 * holding_time) \
                 - self.trans_cost

        return reward

    def step(self, action):
        """Execute one time step within the environment."""
        reward = 0.0
        done = False
        truncated = False
        info = {}

        current_close = self.df['Close'].iloc[self.current_step]

        # --- Enforce valid action constraints ---
        # Cannot BUY if already in position → treat as HOLD
        if action == 1 and self.in_position:
            action = 0
        # Cannot SELL if no position → treat as HOLD
        if action == 2 and not self.in_position:
            action = 0

        if action == 1:  # BUY
            self.in_position = True
            self.entry_price = current_close * (1 + self.trans_cost)  # Entry with cost
            self.entry_step = self.current_step
            self.trade_count += 1
            reward = -self.trans_cost  # Small penalty for transaction cost on entry

        elif action == 2 and self.in_position:  # SELL
            exit_price = current_close * (1 - self.trans_cost)  # Exit with cost
            pnl = (exit_price - self.entry_price) / self.entry_price
            holding_time = self.current_step - self.entry_step

            reward = self._calculate_reward(pnl, holding_time, "Model-Sell")

            self.trades.append({
                'entry_step': self.entry_step,
                'exit_step': self.current_step,
                'entry_price': self.entry_price,
                'exit_price': exit_price,
                'pnl': pnl,
                'holding_time': holding_time,
                'exit_reason': 'Model-Sell',
                'reward': reward
            })

            self.capital *= (1 + pnl)
            self.in_position = False

        elif action == 0:  # HOLD
            if self.in_position:
                # Check stop-loss and take-profit
                unrealized_pnl = (current_close - self.entry_price) / self.entry_price
                holding_time = self.current_step - self.entry_step

                if unrealized_pnl <= -self.stop_loss:
                    # Stop-loss hit
                    exit_price = self.entry_price * (1 - self.stop_loss)
                    actual_exit = exit_price * (1 - self.trans_cost)
                    pnl = (actual_exit - self.entry_price) / self.entry_price

                    reward = self._calculate_reward(pnl, holding_time, "Stop-Loss")

                    self.trades.append({
                        'entry_step': self.entry_step,
                        'exit_step': self.current_step,
                        'entry_price': self.entry_price,
                        'exit_price': actual_exit,
                        'pnl': pnl,
                        'holding_time': holding_time,
                        'exit_reason': 'Stop-Loss',
                        'reward': reward
                    })

                    self.capital *= (1 + pnl)
                    self.in_position = False

                elif unrealized_pnl >= self.take_profit:
                    # Take-profit hit
                    exit_price = self.entry_price * (1 + self.take_profit)
                    actual_exit = exit_price * (1 - self.trans_cost)
                    pnl = (actual_exit - self.entry_price) / self.entry_price

                    reward = self._calculate_reward(pnl, holding_time, "Take-Profit")

                    self.trades.append({
                        'entry_step': self.entry_step,
                        'exit_step': self.current_step,
                        'entry_price': self.entry_price,
                        'exit_price': actual_exit,
                        'pnl': pnl,
                        'holding_time': holding_time,
                        'exit_reason': 'Take-Profit',
                        'reward': reward
                    })

                    self.capital *= (1 + pnl)
                    self.in_position = False
                else:
                    # Small holding penalty to discourage doing nothing forever
                    reward = -0.0005

            else:
                # Not in position, just holding cash — tiny neutral reward
                reward = 0.0

        # Advance step
        self.current_step += 1

        # Check if episode is done
        if self.current_step >= self.n_steps - 1:
            done = True
            # Force close any open position at end
            if self.in_position:
                exit_price = self.df['Close'].iloc[self.current_step] * (1 - self.trans_cost)
                pnl = (exit_price - self.entry_price) / self.entry_price
                holding_time = self.current_step - self.entry_step

                reward = self._calculate_reward(pnl, holding_time, "End-of-Data")

                self.trades.append({
                    'entry_step': self.entry_step,
                    'exit_step': self.current_step,
                    'entry_price': self.entry_price,
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'holding_time': holding_time,
                    'exit_reason': 'End-of-Data',
                    'reward': reward
                })

                self.capital *= (1 + pnl)
                self.in_position = False

        obs = self._get_observation() if not done else np.zeros(self.observation_space.shape, dtype=np.float32)
        info['capital'] = self.capital
        info['trade_count'] = self.trade_count

        return obs, reward, done, truncated, info


if __name__ == "__main__":
    # Quick test
    from data_loader import load_data
    from features import engineer_features

    df = load_data("AAPL", "1y")
    df_feat, feat_cols = engineer_features(df)

    env = TradingEnv(df_feat, feat_cols)
    obs, _ = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Sample observation: {obs}")

    # Take a few random actions
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(f"Step {i+1}: Action={action}, Reward={reward:.4f}, Capital={info['capital']:.2f}")
    print("RL Environment module ready.")
