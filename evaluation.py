"""
evaluation.py — Walk-Forward Analysis and Monte Carlo Simulation

Supports both ML (RandomForest) and RL (DQN) walk-forward evaluation.
Train → Test → Shift window → Repeat
"""

import pandas as pd
import numpy as np
import logging
from model import train_model, predict
from trading_engine import simulate_trading, simulate_trading_rl
from metrics import calculate_metrics

def walk_forward_analysis(df, feature_cols, train_window_days=365, test_window_days=90):
    """
    Performs walk-forward analysis for ML model.
    Slides a window over the data: trains on train_window, tests on test_window.
    """
    logging.info(f"Starting ML Walk-Forward Analysis (Train: {train_window_days}d, Test: {test_window_days}d)")
    
    all_trades = []
    
    start_idx = 0
    while True:
        start_date = df.index[start_idx]
        train_end_date = start_date + pd.Timedelta(days=train_window_days)
        test_end_date = train_end_date + pd.Timedelta(days=test_window_days)
        
        if test_end_date > df.index[-1]:
            # If the test window exceeds available data, do one last run to the end
            test_end_date = df.index[-1]
            if train_end_date >= test_end_date:
                break
        
        train_data = df[(df.index >= start_date) & (df.index < train_end_date)]
        test_data = df[(df.index >= train_end_date) & (df.index <= test_end_date)]
        
        if len(train_data) < 50 or len(test_data) < 5:
            break
            
        logging.info(f"ML Window: Train {start_date.date()} to {train_end_date.date()}, Test {train_end_date.date()} to {test_end_date.date()}")
        
        # Train model
        model = train_model(train_data, feature_cols)
        
        # Predict on test data
        preds, probs = predict(model, test_data, feature_cols)
        
        # Simulate trading on test data
        trades_df = simulate_trading(test_data, preds, probs)
        
        if not trades_df.empty:
            all_trades.append(trades_df)
            
        # Move window forward by test_window_days
        next_start_idx = np.searchsorted(df.index, train_end_date)
        start_idx = next_start_idx
        
        if start_idx >= len(df) or test_end_date == df.index[-1]:
            break
            
    if all_trades:
        combined_trades = pd.concat(all_trades, ignore_index=True)
        metrics, _ = calculate_metrics(combined_trades)
        logging.info("ML Walk-Forward Analysis Completed.")
        return metrics, combined_trades
    else:
        logging.warning("ML Walk-Forward Analysis generated no trades.")
        return {}, pd.DataFrame()


def walk_forward_analysis_rl(df, feature_cols, train_window_days=365, test_window_days=90,
                              total_timesteps=10000, pretrained_model=None):
    """
    Performs walk-forward analysis for RL agent (DQN).
    Same sliding window logic: Train DQN → Predict actions → Simulate → Shift.

    If pretrained_model is provided, skips training and uses the saved model
    for all windows (fast simulation mode).

    Args:
        df (pd.DataFrame): Feature-engineered dataframe.
        feature_cols (list): Feature column names.
        train_window_days (int): Training window in calendar days.
        test_window_days (int): Testing window in calendar days.
        total_timesteps (int): DQN training timesteps per window.
        pretrained_model: Optional pre-trained DQN model (skips training if provided).

    Returns:
        dict: Aggregated performance metrics.
        pd.DataFrame: Combined trades from all windows.
        dict: Training statistics from the last window (for UI display).
    """
    # Import here to avoid circular imports
    from rl_agent import train_rl_agent, predict_rl_actions

    use_pretrained = pretrained_model is not None
    mode_str = "FAST (pre-trained)" if use_pretrained else "FULL (train per window)"
    logging.info(f"Starting RL Walk-Forward Analysis [{mode_str}] "
                 f"(Train: {train_window_days}d, Test: {test_window_days}d, Timesteps: {total_timesteps})")

    all_trades = []
    last_training_stats = {}

    start_idx = 0
    window_count = 0

    while True:
        start_date = df.index[start_idx]
        train_end_date = start_date + pd.Timedelta(days=train_window_days)
        test_end_date = train_end_date + pd.Timedelta(days=test_window_days)

        if test_end_date > df.index[-1]:
            test_end_date = df.index[-1]
            if train_end_date >= test_end_date:
                break

        train_data = df[(df.index >= start_date) & (df.index < train_end_date)]
        test_data = df[(df.index >= train_end_date) & (df.index <= test_end_date)]

        if len(train_data) < 50 or len(test_data) < 5:
            break

        window_count += 1
        logging.info(f"RL Window {window_count}: Train {start_date.date()} to {train_end_date.date()}, "
                     f"Test {train_end_date.date()} to {test_end_date.date()}")

        if use_pretrained:
            # Use the pre-trained model (fast mode)
            model = pretrained_model
            last_training_stats = {
                'total_episodes': 0, 'avg_reward': 0.0,
                'best_reward': 0.0, 'worst_reward': 0.0,
                'avg_episode_length': 0,
                'total_timesteps': 0, 'algorithm': 'DQN',
                'mode': 'pre-trained'
            }
        else:
            # Train DQN agent on training window
            model, training_stats = train_rl_agent(train_data, feature_cols,
                                                    total_timesteps=total_timesteps)
            last_training_stats = training_stats

        # Generate actions on test data
        actions = predict_rl_actions(model, test_data, feature_cols)

        # Simulate trading with RL actions
        trades_df = simulate_trading_rl(test_data, actions)

        if not trades_df.empty:
            all_trades.append(trades_df)

        # Move window forward
        next_start_idx = np.searchsorted(df.index, train_end_date)
        start_idx = next_start_idx

        if start_idx >= len(df) or test_end_date == df.index[-1]:
            break

    if all_trades:
        combined_trades = pd.concat(all_trades, ignore_index=True)
        metrics, _ = calculate_metrics(combined_trades)
        logging.info(f"RL Walk-Forward Analysis Completed. Windows: {window_count}")
        return metrics, combined_trades, last_training_stats
    else:
        logging.warning("RL Walk-Forward Analysis generated no trades.")
        return {}, pd.DataFrame(), last_training_stats


def monte_carlo_simulation(trades_df, num_simulations=100, initial_capital=100000):
    """
    Shuffles trade sequence and runs multiple simulations to measure stability.
    """
    if trades_df.empty:
        return {}
        
    logging.info(f"Running Monte Carlo Simulation ({num_simulations} iterations)...")
    returns = trades_df['Net_PnL'].values
    
    final_capitals = []
    max_drawdowns = []
    
    for _ in range(num_simulations):
        # Sample with replacement to simulate different paths
        simulated_returns = np.random.choice(returns, size=len(returns), replace=True)
        
        equity_curve = initial_capital * (1 + simulated_returns).cumprod()
        final_capitals.append(equity_curve[-1])
        
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - running_max) / running_max
        max_drawdowns.append(np.min(drawdown))
        
    results = {
        'MC_Final_Capital_Mean': np.mean(final_capitals),
        'MC_Final_Capital_Std': np.std(final_capitals),
        'MC_Max_Drawdown_Mean': np.mean(max_drawdowns),
        'MC_Max_Drawdown_Worst': np.min(max_drawdowns),
        'MC_Win_Probability': sum(1 for cap in final_capitals if cap > initial_capital) / num_simulations
    }
    
    return results
