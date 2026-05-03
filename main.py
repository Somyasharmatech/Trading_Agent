"""
main.py — Main Pipeline Orchestrator

Supports both ML (RandomForest) and RL (DQN) pipelines.
RL model persistence: train once → save → use for fast simulation.
Includes multi-stock testing capability.
"""

import logging
import pandas as pd
import numpy as np
from data_loader import load_data
from features import engineer_features
from evaluation import walk_forward_analysis, walk_forward_analysis_rl, monte_carlo_simulation
from metrics import get_baseline_performance, compute_daily_equity
from model import train_model, predict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def train_rl_only(ticker="AAPL", period="2y", rl_timesteps=10000):
    """
    Trains an RL agent on the data and saves it to disk.
    Uses last 1–2 years of data for faster, more relevant training.

    Args:
        ticker (str): Stock ticker symbol.
        period (str): Data period for training (default '2y' for speed).
        rl_timesteps (int): DQN training timesteps.

    Returns:
        dict: Training statistics.
    """
    from rl_agent import train_rl_agent

    logging.info(f"--- Training RL Model for {ticker} ({period} data, {rl_timesteps} timesteps) ---")

    df = load_data(ticker, period)
    if df.empty:
        return None

    df_features, feature_cols = engineer_features(df)

    model, training_stats = train_rl_agent(
        df_features, feature_cols,
        total_timesteps=rl_timesteps
    )

    training_stats['ticker'] = ticker
    training_stats['data_period'] = period
    training_stats['data_rows'] = len(df_features)

    return training_stats


def run_pipeline(ticker="AAPL", period="5y", train_window=365, test_window=90,
                 model_type="ML", rl_timesteps=10000, use_saved_model=False):
    """
    Runs the complete end-to-end trading pipeline.

    Args:
        ticker (str): Stock ticker symbol.
        period (str): Data period (e.g., '5y').
        train_window (int): Walk-forward train window in days.
        test_window (int): Walk-forward test window in days.
        model_type (str): 'ML' for RandomForest, 'RL' for DQN agent.
        rl_timesteps (int): DQN training timesteps.
        use_saved_model (bool): If True, loads saved RL model instead of retraining.

    Returns:
        dict: Results including trades, metrics, equity curves, and training details.
    """
    logging.info(f"--- Starting {model_type} Pipeline for {ticker} ---")

    # 1. Load Data
    df = load_data(ticker, period)
    if df.empty:
        logging.error("Failed to load data. Exiting pipeline.")
        return None

    # 2. Engineer Features (includes RSI)
    df_features, feature_cols = engineer_features(df)

    # 3. Baseline Performance (Buy & Hold)
    baseline_metrics, baseline_equity = get_baseline_performance(df_features)

    # 4. Run Walk-Forward Analysis based on model type
    training_details = {}

    if model_type == "RL":
        pretrained_model = None

        # Try to load saved model if requested
        if use_saved_model:
            from rl_agent import load_rl_model
            pretrained_model = load_rl_model()
            if pretrained_model:
                logging.info("Using saved RL model (fast simulation mode).")
            else:
                logging.warning("No saved model found. Will train per window.")

        metrics, trades_df, rl_training_stats = walk_forward_analysis_rl(
            df_features, feature_cols,
            train_window_days=train_window,
            test_window_days=test_window,
            total_timesteps=rl_timesteps,
            pretrained_model=pretrained_model
        )

        is_pretrained = rl_training_stats.get('mode') == 'pre-trained'

        training_details = {
            'model_type': 'RL',
            'algorithm': 'DQN',
            'timesteps': rl_training_stats.get('total_timesteps', rl_timesteps),
            'total_episodes': rl_training_stats.get('total_episodes', 0),
            'avg_reward': rl_training_stats.get('avg_reward', 0.0),
            'best_reward': rl_training_stats.get('best_reward', 0.0),
            'worst_reward': rl_training_stats.get('worst_reward', 0.0),
            'avg_episode_length': rl_training_stats.get('avg_episode_length', 0),
            'reward_formula': 'PnL × RR_bonus - overtrading - holding - cost',
            'mode': 'Pre-trained (saved model)' if is_pretrained else f'Fresh training ({rl_timesteps:,} steps)'
        }

    else:  # ML (default)
        metrics, trades_df = walk_forward_analysis(
            df_features, feature_cols,
            train_window_days=train_window,
            test_window_days=test_window
        )

        # Calculate ML accuracy from walk-forward
        if not trades_df.empty:
            winning = len(trades_df[trades_df['Net_PnL'] > 0])
            total = len(trades_df)
            accuracy_approx = winning / total if total > 0 else 0
        else:
            accuracy_approx = 0

        training_details = {
            'model_type': 'ML',
            'algorithm': 'RandomForest',
            'n_estimators': 100,
            'max_depth': 5,
            'accuracy': accuracy_approx,
        }

    if trades_df.empty:
        logging.warning("No trades executed during Walk-Forward Analysis.")
        return None

    # 5. Monte Carlo Simulation
    mc_results = monte_carlo_simulation(trades_df)
    metrics.update(mc_results)

    # 6. Daily Equity Curve
    daily_equity = compute_daily_equity(df_features, trades_df)

    results = {
        'df': df_features,
        'trades': trades_df,
        'metrics': metrics,
        'baseline_metrics': baseline_metrics,
        'daily_equity': daily_equity,
        'baseline_equity': baseline_equity,
        'training_details': training_details,
    }

    return results


def run_multi_stock_test(tickers, period="5y", train_window=365, test_window=90,
                         model_type="ML", rl_timesteps=10000):
    """
    Runs the pipeline on multiple tickers and aggregates results.
    """
    logging.info(f"--- Starting Multi-Stock Test ({model_type}) for {tickers} ---")

    all_results = []
    all_metrics = []

    for ticker in tickers:
        logging.info(f"Processing {ticker}...")
        try:
            result = run_pipeline(
                ticker=ticker, period=period,
                train_window=train_window, test_window=test_window,
                model_type=model_type, rl_timesteps=rl_timesteps
            )
            if result:
                all_results.append({
                    'ticker': ticker,
                    'metrics': result['metrics'],
                    'baseline_metrics': result['baseline_metrics'],
                    'training_details': result['training_details'],
                    'num_trades': len(result['trades']),
                })
                all_metrics.append(result['metrics'])
        except Exception as e:
            logging.warning(f"Failed for {ticker}: {e}")
            all_results.append({
                'ticker': ticker, 'metrics': {}, 'baseline_metrics': {},
                'training_details': {}, 'num_trades': 0, 'error': str(e)
            })

    # Aggregate average metrics
    avg_metrics = {}
    if all_metrics:
        metric_keys = ['Total_Trades', 'Win_Rate', 'Avg_PnL', 'Sharpe_Ratio',
                       'Sortino_Ratio', 'Max_Drawdown', 'Calmar_Ratio', 'Profit_Factor',
                       'Total_Return']
        for key in metric_keys:
            values = [m.get(key, 0) for m in all_metrics if m.get(key) is not None]
            if values:
                avg_metrics[key] = np.mean(values)

    return all_results, avg_metrics


def get_live_prediction(ticker="AAPL", model_type="ML"):
    """
    Trains the model on all available historical data up to yesterday,
    and makes a prediction for TOMORROW based on TODAY's features.
    For RL: uses saved model if available for instant prediction.
    """
    df = load_data(ticker, period="2y")
    if df.empty:
        return None

    df_live = df.copy()
    df_live['Returns'] = df_live['Close'].pct_change()
    df_live['Volatility_10'] = df_live['Returns'].rolling(window=10).std()

    # RSI (14-period)
    import ta
    df_live['RSI'] = ta.momentum.RSIIndicator(df_live['Close'], window=14).rsi()

    high_low_diff = df_live['High'] - df_live['Low']
    high_low_diff = high_low_diff.replace(0, pd.NA)

    df_live['Body_Ratio'] = (df_live['Close'] - df_live['Open']) / high_low_diff
    df_live['Upper_Wick'] = (df_live['High'] - df_live[['Open', 'Close']].max(axis=1)) / high_low_diff
    df_live['Lower_Wick'] = (df_live[['Open', 'Close']].min(axis=1) - df_live['Low']) / high_low_diff

    df_live['Body_Ratio'] = df_live['Body_Ratio'].fillna(0)
    df_live['Upper_Wick'] = df_live['Upper_Wick'].fillna(0)
    df_live['Lower_Wick'] = df_live['Lower_Wick'].fillna(0)

    df_live['SMA_10'] = df_live['Close'].rolling(window=10).mean()
    df_live['SMA_50'] = df_live['Close'].rolling(window=50).mean()

    df_live['SMA_10_Ratio'] = df_live['Close'] / df_live['SMA_10']
    df_live['SMA_50_Ratio'] = df_live['Close'] / df_live['SMA_50']

    feature_cols = ['Returns', 'Volatility_10', 'RSI', 'Body_Ratio', 'Upper_Wick',
                    'Lower_Wick', 'SMA_10_Ratio', 'SMA_50_Ratio']
    df_live = df_live.dropna(subset=feature_cols)

    df_live['Next_Close'] = df_live['Close'].shift(-1)
    df_live['Target'] = (df_live['Next_Close'] > df_live['Close']).astype(int)

    # Training set (all except today)
    train_df = df_live.iloc[:-1].copy()
    today_df = df_live.iloc[[-1]].copy()

    # Scale
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    today_df[feature_cols] = scaler.transform(today_df[feature_cols])

    if model_type == "RL":
        from rl_agent import load_rl_model, train_rl_agent, get_rl_live_prediction

        # Try loading saved model first (instant prediction)
        model = load_rl_model()
        if model is None:
            # No saved model — train a quick one
            model, stats = train_rl_agent(train_df, feature_cols, total_timesteps=5000)

        result = get_rl_live_prediction(model, today_df.iloc[0], feature_cols, in_position=False)

        action_emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}
        return {
            'prediction': result['action_name'],
            'confidence': None,
            'latest_close': today_df['Close'].iloc[0],
            'date': today_df.index[0].strftime('%Y-%m-%d'),
            'model_type': 'RL (DQN)',
            'emoji': action_emoji.get(result['action_name'], '⚪')
        }

    else:  # ML
        model = train_model(train_df, feature_cols)
        preds, probs = predict(model, today_df, feature_cols)

        prediction = "BUY" if preds[0] == 1 else "SELL / AVOID"
        confidence = probs[0] if preds[0] == 1 else 1 - probs[0]

        return {
            'prediction': prediction,
            'confidence': confidence,
            'latest_close': today_df['Close'].iloc[0],
            'date': today_df.index[0].strftime('%Y-%m-%d'),
            'model_type': 'ML (RandomForest)',
            'emoji': '🟢' if preds[0] == 1 else '🔴'
        }


if __name__ == "__main__":
    results = run_pipeline(model_type="ML")
    if results:
        print(f"ML Pipeline successful. Trades: {len(results['trades'])}")

    results_rl = run_pipeline(model_type="RL", rl_timesteps=5000)
    if results_rl:
        print(f"RL Pipeline successful. Trades: {len(results_rl['trades'])}")
