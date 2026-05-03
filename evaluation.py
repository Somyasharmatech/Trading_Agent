import pandas as pd
import numpy as np
import logging
from model import train_model, predict
from trading_engine import simulate_trading
from metrics import calculate_metrics

def walk_forward_analysis(df, feature_cols, train_window_days=365, test_window_days=90):
    """
    Performs walk-forward analysis.
    Slides a window over the data: trains on train_window, tests on test_window.
    """
    logging.info(f"Starting Walk-Forward Analysis (Train: {train_window_days}d, Test: {test_window_days}d)")
    
    all_trades = []
    
    start_idx = 0
    while True:
        # Approximate rows by trading days (252 days per year approx, so train=252, test=63)
        # We will use date offsets instead for accurate calendar days
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
            
        logging.info(f"Window: Train {start_date.date()} to {train_end_date.date()}, Test {train_end_date.date()} to {test_end_date.date()}")
        
        # Train model
        model = train_model(train_data, feature_cols)
        
        # Predict on test data
        preds, probs = predict(model, test_data, feature_cols)
        
        # Simulate trading on test data
        trades_df = simulate_trading(test_data, preds, probs)
        
        if not trades_df.empty:
            all_trades.append(trades_df)
            
        # Move window forward by test_window_days
        # Find the index of the first date >= train_end_date
        next_start_idx = np.searchsorted(df.index, train_end_date)
        start_idx = next_start_idx
        
        if start_idx >= len(df) or test_end_date == df.index[-1]:
            break
            
    if all_trades:
        combined_trades = pd.concat(all_trades, ignore_index=True)
        metrics, _ = calculate_metrics(combined_trades)
        logging.info("Walk-Forward Analysis Completed.")
        return metrics, combined_trades
    else:
        logging.warning("Walk-Forward Analysis generated no trades.")
        return {}, pd.DataFrame()

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
        # Sample with replacement or shuffle? Usually, we just shuffle the actual trades 
        # or sample with replacement to simulate different paths. Let's sample with replacement.
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
