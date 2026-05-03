import logging
import pandas as pd
from data_loader import load_data
from features import engineer_features
from evaluation import walk_forward_analysis, monte_carlo_simulation
from metrics import get_baseline_performance, compute_daily_equity

def run_pipeline(ticker="AAPL", period="5y", train_window=365, test_window=90):
    """
    Runs the complete end-to-end trading pipeline.
    """
    logging.info(f"--- Starting Pipeline for {ticker} ---")
    
    # 1. Load Data
    df = load_data(ticker, period)
    if df.empty:
        logging.error("Failed to load data. Exiting pipeline.")
        return None
        
    # 2. Engineer Features
    df_features, feature_cols = engineer_features(df)
    
    # 3. Baseline Performance
    baseline_metrics, baseline_equity = get_baseline_performance(df_features)
    logging.info(f"Baseline Return: {baseline_metrics['Total_Return']:.2%}")
    logging.info(f"Baseline Max Drawdown: {baseline_metrics['Max_Drawdown']:.2%}")
    
    # 4. Walk-Forward Analysis (Simulating real-world deployment)
    metrics, trades_df = walk_forward_analysis(
        df_features, 
        feature_cols, 
        train_window_days=train_window, 
        test_window_days=test_window
    )
    
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
        'baseline_equity': baseline_equity
    }
    
    logging.info(f"--- Pipeline Completed for {ticker} ---")
    logging.info(f"Total Trades: {metrics.get('Total_Trades', 0)}")
    logging.info(f"Win Rate: {metrics.get('Win_Rate', 0):.2%}")
    logging.info(f"Total Return: {metrics.get('Total_Return', 0):.2%}")
    logging.info(f"Sharpe Ratio: {metrics.get('Sharpe_Ratio', 0):.2f}")
    logging.info(f"Max Drawdown: {metrics.get('Max_Drawdown', 0):.2%}")
    
    return results

if __name__ == "__main__":
    results = run_pipeline()
    if results:
        print("\n--- Summary Metrics ---")
        for k, v in results['metrics'].items():
            if isinstance(v, float):
                print(f"{k}: {v:.4f}")
            else:
                print(f"{k}: {v}")
