import logging
import pandas as pd
from data_loader import load_data
from features import engineer_features
from evaluation import walk_forward_analysis, monte_carlo_simulation
from metrics import get_baseline_performance, compute_daily_equity
from model import train_model, predict

def run_pipeline(ticker="AAPL", period="5y", train_window=365, test_window=90):
    """
    Runs the complete end-to-end trading pipeline (Backtest).
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
    
    return results

def get_live_prediction(ticker="AAPL"):
    """
    Trains the model on all available historical data up to yesterday,
    and makes a prediction for TOMORROW based on TODAY's features.
    """
    df = load_data(ticker, period="2y")
    if df.empty:
        return None
        
    df_live = df.copy()
    df_live['Returns'] = df_live['Close'].pct_change()
    df_live['Volatility_10'] = df_live['Returns'].rolling(window=10).std()
    
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
    
    feature_cols = ['Returns', 'Volatility_10', 'Body_Ratio', 'Upper_Wick', 'Lower_Wick', 'SMA_10_Ratio', 'SMA_50_Ratio']
    df_live = df_live.dropna(subset=feature_cols)
    
    df_live['Next_Close'] = df_live['Close'].shift(-1)
    df_live['Target'] = (df_live['Next_Close'] > df_live['Close']).astype(int)
    
    # Training set (all except today)
    train_df = df_live.iloc[:-1].copy()
    
    # Today's features (last row)
    today_df = df_live.iloc[[-1]].copy()
    
    # Scale
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    today_df[feature_cols] = scaler.transform(today_df[feature_cols])
    
    # Train
    model = train_model(train_df, feature_cols)
    
    # Predict
    preds, probs = predict(model, today_df, feature_cols)
    
    prediction = "BUY" if preds[0] == 1 else "SELL / AVOID"
    confidence = probs[0] if preds[0] == 1 else 1 - probs[0]
    
    return {
        'prediction': prediction,
        'confidence': confidence,
        'latest_close': today_df['Close'].iloc[0],
        'date': today_df.index[0].strftime('%Y-%m-%d')
    }

if __name__ == "__main__":
    results = run_pipeline()
    if results:
        print("Pipeline successful.")
