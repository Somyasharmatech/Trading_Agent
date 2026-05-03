import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def engineer_features(df):
    """
    Engineers advanced financial features including price features, 
    candle structure, and trend features.
    
    Args:
        df (pd.DataFrame): Input dataframe with OHLCV data.
        
    Returns:
        pd.DataFrame: DataFrame with engineered features and target.
        list: List of feature column names.
    """
    df = df.copy()
    
    # 1. Price Features
    df['Returns'] = df['Close'].pct_change()
    df['Volatility_10'] = df['Returns'].rolling(window=10).std()
    
    # Handle division by zero for candle features
    high_low_diff = df['High'] - df['Low']
    high_low_diff = high_low_diff.replace(0, np.nan) # Avoid division by zero
    
    # 2. Candle Structure Features
    df['Body_Ratio'] = (df['Close'] - df['Open']) / high_low_diff
    df['Upper_Wick'] = (df['High'] - df[['Open', 'Close']].max(axis=1)) / high_low_diff
    df['Lower_Wick'] = (df[['Open', 'Close']].min(axis=1) - df['Low']) / high_low_diff
    
    # Fill NaN from div by zero with 0 (doji candles)
    df['Body_Ratio'] = df['Body_Ratio'].fillna(0)
    df['Upper_Wick'] = df['Upper_Wick'].fillna(0)
    df['Lower_Wick'] = df['Lower_Wick'].fillna(0)
    
    # 3. Trend Features
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # SMA position relative to price (normalized trend feature)
    df['SMA_10_Ratio'] = df['Close'] / df['SMA_10']
    df['SMA_50_Ratio'] = df['Close'] / df['SMA_50']
    
    # 4. Target Creation
    # Target: 1 if next day's close > today's close, else 0
    # No leakage: We shift the 'Close' backwards by 1 to align tomorrow's close with today's row
    df['Next_Close'] = df['Close'].shift(-1)
    df['Target'] = (df['Next_Close'] > df['Close']).astype(int)
    
    # Drop rows with NaN (due to rolling windows, pct_change, and shift)
    # This automatically removes the last row which has no 'Next_Close'
    df = df.dropna()
    
    # Define features to use
    feature_cols = [
        'Returns', 'Volatility_10', 'Body_Ratio', 'Upper_Wick', 'Lower_Wick', 
        'SMA_10_Ratio', 'SMA_50_Ratio'
    ]
    
    # 5. Normalization
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    return df, feature_cols

if __name__ == "__main__":
    from data_loader import load_data
    df = load_data()
    if not df.empty:
        df_features, f_cols = engineer_features(df)
        print(f"Engineered {len(df_features)} rows. Features: {f_cols}")
        print(df_features[f_cols + ['Target']].head())
