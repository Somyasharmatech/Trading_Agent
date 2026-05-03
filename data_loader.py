import yfinance as yf
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(ticker="AAPL", period="5y"):
    """
    Fetches stock market data using yfinance.
    
    Args:
        ticker (str): Stock ticker symbol.
        period (str): Timeframe for the data (e.g., '5y', '1y').
        
    Returns:
        pd.DataFrame: DataFrame with Open, High, Low, Close, Volume.
    """
    logging.info(f"Fetching data for {ticker} over the last {period}...")
    try:
        data = yf.download(ticker, period=period, interval="1d", progress=False)
        
        # If MultiIndex (newer yfinance versions when multiple tickers or sometimes just one), flatten it
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
            
        # Select required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in data.columns for col in required_cols):
            logging.error(f"Missing required columns. Available: {data.columns}")
            raise ValueError("Data is missing one of Open, High, Low, Close, Volume.")
            
        data = data[required_cols]
        
        # Sort by date
        data = data.sort_index()
        
        # Remove missing values
        data = data.dropna()
        
        logging.info(f"Successfully loaded {len(data)} rows of data for {ticker}.")
        return data
        
    except Exception as e:
        logging.error(f"Error fetching data: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    df = load_data()
    print(df.head())
