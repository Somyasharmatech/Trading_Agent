import numpy as np
import pandas as pd
import logging

def calculate_metrics(trades_df, initial_capital=100000):
    """
    Calculates basic and advanced performance metrics from trades.
    """
    if trades_df is None or trades_df.empty:
        return {}
        
    metrics = {}
    
    # Basic
    metrics['Total_Trades'] = len(trades_df)
    metrics['Winning_Trades'] = len(trades_df[trades_df['Net_PnL'] > 0])
    metrics['Losing_Trades'] = len(trades_df[trades_df['Net_PnL'] <= 0])
    metrics['Win_Rate'] = metrics['Winning_Trades'] / metrics['Total_Trades'] if metrics['Total_Trades'] > 0 else 0
    metrics['Avg_PnL'] = trades_df['Net_PnL'].mean()
    
    gross_profit = trades_df[trades_df['Net_PnL'] > 0]['Net_PnL'].sum()
    gross_loss = abs(trades_df[trades_df['Net_PnL'] < 0]['Net_PnL'].sum())
    metrics['Profit_Factor'] = gross_profit / gross_loss if gross_loss != 0 else np.inf
    
    # Equity Curve
    # Assuming full compounding for simplicity: Capital_t = Capital_{t-1} * (1 + Net_PnL)
    equity_curve = initial_capital * (1 + trades_df['Net_PnL']).cumprod()
    trades_df['Equity'] = equity_curve
    metrics['Final_Capital'] = equity_curve.iloc[-1] if not equity_curve.empty else initial_capital
    metrics['Total_Return'] = (metrics['Final_Capital'] - initial_capital) / initial_capital
    
    # Daily Returns equivalent (simplified by trade returns, assuming trades are spread evenly)
    # A more precise way is to look at daily equity, but this works for trade-level metrics.
    returns = trades_df['Net_PnL']
    
    # Advanced Metrics (using trade returns approximation)
    # Sharpe Ratio: Mean(Returns) / Std(Returns) * sqrt(Trades_Per_Year)
    # Assuming approx 50 trades per year as a scaler
    scaler = np.sqrt(50)
    metrics['Sharpe_Ratio'] = (returns.mean() / returns.std()) * scaler if returns.std() != 0 else 0
    
    # Sortino Ratio: Mean(Returns) / Downside_Deviation
    downside = returns[returns < 0]
    down_std = downside.std()
    metrics['Sortino_Ratio'] = (returns.mean() / down_std) * scaler if (not downside.empty and down_std != 0) else 0
    
    # Max Drawdown
    running_max = trades_df['Equity'].cummax()
    drawdowns = (trades_df['Equity'] - running_max) / running_max
    metrics['Max_Drawdown'] = drawdowns.min()
    
    # Calmar Ratio: Annual Return / Max Drawdown
    # Simplified annualization based on total duration
    if len(trades_df) > 0 and 'Exit_Date' in trades_df.columns:
        days = (pd.to_datetime(trades_df['Exit_Date'].iloc[-1]) - pd.to_datetime(trades_df['Entry_Date'].iloc[0])).days
        years = days / 365.25 if days > 0 else 1
        annual_return = (metrics['Final_Capital'] / initial_capital) ** (1 / years) - 1
        metrics['Calmar_Ratio'] = annual_return / abs(metrics['Max_Drawdown']) if metrics['Max_Drawdown'] != 0 else np.inf
    else:
        metrics['Calmar_Ratio'] = 0
        
    return metrics, trades_df

def compute_daily_equity(df, trades_df, initial_capital=100000):
    """
    Computes a daily equity curve.
    When in a trade, portfolio grows with price.
    When out of a trade, portfolio is flat (cash).
    """
    daily_equity = pd.Series(index=df.index, dtype=float)
    daily_equity.iloc[0] = initial_capital
    
    cash = initial_capital
    in_trade = False
    current_shares = 0
    
    if trades_df is None or trades_df.empty:
        daily_equity[:] = initial_capital
        return daily_equity
        
    # Mark entry and exits
    trade_idx = 0
    num_trades = len(trades_df)
    
    for i in range(1, len(df)):
        current_date = df.index[i]
        
        if trade_idx < num_trades:
            trade = trades_df.iloc[trade_idx]
            
            # Check for entry
            if current_date == trade['Entry_Date']:
                in_trade = True
                # Buy shares
                current_shares = cash / trade['Entry_Price']
                cash = 0
                
            # Update equity
            if in_trade:
                daily_equity.iloc[i] = current_shares * df['Close'].iloc[i]
            else:
                daily_equity.iloc[i] = cash
                
            # Check for exit
            if current_date == trade['Exit_Date']:
                in_trade = False
                cash = current_shares * trade['Exit_Price']
                current_shares = 0
                daily_equity.iloc[i] = cash
                trade_idx += 1
        else:
            daily_equity.iloc[i] = cash
            
    # Forward fill any missing values just in case
    daily_equity = daily_equity.ffill()
    return daily_equity

def get_baseline_performance(df, initial_capital=100000):
    """
    Calculates Buy & Hold baseline metrics.
    """
    start_price = df['Open'].iloc[0]
    end_price = df['Close'].iloc[-1]
    
    shares = initial_capital / start_price
    final_value = shares * end_price
    
    daily_returns = df['Close'].pct_change().dropna()
    sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    
    running_max = df['Close'].cummax()
    drawdown = (df['Close'] - running_max) / running_max
    max_dd = drawdown.min()
    
    metrics = {
        'Total_Return': (final_value - initial_capital) / initial_capital,
        'Final_Capital': final_value,
        'Sharpe_Ratio': sharpe,
        'Max_Drawdown': max_dd
    }
    
    # Daily equity for plotting
    equity_curve = (initial_capital / start_price) * df['Close']
    
    return metrics, equity_curve
