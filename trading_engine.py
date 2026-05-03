import pandas as pd
import logging

def simulate_trading(df, preds, probs, conf_threshold=0.55, 
                    stop_loss=-0.02, take_profit=0.03, 
                    trans_cost=0.001, slippage=0.0005):
    """
    Simulates trading realistically with risk controls, transaction costs, and logging.
    
    Args:
        df (pd.DataFrame): Dataframe with OHLC prices.
        preds (list or np.array): Model predictions (1 for BUY, 0 for SELL).
        probs (list or np.array): Confidence probabilities for BUY.
        conf_threshold (float): Minimum confidence to take a trade.
        stop_loss (float): Stop loss percentage (e.g., -0.02 for -2%).
        take_profit (float): Take profit percentage (e.g., 0.03 for +3%).
        trans_cost (float): Transaction cost per trade (e.g., 0.001 for 0.1%).
        slippage (float): Slippage per trade.
        
    Returns:
        pd.DataFrame: Trade history.
    """
    
    logging.info("Starting trading simulation...")
    trades = []
    
    in_position = False
    entry_price = 0.0
    entry_date = None
    entry_idx = 0
    
    # Costs applied on entry and exit
    total_friction = trans_cost + slippage
    
    for i in range(1, len(df)):
        current_date = df.index[i]
        today_open = df['Open'].iloc[i]
        today_high = df['High'].iloc[i]
        today_low = df['Low'].iloc[i]
        today_close = df['Close'].iloc[i]
        
        # Signal is generated at the end of PREVIOUS day
        prev_pred = preds[i-1]
        prev_prob = probs[i-1]
        
        if in_position:
            # Check SL and TP (using today's high/low for realistic intra-day hit)
            highest_return = (today_high - entry_price) / entry_price
            lowest_return = (today_low - entry_price) / entry_price
            
            exit_reason = None
            exit_price = 0.0
            
            if lowest_return <= stop_loss:
                # Conservative: assume SL hit before TP if both hit
                exit_price = entry_price * (1 + stop_loss)
                exit_reason = 'Stop-Loss'
            elif highest_return >= take_profit:
                exit_price = entry_price * (1 + take_profit)
                exit_reason = 'Take-Profit'
            elif prev_pred == 0:
                # Sell signal generated yesterday, exit at today's open
                exit_price = today_open
                exit_reason = 'Model-Sell'
                
            if exit_reason:
                # Apply friction to exit price
                actual_exit_price = exit_price * (1 - total_friction)
                raw_pnl = (exit_price - entry_price) / entry_price
                net_pnl = (actual_exit_price - entry_price) / entry_price
                
                trade_duration = i - entry_idx
                
                # RL-Inspired Reward Logic
                reward = calculate_reward(net_pnl, trade_duration, exit_reason)
                
                trades.append({
                    'Entry_Date': entry_date,
                    'Exit_Date': current_date,
                    'Entry_Price': entry_price, # Actual entry price including friction
                    'Exit_Price': actual_exit_price,
                    'Duration': trade_duration,
                    'Raw_PnL': raw_pnl,
                    'Net_PnL': net_pnl,
                    'Exit_Reason': exit_reason,
                    'Reward': reward
                })
                in_position = False
                
        else:
            # Check for entry signal
            # Enter if model says BUY and confidence > threshold
            if prev_pred == 1 and prev_prob >= conf_threshold:
                # Enter at today's open
                # Apply friction to entry price
                entry_price = today_open * (1 + total_friction)
                entry_date = current_date
                entry_idx = i
                in_position = True

    # Close any open position at the end of the simulation
    if in_position:
        last_close = df['Close'].iloc[-1]
        actual_exit_price = last_close * (1 - total_friction)
        net_pnl = (actual_exit_price - entry_price) / entry_price
        trade_duration = len(df) - 1 - entry_idx
        reward = calculate_reward(net_pnl, trade_duration, 'End-of-Data')
        
        trades.append({
            'Entry_Date': entry_date,
            'Exit_Date': df.index[-1],
            'Entry_Price': entry_price,
            'Exit_Price': actual_exit_price,
            'Duration': trade_duration,
            'Raw_PnL': (last_close - entry_price) / entry_price,
            'Net_PnL': net_pnl,
            'Exit_Reason': 'End-of-Data',
            'Reward': reward
        })
        
    logging.info(f"Simulation complete. Total trades: {len(trades)}")
    return pd.DataFrame(trades)

def calculate_reward(pnl, duration, exit_reason):
    """
    Simplified RL-inspired reward logic.
    Reward = Profit * Risk-Reward Factor - Penalties
    """
    reward = pnl * 100 # Base reward scaled to percentage points (e.g. 3% -> 3.0)
    
    # Bonus for hitting take-profit
    if exit_reason == 'Take-Profit':
        reward += 1.0 
        
    # Penalty: Very short trades (e.g., stopped out instantly)
    if duration <= 1 and pnl < 0:
        reward -= 0.5
        
    # Penalty: Long holding without profit
    if duration > 10 and pnl <= 0:
        reward -= 1.0
        
    return reward
