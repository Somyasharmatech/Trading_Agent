"""
trading_engine.py — Trade Simulation Engine

Supports both ML (prediction-based) and RL (action-based) trading simulation
with SL/TP logic, transaction costs, slippage, cooldown, trend filter,
and PDF-aligned reward calculation.
"""

import pandas as pd
import numpy as np
import logging

COOLDOWN_PERIOD = 3  # Days to wait after a trade before entering again


def _check_trend_bullish(df, i):
    """Trend filter: BUY only if SMA_10 > SMA_50."""
    if 'SMA_10' in df.columns and 'SMA_50' in df.columns:
        return df['SMA_10'].iloc[i] > df['SMA_50'].iloc[i]
    return True


def _check_trend_bearish(df, i):
    """Trend filter: SELL signals stronger when SMA_10 < SMA_50."""
    if 'SMA_10' in df.columns and 'SMA_50' in df.columns:
        return df['SMA_10'].iloc[i] < df['SMA_50'].iloc[i]
    return True


def simulate_trading(df, preds, probs, conf_threshold=0.60, 
                    stop_loss=-0.02, take_profit=0.03, 
                    trans_cost=0.001, slippage=0.0005):
    """
    Simulates trading for ML model (prediction-based).
    Includes trend filter and cooldown period for noise reduction.
    
    Args:
        df (pd.DataFrame): Dataframe with OHLC prices.
        preds (list or np.array): Model predictions (1 for BUY, 0 for SELL).
        probs (list or np.array): Confidence probabilities for BUY.
        conf_threshold (float): Minimum confidence to take a trade (0.60 for noise reduction).
        stop_loss (float): Stop loss percentage (e.g., -0.02 for -2%).
        take_profit (float): Take profit percentage (e.g., 0.03 for +3%).
        trans_cost (float): Transaction cost per trade (e.g., 0.001 for 0.1%).
        slippage (float): Slippage per trade.
        
    Returns:
        pd.DataFrame: Trade history.
    """
    
    logging.info("Starting ML trading simulation...")
    trades = []
    
    in_position = False
    entry_price = 0.0
    entry_date = None
    entry_idx = 0
    trade_count = 0
    cooldown_remaining = 0
    
    # Costs applied on entry and exit
    total_friction = trans_cost + slippage
    
    for i in range(1, len(df)):
        current_date = df.index[i]
        today_open = df['Open'].iloc[i]
        today_high = df['High'].iloc[i]
        today_low = df['Low'].iloc[i]
        today_close = df['Close'].iloc[i]
        
        # Decrement cooldown
        if cooldown_remaining > 0:
            cooldown_remaining -= 1
        
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
                trade_count += 1
                
                # PDF-aligned Reward Logic
                reward = calculate_reward(net_pnl, trade_count, trade_duration, exit_reason)
                
                trades.append({
                    'Entry_Date': entry_date,
                    'Exit_Date': current_date,
                    'Entry_Price': entry_price,
                    'Exit_Price': actual_exit_price,
                    'Duration': trade_duration,
                    'Raw_PnL': raw_pnl,
                    'Net_PnL': net_pnl,
                    'Exit_Reason': exit_reason,
                    'Reward': reward
                })
                in_position = False
                cooldown_remaining = COOLDOWN_PERIOD  # Start cooldown
                
        else:
            # Check for entry signal
            # Enter if: model says BUY, confidence > threshold, not in cooldown, trend is bullish
            if (prev_pred == 1 and prev_prob >= conf_threshold 
                and cooldown_remaining <= 0
                and _check_trend_bullish(df, i)):
                # Enter at today's open
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
        trade_count += 1
        reward = calculate_reward(net_pnl, trade_count, trade_duration, 'End-of-Data')
        
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
        
    logging.info(f"ML Simulation complete. Total trades: {len(trades)}")
    return pd.DataFrame(trades)


def simulate_trading_rl(df, actions, stop_loss=-0.02, take_profit=0.03,
                        trans_cost=0.001, slippage=0.0005):
    """
    Simulates trading for RL agent (action-based: 0=HOLD, 1=BUY, 2=SELL).
    Enforces valid action constraints, cooldown, and trend filter.

    Args:
        df (pd.DataFrame): Dataframe with OHLC prices.
        actions (list): Action sequence from RL agent (0=HOLD, 1=BUY, 2=SELL).
        stop_loss (float): Stop loss percentage (e.g., -0.02 for -2%).
        take_profit (float): Take profit percentage (e.g., 0.03 for +3%).
        trans_cost (float): Transaction cost per trade.
        slippage (float): Slippage per trade.

    Returns:
        pd.DataFrame: Trade history.
    """
    logging.info("Starting RL trading simulation...")
    trades = []

    in_position = False
    entry_price = 0.0
    entry_date = None
    entry_idx = 0
    trade_count = 0
    cooldown_remaining = 0

    total_friction = trans_cost + slippage

    for i in range(len(df)):
        current_date = df.index[i]
        today_open = df['Open'].iloc[i]
        today_high = df['High'].iloc[i]
        today_low = df['Low'].iloc[i]
        today_close = df['Close'].iloc[i]

        # Decrement cooldown
        if cooldown_remaining > 0:
            cooldown_remaining -= 1

        action = actions[i]

        # --- Enforce valid action constraints ---
        if action == 1 and in_position:
            action = 0  # Cannot BUY if already in position
        if action == 2 and not in_position:
            action = 0  # Cannot SELL if no position
        if action == 1 and cooldown_remaining > 0:
            action = 0  # Cannot BUY during cooldown
        if action == 1 and not _check_trend_bullish(df, i):
            action = 0  # Trend filter: BUY only in uptrend

        if in_position:
            # Check SL and TP first (before processing action)
            highest_return = (today_high - entry_price) / entry_price
            lowest_return = (today_low - entry_price) / entry_price

            exit_reason = None
            exit_price = 0.0

            if lowest_return <= stop_loss:
                exit_price = entry_price * (1 + stop_loss)
                exit_reason = 'Stop-Loss'
            elif highest_return >= take_profit:
                exit_price = entry_price * (1 + take_profit)
                exit_reason = 'Take-Profit'
            elif action == 2:
                exit_price = today_close
                exit_reason = 'RL-Sell'

            if exit_reason:
                actual_exit_price = exit_price * (1 - total_friction)
                raw_pnl = (exit_price - entry_price) / entry_price
                net_pnl = (actual_exit_price - entry_price) / entry_price
                trade_duration = i - entry_idx
                trade_count += 1

                reward = calculate_reward(net_pnl, trade_count, trade_duration, exit_reason)

                trades.append({
                    'Entry_Date': entry_date,
                    'Exit_Date': current_date,
                    'Entry_Price': entry_price,
                    'Exit_Price': actual_exit_price,
                    'Duration': trade_duration,
                    'Raw_PnL': raw_pnl,
                    'Net_PnL': net_pnl,
                    'Exit_Reason': exit_reason,
                    'Reward': reward,
                    'Action': 'SELL'
                })
                in_position = False
                cooldown_remaining = COOLDOWN_PERIOD

        if not in_position and action == 1:
            # Enter position
            entry_price = today_close * (1 + total_friction)
            entry_date = current_date
            entry_idx = i
            in_position = True

    # Close any open position at the end
    if in_position:
        last_close = df['Close'].iloc[-1]
        actual_exit_price = last_close * (1 - total_friction)
        net_pnl = (actual_exit_price - entry_price) / entry_price
        trade_duration = len(df) - 1 - entry_idx
        trade_count += 1
        reward = calculate_reward(net_pnl, trade_count, trade_duration, 'End-of-Data')

        trades.append({
            'Entry_Date': entry_date,
            'Exit_Date': df.index[-1],
            'Entry_Price': entry_price,
            'Exit_Price': actual_exit_price,
            'Duration': trade_duration,
            'Raw_PnL': (last_close - entry_price) / entry_price,
            'Net_PnL': net_pnl,
            'Exit_Reason': 'End-of-Data',
            'Reward': reward,
            'Action': 'END'
        })

    logging.info(f"RL Simulation complete. Total trades: {len(trades)}")
    return pd.DataFrame(trades)


def calculate_reward(pnl, trade_count, holding_time, exit_reason,
                     tp=0.03, sl=0.02, trans_cost=0.001):
    """
    PDF-aligned reward function.

    reward = (PnL × RR_bonus)
             - overtrading_penalty
             - holding_penalty
             - transaction_cost

    Args:
        pnl (float): Net profit/loss of the trade.
        trade_count (int): Total number of trades so far (for overtrading penalty).
        holding_time (int): Duration of trade in bars.
        exit_reason (str): How the trade was closed.
        tp (float): Take-profit threshold.
        sl (float): Stop-loss threshold.
        trans_cost (float): Transaction cost.

    Returns:
        float: Calculated reward value.
    """
    rr_factor = tp / sl  # 1.5

    # Bonus if take-profit hit, reduced bonus otherwise
    if exit_reason == "Take-Profit":
        rr_bonus = rr_factor
    else:
        rr_bonus = 0.5

    reward = (pnl * rr_bonus) \
             - (0.001 * trade_count) \
             - (0.0005 * holding_time) \
             - trans_cost

    return reward
