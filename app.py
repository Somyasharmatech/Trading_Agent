"""
app.py — Streamlit Dashboard for AI Trading Agent

Features:
- Model selector: ML (RandomForest) vs RL (DQN)
- Live prediction for next trading day
- Walk-forward backtesting simulation
- Performance metrics (Sharpe, Sortino, Drawdown, Calmar, Profit Factor)
- Strategy vs Buy & Hold comparison
- Price + Signals chart, Equity curve, Drawdown chart
- Trade history table with CSV export
- Training details visibility (ML accuracy / RL episodes, reward)
- Model explanation panel
- Performance disclaimer
- Multi-stock testing (bonus)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from main import run_pipeline, get_live_prediction, run_multi_stock_test

st.set_page_config(page_title="AI Trading Agent", layout="wide", page_icon="🤖")

st.title("🤖 AI Trading Agent with Advanced Risk Logic")
st.markdown("AI-driven trading system using **machine learning** and **reinforcement learning**, with risk-aware logic and backtesting.")

# --- SIDEBAR (LEFT PANEL) ---
st.sidebar.header("⚙️ Simulation Parameters")
ticker = st.sidebar.text_input("Ticker Symbol", value="AAPL")
period = st.sidebar.selectbox("Data Period", options=["1y", "2y", "5y", "10y"], index=2)

st.sidebar.markdown("---")

# 🔥 MODEL SELECTOR
st.sidebar.header("🧠 Model Selection")
model_type = st.sidebar.radio(
    "Model Type",
    options=["ML Model", "RL Agent"],
    index=0,
    help="ML Model uses RandomForest classifier. RL Agent uses DQN reinforcement learning."
)

# Map display name to internal key
model_key = "ML" if model_type == "ML Model" else "RL"

# RL-specific controls
rl_timesteps = 30000
if model_key == "RL":
    rl_timesteps = st.sidebar.slider("DQN Training Timesteps", 5000, 50000, 30000, step=5000,
                                      help="More timesteps = better learning but slower training. Recommended: 30,000+")

st.sidebar.markdown("---")
st.sidebar.header("📐 Walk-Forward Settings")
train_window = st.sidebar.slider("Train Window (days)", 100, 1000, 365)
test_window = st.sidebar.slider("Test Window (days)", 30, 365, 90)

# --- MODEL EXPLANATION PANEL ---
with st.expander("🧠 Model Explanation — How This System Works", expanded=False):
    st.markdown("""
    ### How This Trading System Works

    This system uses **two AI approaches** to learn trading decisions from historical stock data:

    **📊 Data Pipeline:**
    - Historical stock data is fetched from Yahoo Finance via `yfinance`
    - Advanced features are engineered: **Returns**, **Volatility**, **RSI**, **Candle Structure** (body ratio, wicks), **SMA indicators**
    - Data is normalized using StandardScaler to ensure fair feature weighting

    **🤖 ML Model (RandomForest):**
    - A **RandomForestClassifier** (100 estimators, max depth 5) learns to predict next-day price direction
    - Prediction confidence must exceed **60%** to trigger a trade (noise reduction)
    - Best for: Stable signals based on feature patterns

    **🧠 RL Agent (DQN — Deep Q-Network):**
    - A **reinforcement learning agent** learns BUY/SELL/HOLD actions through trial and error
    - The agent receives **rewards** based on: `PnL × Risk-Reward Bonus - Overtrading Penalty - Holding Penalty - Transaction Cost`
    - State includes **9 features + position state** (whether currently holding)
    - Best for: Adaptive behavior that optimizes for cumulative reward

    **🛡️ Risk Management:**
    - **Stop-Loss:** -2% (automatic exit to limit losses)
    - **Take-Profit:** +3% (automatic exit to lock in gains)
    - **Transaction Cost:** 0.1% per trade (realistic friction)
    - **3-Day Cooldown:** Prevents overtrading after each exit
    - **Trend Filter:** BUY only allowed when SMA_10 > SMA_50 (uptrend confirmation)

    **📈 Evaluation:**
    - **Walk-Forward Analysis:** Train on 1 year → Test on 3 months → Shift → Repeat
    - **Monte Carlo Simulation:** Shuffles trade order 100 times to test robustness
    - **Metrics:** Sharpe, Sortino, Max Drawdown, Calmar, Profit Factor, Win Rate
    """)

# --- PERFORMANCE DISCLAIMER ---
st.info(
    "⚠️ **Disclaimer:** Financial markets are noisy and difficult to predict. "
    "This system focuses on learning trading behavior and risk management rather than guaranteed profits. "
    "Past performance does not guarantee future results."
)

st.markdown("---")

# --- LIVE PREDICTION ---
st.markdown("### 🔮 Live Prediction for Next Trading Day")
st.caption(f"Using: **{model_type}**")

with st.spinner(f"Generating live prediction for {ticker} using {model_type}..."):
    live_pred = get_live_prediction(ticker, model_type=model_key)

if live_pred:
    emoji = live_pred.get('emoji', '⚪')

    if live_pred.get('confidence') is not None:
        st.success(
            f"**{ticker} ({live_pred['date']}) Last Close:** ${live_pred['latest_close']:.2f}  |  "
            f"**Prediction:** {emoji} {live_pred['prediction']}  |  "
            f"**Confidence:** {live_pred['confidence']:.2%}  |  "
            f"**Model:** {live_pred['model_type']}"
        )
    else:
        st.success(
            f"**{ticker} ({live_pred['date']}) Last Close:** ${live_pred['latest_close']:.2f}  |  "
            f"**Action:** {emoji} {live_pred['prediction']}  |  "
            f"**Model:** {live_pred['model_type']}"
        )
else:
    st.error("Invalid ticker or data not available. Please try another symbol.")

st.markdown("---")

# --- MAIN SIMULATION ---
run_col, multi_col = st.columns(2)

with run_col:
    run_sim = st.button("🚀 Run Historical Simulation", type="primary", use_container_width=True)

with multi_col:
    run_multi = st.button("🌍 Multi-Stock Test (5 Tickers)", use_container_width=True)


# --- SINGLE STOCK SIMULATION ---
if run_sim:
    with st.status(f"Running {model_type} simulation...", expanded=True) as status:
        st.write("📥 Fetching data...")
        st.write(f"🧠 Training {'DQN agent (' + str(rl_timesteps) + ' timesteps)' if model_key == 'RL' else 'RandomForest model'}...")
        st.write("📊 Backtesting strategy (Walk-Forward)...")
        st.write("📈 Calculating metrics...")

        results = run_pipeline(
            ticker, period, train_window, test_window,
            model_type=model_key,
            rl_timesteps=rl_timesteps
        )

        status.update(label="✅ Simulation Complete!", state="complete", expanded=False)

    if results is None:
        st.error("Invalid ticker or data not available. No trades were generated.")
    else:
        metrics = results['metrics']
        baseline = results['baseline_metrics']
        trades_df = results['trades']
        df = results['df']
        daily_equity = results['daily_equity']
        baseline_equity = results['baseline_equity']
        training_details = results.get('training_details', {})

        # --- SECTION 1: PERFORMANCE METRICS ---
        st.subheader("📊 Performance Summary")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Trades", metrics.get('Total_Trades', 0))
        col2.metric("Win Rate", f"{metrics.get('Win_Rate', 0):.2%}")
        col3.metric("Average P/L", f"{metrics.get('Avg_PnL', 0):.2%}")
        col4.metric("Sharpe Ratio", f"{metrics.get('Sharpe_Ratio', 0):.2f}")

        col5, col6, col7, col8 = st.columns(4)
        col5.metric("Sortino Ratio", f"{metrics.get('Sortino_Ratio', 0):.2f}")
        col6.metric("Max Drawdown", f"{metrics.get('Max_Drawdown', 0):.2%}")
        col7.metric("Calmar Ratio", f"{metrics.get('Calmar_Ratio', 0):.2f}")
        col8.metric("Profit Factor", f"{metrics.get('Profit_Factor', 0):.2f}")

        st.markdown("---")

        # --- SECTION 2: STRATEGY vs BUY & HOLD ---
        st.subheader("⚖️ Strategy vs Buy & Hold Comparison")
        strat_return = metrics.get('Total_Return', 0)
        bh_return = baseline.get('Total_Return', 0)
        diff = strat_return - bh_return

        c1, c2, c3 = st.columns(3)
        c1.metric("Strategy Return", f"{strat_return:.2%}")
        c2.metric("Buy & Hold Return", f"{bh_return:.2%}")
        c3.metric("Outperformance", f"{diff:.2%}", delta=f"{diff:.2%}")

        bc1, bc2, bc3, bc4 = st.columns(4)
        bc1.metric("Strategy Sharpe", f"{metrics.get('Sharpe_Ratio', 0):.2f}")
        bc2.metric("B&H Sharpe", f"{baseline.get('Sharpe_Ratio', 0):.2f}")
        bc3.metric("Strategy Max DD", f"{metrics.get('Max_Drawdown', 0):.2%}")
        bc4.metric("B&H Max DD", f"{baseline.get('Max_Drawdown', 0):.2%}")

        st.markdown("---")

        # --- SECTION 3: TRAINING DETAILS ---
        st.subheader("🧠 Training Details")

        if training_details.get('model_type') == 'RL':
            t1, t2, t3, t4 = st.columns(4)
            t1.metric("Algorithm", training_details.get('algorithm', 'DQN'))
            t2.metric("Timesteps", f"{training_details.get('timesteps', 0):,}")
            t3.metric("Episodes", training_details.get('total_episodes', 0))
            t4.metric("Avg Reward", f"{training_details.get('avg_reward', 0):.4f}")

            st.markdown(f"""
            **RL Agent Configuration:**
            - **Algorithm:** DQN (Deep Q-Network)
            - **Policy:** MlpPolicy (Multi-Layer Perceptron)
            - **Total Timesteps:** {training_details.get('timesteps', 0):,}
            - **Episodes Completed:** {training_details.get('total_episodes', 0)}
            - **Average Episode Reward:** {training_details.get('avg_reward', 0):.4f}
            - **Best Episode Reward:** {training_details.get('best_reward', 0):.4f}
            - **Worst Episode Reward:** {training_details.get('worst_reward', 0):.4f}
            - **Avg Episode Length:** {training_details.get('avg_episode_length', 0):.0f} steps
            - **Reward Formula:** `PnL × RR_bonus - overtrading_penalty - holding_penalty - cost`
            - **Actions:** 0=HOLD, 1=BUY, 2=SELL
            - **State:** Returns, Volatility, RSI, Body Ratio, Wicks, SMA Ratios, Position
            - **Constraints:** Cooldown (3 days), Trend Filter (SMA_10 > SMA_50)
            """)

        else:  # ML
            t1, t2, t3, t4 = st.columns(4)
            t1.metric("Algorithm", "RandomForest")
            t2.metric("Estimators", training_details.get('n_estimators', 100))
            t3.metric("Max Depth", training_details.get('max_depth', 5))
            t4.metric("Win Rate (Approx)", f"{training_details.get('accuracy', 0):.2%}")

            st.markdown(f"""
            **ML Model Configuration:**
            - **Model:** RandomForestClassifier
            - **n_estimators:** {training_details.get('n_estimators', 100)}
            - **max_depth:** {training_details.get('max_depth', 5)}
            - **Confidence Threshold:** 60% (only trades above this)
            - **Features Used:**
              - Returns, 10-day Volatility, RSI (14-period)
              - Candle structure (Body Ratio, Upper Wick, Lower Wick)
              - SMA indicators (SMA_10 ratio, SMA_50 ratio)
            - **Dataset Size:** {len(df)} trading days
            - **Constraints:** Cooldown (3 days), Trend Filter (SMA_10 > SMA_50)
            - **Validation:** Walk-Forward Analysis (Train: {train_window}d, Test: {test_window}d)
            """)

        st.markdown("---")

        # --- SECTION 4: VISUALIZATIONS ---
        show_charts = st.checkbox("📉 Show Visualizations", value=True)

        if show_charts:
            st.subheader("📈 Charts")

            tab1, tab2, tab3 = st.tabs(["Price Chart with Signals", "Equity Curve", "Drawdown Chart"])

            with tab1:
                fig_price = go.Figure()
                fig_price.add_trace(go.Scatter(
                    x=df.index, y=df['Close'], mode='lines',
                    name='Close Price', line=dict(color='#1f77b4', width=1)
                ))

                if not trades_df.empty:
                    entries = trades_df[['Entry_Date', 'Entry_Price']].copy()
                    fig_price.add_trace(go.Scatter(
                        x=entries['Entry_Date'], y=entries['Entry_Price'],
                        mode='markers', name='BUY',
                        marker=dict(color='#00cc96', size=10, symbol='triangle-up')
                    ))

                    exits = trades_df[['Exit_Date', 'Exit_Price']].copy()
                    fig_price.add_trace(go.Scatter(
                        x=exits['Exit_Date'], y=exits['Exit_Price'],
                        mode='markers', name='SELL',
                        marker=dict(color='#ef553b', size=10, symbol='triangle-down')
                    ))

                fig_price.update_layout(
                    title=f"Stock Price with {model_type} Trade Signals",
                    xaxis_title="Date", yaxis_title="Price ($)",
                    template="plotly_white"
                )
                st.plotly_chart(fig_price, use_container_width=True)

            with tab2:
                fig_equity = go.Figure()
                fig_equity.add_trace(go.Scatter(
                    x=daily_equity.index, y=daily_equity.values,
                    mode='lines', name=f'{model_type} Strategy',
                    line=dict(color='#636efa', width=2)
                ))
                fig_equity.add_trace(go.Scatter(
                    x=baseline_equity.index, y=baseline_equity.values,
                    mode='lines', name='Buy & Hold',
                    line=dict(color='#ffa15a', dash='dash', width=2)
                ))
                fig_equity.update_layout(
                    title=f"Capital Growth: {model_type} Strategy vs Buy & Hold (Initial: $100,000)",
                    xaxis_title="Date", yaxis_title="Portfolio Value ($)",
                    template="plotly_white"
                )
                st.plotly_chart(fig_equity, use_container_width=True)

            with tab3:
                running_max = daily_equity.cummax()
                drawdown = (daily_equity - running_max) / running_max

                fig_dd = go.Figure()
                fig_dd.add_trace(go.Scatter(
                    x=drawdown.index, y=drawdown.values,
                    fill='tozeroy', name='Drawdown',
                    line=dict(color='#ef553b')
                ))
                fig_dd.update_layout(
                    title=f"{model_type} Strategy Drawdown Over Time",
                    xaxis_title="Date", yaxis_title="Drawdown (%)",
                    template="plotly_white"
                )
                fig_dd.layout.yaxis.tickformat = ',.1%'
                st.plotly_chart(fig_dd, use_container_width=True)

        st.markdown("---")

        # --- SECTION 5: TRADE HISTORY TABLE ---
        st.subheader("📋 Trade History")
        if not trades_df.empty:
            display_cols = ['Entry_Date', 'Exit_Date', 'Entry_Price', 'Exit_Price',
                          'Net_PnL', 'Duration', 'Exit_Reason', 'Reward']
            available_cols = [c for c in display_cols if c in trades_df.columns]
            hist_df = trades_df[available_cols].copy()

            hist_df['Entry_Date'] = pd.to_datetime(hist_df['Entry_Date']).dt.date
            hist_df['Exit_Date'] = pd.to_datetime(hist_df['Exit_Date']).dt.date
            hist_df['Entry_Price'] = hist_df['Entry_Price'].round(2)
            hist_df['Exit_Price'] = hist_df['Exit_Price'].round(2)
            hist_df['Net_PnL'] = hist_df['Net_PnL'].apply(lambda x: f"{x:.2%}")
            hist_df['Reward'] = hist_df['Reward'].apply(lambda x: f"{x:.4f}")

            st.dataframe(hist_df, use_container_width=True)

            csv = hist_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Trade History (CSV)",
                data=csv,
                file_name=f'{ticker}_{model_key}_trade_history.csv',
                mime='text/csv',
            )
        else:
            st.write("No trades executed.")

        st.markdown("---")

        # --- SECTION 6: MONTE CARLO RESULTS ---
        if metrics.get('MC_Final_Capital_Mean'):
            st.subheader("🎲 Monte Carlo Robustness Test")
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Avg Final Capital", f"${metrics.get('MC_Final_Capital_Mean', 0):,.0f}")
            mc2.metric("Capital Std Dev", f"${metrics.get('MC_Final_Capital_Std', 0):,.0f}")
            mc3.metric("Worst Drawdown", f"{metrics.get('MC_Max_Drawdown_Worst', 0):.2%}")
            mc4.metric("Win Probability", f"{metrics.get('MC_Win_Probability', 0):.2%}")


# --- MULTI-STOCK TESTING ---
if run_multi:
    multi_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    
    with st.status(f"Running {model_type} on {len(multi_tickers)} tickers...", expanded=True) as status:
        for t in multi_tickers:
            st.write(f"📊 Processing {t}...")
        
        all_results, avg_metrics = run_multi_stock_test(
            tickers=multi_tickers,
            period=period,
            train_window=train_window,
            test_window=test_window,
            model_type=model_key,
            rl_timesteps=rl_timesteps
        )
        
        status.update(label="✅ Multi-Stock Test Complete!", state="complete", expanded=False)
    
    st.subheader(f"🌍 Multi-Stock Test Results ({model_type})")
    
    # Average metrics
    if avg_metrics:
        st.markdown("#### 📊 Average Performance Across All Tickers")
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Avg Total Trades", f"{avg_metrics.get('Total_Trades', 0):.0f}")
        a2.metric("Avg Win Rate", f"{avg_metrics.get('Win_Rate', 0):.2%}")
        a3.metric("Avg Return", f"{avg_metrics.get('Total_Return', 0):.2%}")
        a4.metric("Avg Sharpe", f"{avg_metrics.get('Sharpe_Ratio', 0):.2f}")
    
    # Per-ticker breakdown
    st.markdown("#### 📋 Per-Ticker Breakdown")
    
    table_data = []
    for r in all_results:
        m = r.get('metrics', {})
        bm = r.get('baseline_metrics', {})
        table_data.append({
            'Ticker': r['ticker'],
            'Trades': r.get('num_trades', 0),
            'Strategy Return': f"{m.get('Total_Return', 0):.2%}" if m else "N/A",
            'B&H Return': f"{bm.get('Total_Return', 0):.2%}" if bm else "N/A",
            'Win Rate': f"{m.get('Win_Rate', 0):.2%}" if m else "N/A",
            'Sharpe': f"{m.get('Sharpe_Ratio', 0):.2f}" if m else "N/A",
            'Max DD': f"{m.get('Max_Drawdown', 0):.2%}" if m else "N/A",
            'Status': '✅' if r.get('num_trades', 0) > 0 else '❌'
        })
    
    st.dataframe(pd.DataFrame(table_data), use_container_width=True)
