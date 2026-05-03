import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from main import run_pipeline, get_live_prediction

st.set_page_config(page_title="AI Trading Agent", layout="wide")

st.title("🤖 AI Trading Agent with Advanced Risk Logic")

# Sidebar for inputs
st.sidebar.header("Simulation Parameters")
ticker = st.sidebar.text_input("Ticker Symbol", value="AAPL")
period = st.sidebar.selectbox("Data Period", options=["1y", "2y", "5y", "10y"], index=2)

train_window = st.sidebar.slider("Walk-Forward Train Window (days)", 100, 1000, 365)
test_window = st.sidebar.slider("Walk-Forward Test Window (days)", 30, 365, 90)

# --- Live Prediction Section ---
st.markdown("### 🔮 Live Prediction for Next Trading Day")
with st.spinner(f"Generating live prediction for {ticker}..."):
    live_pred = get_live_prediction(ticker)

if live_pred:
    pred_color = "🟢" if live_pred['prediction'] == "BUY" else "🔴"
    st.info(f"**{ticker} ({live_pred['date']}) Last Close:** ₹{live_pred['latest_close']:.2f}  |  **Prediction:** {pred_color} {live_pred['prediction']}  |  **Confidence:** {live_pred['confidence']:.2%}")
else:
    st.warning("Could not generate live prediction.")

st.markdown("---")

if st.sidebar.button("Run Historical Simulation"):
    with st.spinner(f"Running complete pipeline for {ticker}..."):
        results = run_pipeline(ticker, period, train_window, test_window)
        
    if results is None:
        st.error("Simulation failed. Check logs or try a different ticker.")
    else:
        st.success("Simulation Complete!")
        
        metrics = results['metrics']
        baseline = results['baseline_metrics']
        trades_df = results['trades']
        df = results['df']
        daily_equity = results['daily_equity']
        baseline_equity = results['baseline_equity']
        
        # --- Metrics Display ---
        st.subheader("📊 Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Return (Agent)", f"{metrics['Total_Return']:.2%}")
        col2.metric("Total Return (Baseline)", f"{baseline['Total_Return']:.2%}")
        col3.metric("Win Rate", f"{metrics['Win_Rate']:.2%}")
        col4.metric("Total Trades", metrics['Total_Trades'])
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Agent Sharpe Ratio", f"{metrics['Sharpe_Ratio']:.2f}")
        col2.metric("Baseline Sharpe Ratio", f"{baseline['Sharpe_Ratio']:.2f}")
        col3.metric("Max Drawdown", f"{metrics['Max_Drawdown']:.2%}")
        col4.metric("Profit Factor", f"{metrics['Profit_Factor']:.2f}")
        
        # --- Equity Curve ---
        st.subheader("📈 Equity Curve Comparison")
        fig_equity = go.Figure()
        
        # We need to align the equity curves to the same index to plot cleanly
        # To avoid index mismatch errors, let's just plot them directly
        fig_equity.add_trace(go.Scatter(x=daily_equity.index, y=daily_equity.values, 
                                        mode='lines', name='Agent Equity', line=dict(color='blue')))
        fig_equity.add_trace(go.Scatter(x=baseline_equity.index, y=baseline_equity.values, 
                                        mode='lines', name='Buy & Hold Baseline', line=dict(color='gray', dash='dash')))
        
        fig_equity.update_layout(title="Agent vs Baseline Equity Growth", xaxis_title="Date", yaxis_title="Portfolio Value (₹)")
        st.plotly_chart(fig_equity, use_container_width=True)
        
        # --- Stock Price & Trade Markers ---
        st.subheader(f"📉 {ticker} Price with Trade Executions")
        fig_price = go.Figure()
        
        # Plot price
        fig_price.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price', line=dict(color='black', width=1)))
        
        # Plot entries
        entries = trades_df[['Entry_Date', 'Entry_Price']].copy()
        fig_price.add_trace(go.Scatter(x=entries['Entry_Date'], y=entries['Entry_Price'], 
                                       mode='markers', name='BUY', marker=dict(color='green', size=10, symbol='triangle-up')))
        
        # Plot exits
        exits = trades_df[['Exit_Date', 'Exit_Price', 'Net_PnL']].copy()
        # Separate profitable and losing exits for better visualization
        prof_exits = exits[exits['Net_PnL'] > 0]
        loss_exits = exits[exits['Net_PnL'] <= 0]
        
        fig_price.add_trace(go.Scatter(x=prof_exits['Exit_Date'], y=prof_exits['Exit_Price'], 
                                       mode='markers', name='SELL (Profit)', marker=dict(color='blue', size=8, symbol='triangle-down')))
        fig_price.add_trace(go.Scatter(x=loss_exits['Exit_Date'], y=loss_exits['Exit_Price'], 
                                       mode='markers', name='SELL (Loss)', marker=dict(color='red', size=8, symbol='triangle-down')))
                                       
        fig_price.update_layout(title="Trade Entry and Exits", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig_price, use_container_width=True)
        
        # --- Monte Carlo & Drawdown ---
        st.subheader("🛡️ Risk & Robustness Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Monte Carlo Simulation Results (100 runs)**")
            st.write(f"- **Mean Final Capital:** ₹{metrics['MC_Final_Capital_Mean']:,.2f}")
            st.write(f"- **Win Probability:** {metrics['MC_Win_Probability']:.2%}")
            st.write(f"- **Worst Case Drawdown:** {metrics['MC_Max_Drawdown_Worst']:.2%}")
            
        with col2:
            st.write("**Recent Trades Log**")
            display_trades = trades_df[['Entry_Date', 'Exit_Date', 'Duration', 'Net_PnL', 'Exit_Reason', 'Reward']].tail(10).copy()
            # Format PnL
            display_trades['Net_PnL'] = display_trades['Net_PnL'].apply(lambda x: f"{x:.2%}")
            st.dataframe(display_trades)
            
st.markdown("---")
st.markdown("Built by an AI Agent.")
