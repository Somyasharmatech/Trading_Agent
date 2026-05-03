import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from main import run_pipeline, get_live_prediction

st.set_page_config(page_title="AI Trading Agent", layout="wide")

st.title("🤖 AI Trading Agent with Advanced Risk Logic")
st.markdown("AI-driven trading system using machine learning, risk-aware logic, and backtesting.")

# --- SIDEBAR (LEFT PANEL) ---
st.sidebar.header("Simulation Parameters")
ticker = st.sidebar.text_input("Ticker Symbol", value="AAPL")
period = st.sidebar.selectbox("Data Period", options=["1y", "2y", "5y", "10y"], index=2)

train_window = st.sidebar.slider("Walk-Forward Train Window (days)", 100, 1000, 365)
test_window = st.sidebar.slider("Walk-Forward Test Window (days)", 30, 365, 90)

# --- LIVE PREDICTION ---
st.markdown("### 🔮 Live Prediction for Next Trading Day")
with st.spinner(f"Generating live prediction for {ticker}..."):
    live_pred = get_live_prediction(ticker)

if live_pred:
    pred_color = "🟢" if live_pred['prediction'] == "BUY" else "🔴"
    st.info(f"**{ticker} ({live_pred['date']}) Last Close:** ₹{live_pred['latest_close']:.2f}  |  **Prediction:** {pred_color} {live_pred['prediction']}  |  **Confidence:** {live_pred['confidence']:.2%}")
else:
    st.error("Invalid ticker or data not available. Please try another symbol.")

st.markdown("---")

# --- MAIN EXECUTION ---
if st.sidebar.button("Run Historical Simulation"):
    
    # 2. Add Loading State
    with st.status("Running simulation...", expanded=True) as status:
        st.write("Fetching data...")
        st.write("Training model...")
        st.write("Backtesting strategy...")
        st.write("Calculating metrics...")
        results = run_pipeline(ticker, period, train_window, test_window)
        status.update(label="Simulation Complete!", state="complete", expanded=False)
        
    if results is None:
        st.error("Invalid ticker or data not available. Please try another symbol.")
    else:
        metrics = results['metrics']
        baseline = results['baseline_metrics']
        trades_df = results['trades']
        df = results['df']
        daily_equity = results['daily_equity']
        baseline_equity = results['baseline_equity']
        
        # --- SECTION 1: PERFORMANCE METRICS ---
        st.subheader("📊 Performance Summary")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Trades", metrics.get('Total_Trades', 0))
        col2.metric("Win Rate (%)", f"{metrics.get('Win_Rate', 0):.2%}")
        col3.metric("Average Profit/Loss", f"{metrics.get('Avg_PnL', 0):.2%}")
        col4.metric("Sharpe Ratio", f"{metrics.get('Sharpe_Ratio', 0):.2f}")
        
        col5, col6, col7, col8 = st.columns(4)
        col5.metric("Sortino Ratio", f"{metrics.get('Sortino_Ratio', 0):.2f}")
        col6.metric("Maximum Drawdown", f"{metrics.get('Max_Drawdown', 0):.2%}")
        col7.metric("Calmar Ratio", f"{metrics.get('Calmar_Ratio', 0):.2f}")
        col8.metric("Profit Factor", f"{metrics.get('Profit_Factor', 0):.2f}")
        
        st.markdown("---")
        
        # --- SECTION 4: STRATEGY COMPARISON ---
        st.subheader("📈 Strategy vs Buy & Hold")
        strat_return = metrics.get('Total_Return', 0)
        bh_return = baseline.get('Total_Return', 0)
        diff = strat_return - bh_return
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Strategy Return (%)", f"{strat_return:.2%}")
        c2.metric("Buy & Hold Return (%)", f"{bh_return:.2%}")
        c3.metric("Difference", f"{diff:.2%}", delta=f"{diff:.2%}")
        
        st.markdown("---")
        
        # BONUS: Toggle for charts
        show_charts = st.checkbox("Show Visualizations", value=True)
        
        if show_charts:
            # --- SECTION 2: CHARTS ---
            st.subheader("📉 Visualizations")
            
            tab1, tab2, tab3 = st.tabs(["Price Chart with Signals", "Equity Curve", "Drawdown Chart"])
            
            with tab1:
                fig_price = go.Figure()
                fig_price.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price', line=dict(color='black', width=1)))
                
                if not trades_df.empty:
                    entries = trades_df[['Entry_Date', 'Entry_Price']].copy()
                    fig_price.add_trace(go.Scatter(x=entries['Entry_Date'], y=entries['Entry_Price'], 
                                                   mode='markers', name='BUY', marker=dict(color='green', size=10, symbol='triangle-up')))
                    
                    exits = trades_df[['Exit_Date', 'Exit_Price']].copy()
                    fig_price.add_trace(go.Scatter(x=exits['Exit_Date'], y=exits['Exit_Price'], 
                                                   mode='markers', name='SELL', marker=dict(color='red', size=10, symbol='triangle-down')))
                
                fig_price.update_layout(title="Stock Price with Trade Signals", xaxis_title="Date", yaxis_title="Price")
                st.plotly_chart(fig_price, use_container_width=True)
                
            with tab2:
                fig_equity = go.Figure()
                fig_equity.add_trace(go.Scatter(x=daily_equity.index, y=daily_equity.values, 
                                                mode='lines', name='Strategy Equity', line=dict(color='blue')))
                fig_equity.add_trace(go.Scatter(x=baseline_equity.index, y=baseline_equity.values, 
                                                mode='lines', name='Buy & Hold', line=dict(color='gray', dash='dash')))
                fig_equity.update_layout(title="Capital Growth over Time (Initial: ₹100,000)", xaxis_title="Date", yaxis_title="Portfolio Value (₹)")
                st.plotly_chart(fig_equity, use_container_width=True)
                
            with tab3:
                # Calculate drawdown
                running_max = daily_equity.cummax()
                drawdown = (daily_equity - running_max) / running_max
                
                fig_dd = go.Figure()
                fig_dd.add_trace(go.Scatter(x=drawdown.index, y=drawdown.values, fill='tozeroy', name='Drawdown', line=dict(color='red')))
                fig_dd.update_layout(title="Strategy Drawdown % Over Time", xaxis_title="Date", yaxis_title="Drawdown (%)")
                # Format y-axis as percentage
                fig_dd.layout.yaxis.tickformat = ',.1%'
                st.plotly_chart(fig_dd, use_container_width=True)

        st.markdown("---")
        
        # --- SECTION 3: TRADE HISTORY TABLE ---
        st.subheader("📋 Trade History")
        if not trades_df.empty:
            display_cols = ['Entry_Date', 'Exit_Date', 'Entry_Price', 'Exit_Price', 'Net_PnL', 'Duration']
            hist_df = trades_df[display_cols].copy()
            hist_df['Entry_Date'] = pd.to_datetime(hist_df['Entry_Date']).dt.date
            hist_df['Exit_Date'] = pd.to_datetime(hist_df['Exit_Date']).dt.date
            hist_df['Entry_Price'] = hist_df['Entry_Price'].round(2)
            hist_df['Exit_Price'] = hist_df['Exit_Price'].round(2)
            hist_df['Net_PnL'] = hist_df['Net_PnL'].apply(lambda x: f"{x:.2%}")
            
            st.dataframe(hist_df, use_container_width=True)
            
            # BONUS: Downloadable Results (CSV)
            csv = hist_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Trade History (CSV)",
                data=csv,
                file_name=f'{ticker}_trade_history.csv',
                mime='text/csv',
            )
        else:
            st.write("No trades executed.")

        st.markdown("---")
        
        # --- SECTION 5: MODEL INFO ---
        st.subheader("🧠 Model Information")
        st.markdown(f"""
        - **Model:** RandomForestClassifier (n_estimators=100, max_depth=5)
        - **Features Used:** 
          - Returns
          - 10-day Volatility
          - Candle structure (Body Ratio, Upper Wick Ratio, Lower Wick Ratio)
          - SMA indicators (SMA_10 ratio, SMA_50 ratio)
        - **Dataset Size:** {len(df)} trading days
        - **Validation Method:** Walk-Forward Analysis (Train: {train_window}d, Test: {test_window}d)
        """)
