import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
import plotly.graph_objects as go
import time

st.set_page_config(page_title="Stock Analysis Toolkit", layout="wide")
st.title("Stock Analysis Toolkit")

tab1, tab2, tab3 = st.tabs(["Historical Backtest", "Current Analysis", "Simulation Results"])

def calculate_performance_metrics(dates, values, risk_free_rate=0.0):
    """Calculate comprehensive performance metrics"""
    if len(values) <= 1:
        return {}

    returns = pd.Series(values).pct_change().dropna()
    if returns.empty:
        return {}

    values = [float(v) for v in values]

    total_return = (values[-1] - values[0]) / values[0]

    years = (dates[-1] - dates[0]).days / 365.25
    cagr = (values[-1] / values[0]) ** (1 / years) - 1 if years > 0 else total_return

    std_dev = float(returns.std() * np.sqrt(252))

    cumulative = pd.Series(values)
    rolling_max = cumulative.expanding().max()
    drawdowns = (cumulative - rolling_max) / rolling_max
    max_drawdown = float(drawdowns.min())

    excess_returns = returns - risk_free_rate / 252
    sharpe = float(excess_returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0.0

    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
    sortino = float((returns.mean() * 252 - risk_free_rate) / downside_std) if downside_std > 0 else 0.0

    yearly_returns = []
    current_year = dates[0].year
    year_values = [values[0]]

    for i in range(1, len(dates)):
        if dates[i].year != current_year:
            if len(year_values) > 1:
                year_return = (year_values[-1] - year_values[0]) / year_values[0]
                yearly_returns.append(float(year_return))
            current_year = dates[i].year
            year_values = [values[i]]
        else:
            year_values.append(values[i])

    if len(year_values) > 1:
        year_return = (year_values[-1] - year_values[0]) / year_values[0]
        yearly_returns.append(float(year_return))

    best_year = max(yearly_returns) if yearly_returns else 0.0
    worst_year = min(yearly_returns) if yearly_returns else 0.0

    return {
        'total_return': float(total_return),
        'cagr': float(cagr),
        'std_dev': float(std_dev),
        'max_drawdown': float(max_drawdown),
        'sharpe': float(sharpe),
        'sortino': float(sortino),
        'best_year': float(best_year),
        'worst_year': float(worst_year)
    }

def safe_format(value, format_str=".2f", suffix="%"):
    """Safely format values that might be Series or NaN"""
    if value is None or pd.isna(value):
        return "N/A"

    try:

        if hasattr(value, 'iloc'):
            value = value.iloc[0] if len(value) > 0 else 0.0
        elif hasattr(value, 'item'):
            value = value.item()

        formatted = f"{float(value):{format_str}}"
        return f"{formatted}{suffix}" if suffix else formatted
    except (ValueError, TypeError, IndexError):
        return "N/A"

def run_backtest_simulation(stock_data_full, irx_data_full, start_date, end_date, 
                           sma_months, timing, initial_investment):
    """Run a single backtest simulation with given parameters"""

    stock_data_full = stock_data_full.copy()
    stock_data_full.index = pd.to_datetime(stock_data_full.index)
    irx_data_full.index = pd.to_datetime(irx_data_full.index)

    window_size = sma_months * 21
    stock_data_full[f"{sma_months}M_SMA"] = stock_data_full["Close"].rolling(
        window=window_size, min_periods=1
    ).mean()

    stock_data = stock_data_full.loc[stock_data_full.index >= pd.to_datetime(start_date)].copy()
    irx_data = irx_data_full.reindex(stock_data.index, method="ffill")

    if stock_data[f"{sma_months}M_SMA"].isna().any():
        first_valid_sma = stock_data[f"{sma_months}M_SMA"].first_valid_index()
        if first_valid_sma is not None:
            first_valid_value = stock_data.loc[first_valid_sma, f"{sma_months}M_SMA"]
            stock_data[f"{sma_months}M_SMA"].fillna(first_valid_value, inplace=True)
        else:
            stock_data[f"{sma_months}M_SMA"] = stock_data["Close"]

    initial_cash = initial_investment
    cash = initial_cash
    primary_shares = 0
    portfolio_values = []
    bh_values = []

    dates = stock_data.index
    buy_and_hold_shares = initial_cash / stock_data["Close"].iloc[0]

    avg_irx = float(irx_data["Close"].mean() / 100) if not irx_data.empty else 0.0

    for i in range(len(stock_data)):
        date_i = dates[i]
        price = float(stock_data["Close"].iloc[i])
        sma = float(stock_data[f"{sma_months}M_SMA"].iloc[i])

        irx_yield = 0.0

        try:
            irx_value = irx_data.loc[date_i]
            if isinstance(irx_value, pd.DataFrame):
                irx_close = irx_value["Close"].iloc[0] if not irx_value.empty else 0.0
            else:
                irx_close = irx_value["Close"]

            if hasattr(irx_close, 'item'):
                irx_close = irx_close.item()
            elif hasattr(irx_close, 'iloc'):
                irx_close = irx_close.iloc[0] if len(irx_close) > 0 else 0.0

            if not pd.isna(irx_close) and irx_close > 0:
                irx_yield = float(irx_close) / 100
        except (KeyError, IndexError, ValueError):
            irx_yield = 0.0

        irx_daily_return = (1 + irx_yield) ** (1 / 252) - 1

        if cash > 0:
            cash *= (1 + irx_daily_return)

        should_trade = False

        if timing == "At Signal":
            should_trade = True
        elif timing == "Weekly" and date_i.weekday() == 4:
            should_trade = True
        elif timing == "Biweekly":
            if date_i.weekday() == 4:
                week_num = date_i.isocalendar()[1]
                should_trade = (week_num % 2 == 0)
        elif timing == "Monthly":
            if i == len(stock_data) - 1:
                should_trade = True
            else:
                next_date = dates[i + 1]
                should_trade = (date_i.month != next_date.month)
        elif timing == "Bimonthly":
            if i == len(stock_data) - 1:
                should_trade = (date_i.month % 2 == 0)
            else:
                next_date = dates[i + 1]
                should_trade = (date_i.month != next_date.month and date_i.month % 2 == 0)
        elif timing == "Quarterly":
            if i == len(stock_data) - 1:
                should_trade = (date_i.month in [3, 6, 9, 12])
            else:
                next_date = dates[i + 1]
                should_trade = (date_i.month != next_date.month and date_i.month in [3, 6, 9, 12])

        if should_trade:
            if price >= sma and cash > 0:
                primary_shares = cash / price
                cash = 0
            elif price < sma and primary_shares > 0:
                cash = primary_shares * price
                primary_shares = 0

        portfolio_value = cash + (primary_shares * price)
        portfolio_values.append(float(portfolio_value))

        bh_value = float(buy_and_hold_shares * price)
        bh_values.append(float(bh_value))

    final_portfolio = float(portfolio_values[-1]) if portfolio_values else initial_cash
    final_bh = float(bh_values[-1]) if bh_values else initial_cash

    sma_metrics = calculate_performance_metrics(dates, portfolio_values, avg_irx)

    return {
        'sma_months': sma_months,
        'timing': timing,
        'final_value': final_portfolio,
        'total_return': (final_portfolio - initial_investment) / initial_investment,
        'cagr': sma_metrics.get('cagr', 0),
        'sharpe': sma_metrics.get('sharpe', 0),
        'max_drawdown': sma_metrics.get('max_drawdown', 0)
    }

if 'simulation_results' not in st.session_state:
    st.session_state.simulation_results = None

with tab1:

    st.header("SMA Strategy vs Buy & Hold Backtest")

    with st.sidebar:
        ticker = st.text_input("Stock Ticker", "VTI", key="hist_ticker").upper()
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", date(2015, 12, 31), key="hist_start")
        with col2:
            end_date = st.date_input("End Date", date.today(), key="hist_end")
        sma_months = st.number_input("Analysis Period (months)", min_value=1, max_value=240, value=11, step=1, key="hist_months")
        timing = st.selectbox("Rebalance Frequency", ["At Signal", "Weekly", "Biweekly", "Monthly", "Bimonthly", "Quarterly"], key="hist_timing")
        initial_investment = st.number_input("Initial Capital ($)", min_value=1000, max_value=1000000, value=10000, step=1000, key="hist_investment")

        st.markdown("---")
        simulate_button = st.button("Run Parameter Simulation", use_container_width=True)

    sma_lookback = relativedelta(months=sma_months)
    fetch_start_date = start_date - sma_lookback

    with st.spinner("Fetching historical data..."):
        stock_data_full = yf.download(ticker, start=fetch_start_date, end=end_date)
        irx_data_full = yf.download("^IRX", start=fetch_start_date, end=end_date)

    if stock_data_full.empty or irx_data_full.empty:
        st.error("No data found for stock or ^IRX.")
    else:

        stock_data_full.index = pd.to_datetime(stock_data_full.index)
        irx_data_full.index = pd.to_datetime(irx_data_full.index)

        window_size = sma_months * 21
        stock_data_full[f"{sma_months}M_SMA"] = stock_data_full["Close"].rolling(
            window=window_size, min_periods=1
        ).mean()

        stock_data = stock_data_full.loc[stock_data_full.index >= pd.to_datetime(start_date)].copy()
        irx_data = irx_data_full.reindex(stock_data.index, method="ffill")

        if stock_data[f"{sma_months}M_SMA"].isna().any():
            first_valid_sma = stock_data[f"{sma_months}M_SMA"].first_valid_index()
            if first_valid_sma is not None:
                first_valid_value = stock_data.loc[first_valid_sma, f"{sma_months}M_SMA"]
                stock_data[f"{sma_months}M_SMA"].fillna(first_valid_value, inplace=True)
            else:
                stock_data[f"{sma_months}M_SMA"] = stock_data["Close"]

        initial_cash = initial_investment
        cash = initial_cash
        primary_shares = 0
        portfolio_values = []
        bh_values = []

        dates = stock_data.index
        buy_and_hold_shares = initial_cash / stock_data["Close"].iloc[0]

        avg_irx = float(irx_data["Close"].mean() / 100) if not irx_data.empty else 0.0

        for i in range(len(stock_data)):
            date_i = dates[i]
            price = float(stock_data["Close"].iloc[i])
            sma = float(stock_data[f"{sma_months}M_SMA"].iloc[i])

            irx_yield = 0.0

            try:
                irx_value = irx_data.loc[date_i]
                if isinstance(irx_value, pd.DataFrame):
                    irx_close = irx_value["Close"].iloc[0] if not irx_value.empty else 0.0
                else:
                    irx_close = irx_value["Close"]

                if hasattr(irx_close, 'item'):
                    irx_close = irx_close.item()
                elif hasattr(irx_close, 'iloc'):
                    irx_close = irx_close.iloc[0] if len(irx_close) > 0 else 0.0

                if not pd.isna(irx_close) and irx_close > 0:
                    irx_yield = float(irx_close) / 100
            except (KeyError, IndexError, ValueError):
                irx_yield = 0.0

            irx_daily_return = (1 + irx_yield) ** (1 / 252) - 1

            if cash > 0:
                cash *= (1 + irx_daily_return)

            should_trade = False

            if timing == "At Signal":
                should_trade = True
            elif timing == "Weekly" and date_i.weekday() == 4:
                should_trade = True
            elif timing == "Biweekly":
                if date_i.weekday() == 4:
                    week_num = date_i.isocalendar()[1]
                    should_trade = (week_num % 2 == 0)
            elif timing == "Monthly":
                if i == len(stock_data) - 1:
                    should_trade = True
                else:
                    next_date = dates[i + 1]
                    should_trade = (date_i.month != next_date.month)
            elif timing == "Bimonthly":
                if i == len(stock_data) - 1:
                    should_trade = (date_i.month % 2 == 0)
                else:
                    next_date = dates[i + 1]
                    should_trade = (date_i.month != next_date.month and date_i.month % 2 == 0)
            elif timing == "Quarterly":
                if i == len(stock_data) - 1:
                    should_trade = (date_i.month in [3, 6, 9, 12])
                else:
                    next_date = dates[i + 1]
                    should_trade = (date_i.month != next_date.month and date_i.month in [3, 6, 9, 12])

            if should_trade:
                if price >= sma and cash > 0:
                    primary_shares = cash / price
                    cash = 0
                elif price < sma and primary_shares > 0:
                    cash = primary_shares * price
                    primary_shares = 0

            portfolio_value = cash + (primary_shares * price)
            portfolio_values.append(float(portfolio_value))

            bh_value = float(buy_and_hold_shares * price)
            bh_values.append(float(bh_value))

        final_portfolio = float(portfolio_values[-1]) if portfolio_values else initial_cash
        final_bh = float(bh_values[-1]) if bh_values else initial_cash

        sma_metrics = calculate_performance_metrics(dates, portfolio_values, avg_irx)
        bh_metrics = calculate_performance_metrics(dates, bh_values, avg_irx)

        sma_return = (final_portfolio - initial_cash) / initial_cash * 100
        bh_return = (final_bh - initial_cash) / initial_cash * 100
        relative_perf = (final_portfolio / final_bh - 1) * 100

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                label="SMA Strategy Value",
                value=f"${final_portfolio:,.0f}",
                delta=f"{sma_return:+.1f}%"
            )

        with col2:
            st.metric(
                label="Buy & Hold Value",
                value=f"${final_bh:,.0f}",
                delta=f"{bh_return:+.1f}%"
            )

        with col3:
            st.metric(
                label="Relative Performance",
                value=f"{relative_perf:+.1f}%",
                delta=None
            )

        with col4:
            total_return = final_portfolio - initial_cash
            st.metric(
                label="Absolute Return",
                value=f"${total_return:,.0f}",
                delta=None
            )

        fig = go.Figure()

        monthly_dates = []
        monthly_portfolio_values = []
        monthly_bh_values = []

        for i in range(len(dates)):
            current_date = dates[i]

            if i == len(dates) - 1:

                monthly_dates.append(current_date)
                monthly_portfolio_values.append(portfolio_values[i])
                monthly_bh_values.append(bh_values[i])
            else:
                next_date = dates[i + 1]
                if current_date.month != next_date.month:
                    monthly_dates.append(current_date)
                    monthly_portfolio_values.append(portfolio_values[i])
                    monthly_bh_values.append(bh_values[i])

        fig.add_trace(go.Scatter(x=monthly_dates, y=monthly_portfolio_values, name="SMA Strategy", line=dict(width=2.5)))
        fig.add_trace(go.Scatter(x=monthly_dates, y=monthly_bh_values, name="Buy & Hold", line=dict(width=2.5)))

        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            template="plotly_dark",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=20, t=60, b=40),
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Parameter Simulation Results")

    if simulate_button:

            sma_months_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 18, 21, 24, 27, 30, 33, 36]
            rebalance_freqs = ["At Signal", "Weekly", "Biweekly", "Monthly", "Bimonthly", "Quarterly"]

            results = []
            total_combinations = len(sma_months_range) * len(rebalance_freqs)
            progress_bar = st.progress(0)
            status_text = st.empty()

            current_combination = 0

            for sma_test in sma_months_range:
                for timing_test in rebalance_freqs:
                    current_combination += 1
                    progress = current_combination / total_combinations
                    progress_bar.progress(progress)
                    status_text.text(f"Testing SMA {sma_test} months with {timing_test} rebalancing... ({current_combination}/{total_combinations})")

                    try:
                        result = run_backtest_simulation(
                            stock_data_full, irx_data_full, start_date, end_date,
                            sma_test, timing_test, initial_investment
                        )
                        results.append(result)
                    except Exception as e:
                        st.warning(f"Failed for SMA {sma_test} months, {timing_test}: {str(e)}")

            st.session_state.simulation_results = {
                'results': results,
                'ticker': ticker,
                'start_date': start_date,
                'end_date': end_date,
                'initial_investment': initial_investment
            }

            progress_bar.empty()
            status_text.empty()

    if st.session_state.simulation_results is None:
        st.info("Run a parameter simulation to see results here!")
    else:
        results = st.session_state.simulation_results['results']
        ticker = st.session_state.simulation_results['ticker']
        start_date = st.session_state.simulation_results['start_date']
        end_date = st.session_state.simulation_results['end_date']
        initial_investment = st.session_state.simulation_results['initial_investment']

        if results:
            results_df = pd.DataFrame(results)

            best_by_return = results_df.loc[results_df['total_return'].idxmax()]
            best_by_cagr = results_df.loc[results_df['cagr'].idxmax()]
            best_by_sharpe = results_df.loc[results_df['sharpe'].idxmax()]
            best_by_drawdown = results_df.loc[results_df['max_drawdown'].idxmin()]

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Best Total Return",
                    f"SMA {int(best_by_return['sma_months'])}M\n{best_by_return['timing']}",
                    f"{best_by_return['total_return']*100:.1f}%"
                )

            with col2:
                st.metric(
                    "Best CAGR",
                    f"SMA {int(best_by_cagr['sma_months'])}M\n{best_by_cagr['timing']}",
                    f"{best_by_cagr['cagr']*100:.1f}%"
                )

            with col3:
                st.metric(
                    "Best Sharpe Ratio",
                    f"SMA {int(best_by_sharpe['sma_months'])}M\n{best_by_sharpe['timing']}",
                    f"{best_by_sharpe['sharpe']:.2f}"
                )

            top_results = results_df.nlargest(10, 'total_return')[['sma_months', 'timing', 'final_value', 'total_return', 'cagr', 'sharpe', 'max_drawdown']]
            top_results['total_return'] = (top_results['total_return'] * 100).round(1)
            top_results['cagr'] = (top_results['cagr'] * 100).round(1)
            top_results['max_drawdown'] = (top_results['max_drawdown'] * 100).round(1)
            top_results['sharpe'] = top_results['sharpe'].round(2)
            top_results['final_value'] = top_results['final_value'].round(0)

            top_results.columns = ['SMA Months', 'Rebalance Freq', 'Final Value', 'Total Return %', 'CAGR %', 'Sharpe Ratio', 'Max Drawdown %']
            st.dataframe(top_results, use_container_width=True)

with tab2:

    st.header("Current Market Analysis")

    @st.cache_data(show_spinner=True)
    def fetch_stock(ticker_symbol: str, start_date, end_date):
        return yf.download(ticker_symbol.upper(), start=start_date, end=end_date)

    start_date_current = end_date - relativedelta(months=72)
    try:
        stock_data = fetch_stock(ticker, start_date_current, end_date)
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        stock_data = pd.DataFrame()

    if stock_data.empty:
        st.warning(f"No data found for {ticker} in the selected date range.")
    else:
        price_column = "Close"
        price_series = stock_data[price_column].squeeze()

        last_price = float(price_series.iloc[-1])
        avg_price = float(price_series.mean())

        sma_days = sma_months * 21
        df = pd.DataFrame({price_column: price_series})
        df[f"SMA {sma_days}"] = df[price_column].rolling(window=sma_days, min_periods=1).mean()
        df["Signal"] = (df[price_column] >= df[f"SMA {sma_days}"]).astype(int)

        latest_signal = "BUY" if df["Signal"].iloc[-1] == 1 else "SELL"
        signal_color = "normal" if latest_signal == "BUY" else "inverse"

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Price", f"${last_price:,.2f}")
        with col2:
            st.metric("Period Average", f"${df[f"SMA {sma_days}"].iloc[-1]:,.2f}")
        with col3:
            st.metric("Trading Signal", latest_signal, delta=None, delta_color=signal_color)

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[price_column],
            mode='lines',
            name='Price',
            line=dict(color='#1f77b4', width=2.5)
        ))

        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[f"SMA {sma_days}"],
            mode='lines',
            name=f'SMA {sma_days}',
            line=dict(width=2, dash='dash', color='#ff7f0e')
        ))

        buy_signals = df[df["Signal"].diff() == 1]
        sell_signals = df[df["Signal"].diff() == -1]

        if not buy_signals.empty:
            fig.add_trace(go.Scatter(
                x=buy_signals.index,
                y=buy_signals[price_column],
                mode='markers',
                name='Buy Signal',
                marker=dict(color='#2ecc71', size=8, symbol='triangle-up', line=dict(width=1, color='DarkSlateGrey'))
            ))

        if not sell_signals.empty:
            fig.add_trace(go.Scatter(
                x=sell_signals.index,
                y=sell_signals[price_column],
                mode='markers',
                name='Sell Signal',
                marker=dict(color='#e74c3c', size=8, symbol='triangle-down', line=dict(width=1, color='DarkSlateGrey'))
            ))

        fig.update_layout(
            template="plotly_dark",
            hovermode="x unified",
            xaxis=dict(
                rangeselector=dict(
                    buttons=[
                        dict(count=1, label="1M", step="month", stepmode="backward"),
                        dict(count=3, label="3M", step="month", stepmode="backward"),
                        dict(count=6, label="6M", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(step="all")
                    ]
                ),
                rangeslider=dict(visible=False),
                type="date"
            ),
            yaxis=dict(title="Price ($)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=20, t=60, b=40),
            height=500,
        )

        st.plotly_chart(fig, use_container_width=True)

        with st.expander("View Historical Data"):
            st.dataframe(
                df[[price_column, f"SMA {sma_days}", "Signal"]].tail(20),
                use_container_width=True
            )