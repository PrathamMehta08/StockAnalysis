import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
import plotly.graph_objects as go
import empyrical as ep

st.set_page_config(page_title="Stock Analysis Toolkit", layout="wide")
st.title("Stock Analysis Toolkit")

tab2, tab1, tab3 = st.tabs(["Current Analysis", "Historical Backtest", "Simulation Results"])

def calculate_performance_metrics(dates, values, irx_data=None, risk_free_rate=0.02):
    if len(values) <= 1:
        return {}

    df = pd.DataFrame({"date": pd.to_datetime(dates), "value": values})
    df.set_index("date", inplace=True)
    df = df.sort_index()

    monthly_values = df["value"].resample('ME').last()  
    monthly_returns = monthly_values.pct_change().dropna()

    if monthly_returns.empty:
        return {}

    total_return = (values[-1] - values[0]) / values[0]
    years = (df.index[-1] - df.index[0]).days / 365.25
    cagr = (values[-1] / values[0]) ** (1 / years) - 1 if years > 0 else total_return

    excess_returns = []
    monthly_rf_rates = []

    for date_i, ret_i in zip(monthly_returns.index, monthly_returns.values):
        rf_annual = risk_free_rate  

        if irx_data is not None and not irx_data.empty:
            try:

                irx_value = irx_data.loc[irx_data.index <= date_i].iloc[-1]

                if isinstance(irx_value, pd.Series):
                    irx_close = irx_value["Close"]
                    if isinstance(irx_close, pd.Series):
                        irx_close = irx_close.iloc[0]
                else:
                    irx_close = float(irx_value)

                if irx_close > 0:
                    rf_annual = float(irx_close) / 100
            except (KeyError, IndexError, ValueError):
                pass

        rf_monthly = (1 + rf_annual) ** (1 / 12) - 1
        monthly_rf_rates.append(rf_monthly)
        excess_returns.append(ret_i - rf_monthly)

    excess_returns = np.array(excess_returns)

    try:
        sharpe_ratio = ep.sharpe_ratio(excess_returns, risk_free=0.0, period='monthly')
        sortino_ratio = ep.sortino_ratio(excess_returns, required_return=0.0, period='monthly')

        sharpe_ratio = float(sharpe_ratio) if not pd.isna(sharpe_ratio) else 0.0
        sortino_ratio = float(sortino_ratio) if not pd.isna(sortino_ratio) else 0.0
    except (ValueError, ZeroDivisionError, IndexError) as e:
        print("Empyrical calculation failed:", e)
        sharpe_ratio = 0.0
        sortino_ratio = 0.0

    return {
        'total_return': float(total_return),
        'cagr': float(cagr),
        'sharpe': sharpe_ratio,
        'sortino': sortino_ratio
    }

def calculate_exact_month_sma(price_data, months_lookback, return_lookback_data=False):
    sma_values = []
    dates = price_data.index

    for current_date in dates:
        target_date = current_date - relativedelta(months=months_lookback)

        start_idx = price_data.index.searchsorted(target_date)
        end_idx = price_data.index.get_loc(current_date)

        if start_idx <= end_idx:
            lookback_data = price_data.iloc[start_idx:end_idx + 1]
            sma = lookback_data.mean()
        else:
            sma = price_data.loc[current_date]

        sma_values.append(sma)

    if return_lookback_data:
        return pd.Series(sma_values, index=dates), lookback_data

    return pd.Series(sma_values, index=dates)

def calculate_drawdowns(values):
    if not values or len(values) == 0:
        return []

    peak = values[0]
    drawdowns = []

    for value in values:
        if value > peak:
            peak = value
        drawdown = (value - peak) / peak
        drawdowns.append(drawdown)

    return drawdowns

def safe_format(value, format_str=".2f", suffix="%"):
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

    stock_data_full = stock_data_full.copy()
    stock_data_full.index = pd.to_datetime(stock_data_full.index)
    irx_data_full.index = pd.to_datetime(irx_data_full.index)

    stock_data_full[f"SMA {sma_months}M"] = calculate_exact_month_sma(
        stock_data_full["Close"], sma_months
    )

    stock_data = stock_data_full.loc[stock_data_full.index >= pd.to_datetime(start_date)].copy()
    irx_data = irx_data_full.reindex(stock_data.index, method="ffill")

    if stock_data[f"SMA {sma_months}M"].isna().any():
        first_valid_sma = stock_data[f"SMA {sma_months}M"].first_valid_index()
        if first_valid_sma is not None:
            first_valid_value = stock_data.loc[first_valid_sma, f"SMA {sma_months}M"]
            stock_data[f"SMA {sma_months}M"].fillna(first_valid_value, inplace=True)
        else:
            stock_data[f"SMA {sma_months}M"] = stock_data["Close"]

    initial_cash = initial_investment
    cash = initial_cash
    primary_shares = 0
    portfolio_values = []
    bh_values = []

    dates = stock_data.index
    buy_and_hold_shares = initial_cash / stock_data["Close"].iloc[0]

    for i in range(len(stock_data)):
        date_i = dates[i]
        price = float(stock_data["Close"].iloc[i])
        sma = float(stock_data[f"SMA {sma_months}M"].iloc[i])

        irx_yield = 0.02  
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
            irx_yield = 0.02

        irx_daily_return = (1 + irx_yield) ** (1 / 252) - 1

        if cash > 0:
            cash *= (1 + irx_daily_return)

        should_trade = False

        if timing == "At Signal":
            should_trade = True
        elif timing == "Weekly":
            if date_i.weekday() == 4:
                should_trade = True
            elif date_i.weekday() == 3:
                next_day_exists = i + 1 < len(dates) and dates[i + 1].weekday() == 4
                if not next_day_exists:
                    should_trade = True
        elif timing == "Biweekly":
            if date_i.weekday() == 4:  
                week_num = date_i.isocalendar()[1]
                should_trade = (week_num % 2 == 0)
            elif date_i.weekday() == 3:  
                next_day_exists = i + 1 < len(dates) and dates[i + 1].weekday() == 4
                week_num = date_i.isocalendar()[1]
                if not next_day_exists:
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

    sma_metrics = calculate_performance_metrics(dates, portfolio_values, irx_data)

    return {
        'sma_months': sma_months,
        'timing': timing,
        'final_value': final_portfolio,
        'total_return': (final_portfolio - initial_investment) / initial_investment,
        'cagr': sma_metrics.get('cagr', 0),
        'sharpe': sma_metrics.get('sharpe', 0),
        'sortino': sma_metrics.get('sortino', 0),
    }

if 'simulation_results' not in st.session_state:
    st.session_state.simulation_results = None

with tab1:

    st.header("SMA vs Buy & Hold Backtest")

    with st.sidebar:
        ticker = st.text_input("Stock Ticker", "VTI", key="hist_ticker").upper()
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", date(2015, 12, 31), key="hist_start", min_value=date(1899, 12, 31), max_value=date.today())
        with col2:
            end_date = st.date_input("End Date", date.today(), key="hist_end", min_value=date(1899, 12, 31), max_value=date.today()) + relativedelta(days=1)
        sma_months = st.number_input("Analysis Period (months)", min_value=1, max_value=240, value=11, step=1, key="hist_months")
        timing = st.selectbox("Rebalance Frequency", ["At Signal", "Weekly", "Biweekly", "Monthly", "Bimonthly", "Quarterly"], key="hist_timing")
        initial_investment = st.number_input("Initial Capital ($)", min_value=1000, max_value=1000000, value=10000, step=1000, key="hist_investment")

        st.markdown("---")

    extra_lookback = relativedelta(months=sma_months + 12)  
    fetch_start_date = start_date - extra_lookback

    with st.spinner("Fetching historical data..."):
        stock_data_full = yf.download(ticker, start=fetch_start_date, end=end_date)
        irx_data_full = yf.download("^IRX", start=fetch_start_date, end=end_date)

    if stock_data_full.empty or irx_data_full.empty:
        st.error("No data found for stock or ^IRX.")
    else:

        stock_data_full.index = pd.to_datetime(stock_data_full.index)
        irx_data_full.index = pd.to_datetime(irx_data_full.index)

        stock_data_full[f"SMA {sma_months}M"] = calculate_exact_month_sma(
            stock_data_full["Close"], sma_months
        )

        stock_data = stock_data_full.loc[stock_data_full.index >= pd.to_datetime(start_date)].copy()
        irx_data = irx_data_full.reindex(stock_data.index, method="ffill")

        if stock_data[f"SMA {sma_months}M"].isna().any():
            first_valid_sma = stock_data[f"SMA {sma_months}M"].first_valid_index()
            if first_valid_sma is not None:
                first_valid_value = stock_data.loc[first_valid_sma, f"SMA {sma_months}M"]
                stock_data[f"SMA {sma_months}M"].fillna(first_valid_value, inplace=True)
            else:
                stock_data[f"SMA {sma_months}M"] = stock_data["Close"]

        initial_cash = initial_investment
        cash = initial_cash
        primary_shares = 0
        portfolio_values = []
        bh_values = []

        trades = []
        trade_id = 0
        position = 0  
        last_action = "INITIAL"

        dates = stock_data.index
        buy_and_hold_shares = initial_cash / stock_data["Close"].iloc[0]

        for i in range(len(stock_data)):
            date_i = dates[i]
            price = float(stock_data["Close"].iloc[i])
            sma = float(stock_data[f"SMA {sma_months}M"].iloc[i])

            irx_yield = 0.02  
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
                irx_yield = 0.02

            irx_daily_return = (1 + irx_yield) ** (1 / 252) - 1

            if cash > 0:
                cash *= (1 + irx_daily_return)

            should_trade = False

            if timing == "At Signal":
                should_trade = True
            elif timing == "Weekly":
                if date_i.weekday() == 4:
                    should_trade = True
                elif date_i.weekday() == 3:
                    next_day_exists = i + 1 < len(dates) and dates[i + 1].weekday() == 4
                    if not next_day_exists:
                        should_trade = True
            elif timing == "Biweekly":
                if date_i.weekday() == 4:  
                    week_num = date_i.isocalendar()[1]
                    should_trade = (week_num % 2 == 0)
                elif date_i.weekday() == 3:  
                    next_day_exists = i + 1 < len(dates) and dates[i + 1].weekday() == 4
                    week_num = date_i.isocalendar()[1]
                    if not next_day_exists:
                        should_trade = (week_num % 2 == 0)
            elif timing == "Monthly":
                if i == len(stock_data) - 1:
                    should_trade = True
                else:
                    next_date = dates[i + 1]
                    should_trade = (date_i.month != next_date.month)
            elif timing == "Bimonthly":
                if i == len(stock_data) - 1:
                    should_trade = True
                else:
                    next_date = dates[i + 1]
                    should_trade = (date_i.month != next_date.month and date_i.month % 2 == 0)
            elif timing == "Quarterly":
                if i == len(stock_data) - 1:
                    should_trade = (date_i.month in [3, 6, 9, 12])
                else:
                    next_date = dates[i + 1]
                    should_trade = (date_i.month != next_date.month and date_i.month in [3, 6, 9, 12])

            trade_executed = False
            trade_action = ""
            trade_shares = 0
            trade_amount = 0
            trade_price = price

            if should_trade:
                if price >= sma and cash > 0 and position == 0:

                    trade_shares = cash / price
                    trade_amount = cash
                    primary_shares = trade_shares
                    cash = 0
                    position = 1
                    trade_action = "BUY"
                    trade_executed = True
                    last_action = "BUY"

                elif price < sma and primary_shares > 0 and position == 1:

                    trade_shares = primary_shares
                    trade_amount = primary_shares * price
                    cash = trade_amount
                    primary_shares = 0
                    position = 0
                    trade_action = "SELL"
                    trade_executed = True
                    last_action = "SELL"

            portfolio_value = cash + (primary_shares * price)
            portfolio_values.append(float(portfolio_value))

            bh_value = float(buy_and_hold_shares * price)
            bh_values.append(float(bh_value))

            if trade_executed:
                trade_id += 1
                trades.append({
                    'trade_id': trade_id,
                    'date': date_i,
                    'action': trade_action,
                    'price': trade_price,
                    'shares': trade_shares,
                    'amount': trade_amount,
                    'portfolio_value': portfolio_value,
                    'signal': f"Price {'≥' if trade_action == 'BUY' else '<'} SMA ({sma_months}M)",
                    'cash_after': cash,
                    'shares_after': primary_shares
                })

        final_portfolio = float(portfolio_values[-1]) if portfolio_values else initial_cash
        final_bh = float(bh_values[-1]) if bh_values else initial_cash

        sma_metrics = calculate_performance_metrics(dates, portfolio_values, irx_data)
        bh_metrics = calculate_performance_metrics(dates, bh_values, irx_data)

        sma_return = (final_portfolio - initial_cash) / initial_cash * 100
        bh_return = (final_bh - initial_cash) / initial_cash * 100
        relative_perf = (final_portfolio / final_bh - 1) * 100

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                label="SMA Value",
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

        metrics = {
            "Metric": ["CAGR", "Sharpe Ratio", "Sortino Ratio"],
            "SMA": [
                sma_metrics.get("cagr", 0),
                sma_metrics.get("sharpe", 0),
                sma_metrics.get("sortino", 0)
            ],
            "Buy & Hold": [
                bh_metrics.get("cagr", 0),
                bh_metrics.get("sharpe", 0),
                bh_metrics.get("sortino", 0)
            ]
        }

        df_metrics = pd.DataFrame(metrics)

        df_metrics["Buy & Hold"] = df_metrics.apply(
            lambda row: f"{row['Buy & Hold']*100:.2f}%" if row["Metric"]=="CAGR" else f"{row['Buy & Hold']:.2f}", axis=1
        )
        df_metrics["SMA"] = df_metrics.apply(
            lambda row: f"{row['SMA']*100:.2f}%" if row["Metric"]=="CAGR" else f"{row['SMA']:.2f}", axis=1
        )

        df_metrics.set_index("Metric", inplace=True)

        st.dataframe(df_metrics, use_container_width=True)

        with st.expander("Trade History", expanded=False):
            if trades:
                trades_df = pd.DataFrame(trades)

                trades_display = trades_df.copy()
                trades_display['date'] = trades_display['date'].dt.strftime('%Y-%m-%d')
                trades_display['price'] = trades_display['price'].apply(lambda x: f"${x:.2f}")
                trades_display['amount'] = trades_display['amount'].apply(lambda x: f"${x:,.2f}")
                trades_display['portfolio_value'] = trades_display['portfolio_value'].apply(lambda x: f"${x:,.2f}")
                trades_display['cash_after'] = trades_display['cash_after'].apply(lambda x: f"${x:,.2f}")
                trades_display['shares'] = trades_display['shares'].apply(lambda x: f"{x:,.2f}")
                trades_display['shares_after'] = trades_display['shares_after'].apply(lambda x: f"{x:,.2f}")

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    total_trades = len(trades)
                    st.metric("Total Trades", total_trades)
                with col2:
                    buy_trades = len([t for t in trades if t['action'] == 'BUY'])
                    st.metric("Buy Trades", buy_trades)
                with col3:
                    sell_trades = len([t for t in trades if t['action'] == 'SELL'])
                    st.metric("Sell Trades", sell_trades)
                with col4:
                    avg_trade_amount = np.mean([t['amount'] for t in trades])
                    st.metric("Avg Trade Amount", f"${avg_trade_amount:,.0f}")

                st.dataframe(
                    trades_display[[
                        'trade_id', 'date', 'action', 'price', 'shares', 'amount', 
                        'signal', 'portfolio_value'
                    ]],
                    use_container_width=True
                )

                st.subheader("Trade Analysis")
                if len(trades) >= 2:
                    trade_returns = []
                    for i in range(1, len(trades), 2):  
                        if i < len(trades) and trades[i-1]['action'] == 'BUY' and trades[i]['action'] == 'SELL':
                            buy_price = trades[i-1]['price']
                            sell_price = trades[i]['price']
                            trade_return = (sell_price - buy_price) / buy_price * 100
                            trade_returns.append(trade_return)

                    if trade_returns:
                        col1, col2 = st.columns(2)
                        with col1:
                            avg_return = np.mean(trade_returns)
                            st.metric("Average Trade Return", f"{avg_return:.1f}%")
                        with col2:
                            best_trade = max(trade_returns)
                            st.metric("Best Trade", f"{best_trade:.1f}%")
            else:
                st.info("No trades were executed during this backtest period.")

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

        fig.add_trace(go.Scatter(x=monthly_dates, y=monthly_portfolio_values, name="SMA", line=dict(width=1.5, color='#4169C0')))
        fig.add_trace(go.Scatter(x=monthly_dates, y=monthly_bh_values, name="Buy & Hold", line=dict(width=1.5, color='#FFEE8C')))

        if trades:
            buy_trades = [t for t in trades if t['action'] == 'BUY']
            sell_trades = [t for t in trades if t['action'] == 'SELL']

            if buy_trades:
                fig.add_trace(go.Scatter(
                    x=[t['date'] for t in buy_trades],
                    y=[t['portfolio_value'] for t in buy_trades],
                    mode='markers',
                    name='Buy Trade',
                    marker=dict(color='#2ecc71', size=16, symbol='triangle-up', line=dict(width=0.5, color='DarkSlateGrey'))
                ))

            if sell_trades:
                fig.add_trace(go.Scatter(
                    x=[t['date'] for t in sell_trades],
                    y=[t['portfolio_value'] for t in sell_trades],
                    mode='markers',
                    name='Sell Trade',
                    marker=dict(color='#e74c3c', size=16, symbol='triangle-down', line=dict(width=0.5, color='DarkSlateGrey'))
                ))

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

        st.subheader("Annual Returns")

        annual_data = {}
        current_year = None
        year_start_portfolio = None
        year_start_bh = None

        for i in range(len(dates)):
            date = dates[i]
            year = date.year

            if year != current_year:
                if current_year is not None and year_start_portfolio is not None:
                    year_end_portfolio = portfolio_values[i-1] if i > 0 else portfolio_values[i]
                    year_end_bh = bh_values[i-1] if i > 0 else bh_values[i]

                    portfolio_return = (year_end_portfolio - year_start_portfolio) / year_start_portfolio * 100
                    bh_return = (year_end_bh - year_start_bh) / year_start_bh * 100

                    annual_data[current_year] = {
                        'portfolio_return': portfolio_return,
                        'bh_return': bh_return
                    }

                current_year = year
                year_start_portfolio = portfolio_values[i]
                year_start_bh = bh_values[i]

        if current_year is not None and year_start_portfolio is not None:
            year_end_portfolio = portfolio_values[-1]
            year_end_bh = bh_values[-1]

            portfolio_return = (year_end_portfolio - year_start_portfolio) / year_start_portfolio * 100
            bh_return = (year_end_bh - year_start_bh) / year_start_bh * 100

            annual_data[current_year] = {
                'portfolio_return': portfolio_return,
                'bh_return': bh_return
            }

        if annual_data:
            years = list(annual_data.keys())
            portfolio_returns = [annual_data[year]['portfolio_return'] for year in years]
            if portfolio_returns[0] == 0:
                portfolio_returns = portfolio_returns[1:]
                years = years[1:]
            bh_returns = [annual_data[year]['bh_return'] for year in years]

            fig_annual = go.Figure()

            fig_annual.add_trace(go.Bar(
                x=years,
                y=portfolio_returns,
                name="SMA",
                marker_color='#4169C0'
            ))

            fig_annual.add_trace(go.Bar(
                x=years,
                y=bh_returns,
                name="Buy & Hold",
                marker_color='#FFEE8C'
            ))

            fig_annual.update_layout(
                xaxis_title="Year",
                yaxis_title="Annual Return (%)",
                template="plotly_dark",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=40, r=20, t=60, b=40),
                height=500,
                bargap=0.15,
                bargroupgap=0.1
            )

            fig_annual.update_traces(
                texttemplate='%{y:.1f}%',
                textposition='outside'
            )

            st.plotly_chart(fig_annual, use_container_width=True)

        else:
            st.info("Insufficient data to calculate annual returns.")

        st.subheader("Monthly Drawdowns")

        sma_drawdowns = calculate_drawdowns(portfolio_values)
        bh_drawdowns = calculate_drawdowns(bh_values)

        monthly_drawdown_dates = []
        monthly_sma_drawdowns = []
        monthly_bh_drawdowns = []

        dates_series = pd.Series(dates)
        grouped = dates_series.groupby([dates_series.dt.year, dates_series.dt.month])

        monthly_drawdown_dates = []
        monthly_sma_drawdowns = []
        monthly_bh_drawdowns = []

        for (year, month), idx in grouped.indices.items():
            last_idx = idx[-1]
            monthly_drawdown_dates.append(dates[last_idx])
            monthly_sma_drawdowns.append(sma_drawdowns[last_idx])
            monthly_bh_drawdowns.append(bh_drawdowns[last_idx])

        col1, col2 = st.columns(2)

        with col1:
            max_sma_dd = min(monthly_sma_drawdowns) * 100 if monthly_sma_drawdowns else 0
            st.metric("Max SMA Drawdown", f"{max_sma_dd:.1f}%")

        with col2:
            max_bh_dd = min(monthly_bh_drawdowns) * 100 if monthly_bh_drawdowns else 0
            st.metric("Max Buy & Hold Drawdown", f"{max_bh_dd:.1f}%")

        fig_drawdown = go.Figure()

        fig_drawdown.add_trace(go.Scatter(
            x=monthly_drawdown_dates, 
            y=[dd * 100 for dd in monthly_bh_drawdowns],
            name="Buy & Hold",
            line=dict(width=2.5, color="#FFEE8C"),
            fill='tozeroy'
        ))

        fig_drawdown.add_trace(go.Scatter(
            x=monthly_drawdown_dates, 
            y=[dd * 100 for dd in monthly_sma_drawdowns],
            name="SMA",
            line=dict(width=2.5, color="#4169C0"),
            fill='tozeroy'
        ))

        fig_drawdown.update_layout(
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            template="plotly_dark",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=20, t=60, b=40),
            height=500,
            yaxis=dict(ticksuffix="%")
        )

        st.plotly_chart(fig_drawdown, use_container_width=True)

with tab2:

    st.header("Current Market Analysis")

    end_date_tab2 = st.session_state.get("hist_end", date.today()) + relativedelta(days=1)

    @st.cache_data(show_spinner=True)
    def fetch_stock(ticker_symbol: str, start_date, end_date):
        return yf.download(ticker_symbol.upper(), start=start_date, end=end_date)

    sma_months_tab2 = st.session_state.get("hist_months", 11)

    extra_lookback_tab2 = relativedelta(months=sma_months_tab2 + 12)  
    start_date_current = end_date_tab2 - extra_lookback_tab2

    try:
        stock_data = fetch_stock(ticker, start_date_current, end_date_tab2)
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        stock_data = pd.DataFrame()

    if stock_data.empty:
        st.warning(f"No data found for {ticker} in the selected date range.")
    else:
        price_column = "Close"
        price_series = stock_data[price_column].squeeze()

        last_price = float(price_series.iloc[-1])

        df = pd.DataFrame({price_column: price_series})
        sma, look = calculate_exact_month_sma(df[price_column], sma_months_tab2, True) 

        signal = last_price >= sma.iloc[-1]
        latest_signal = "BUY" if signal else "SELL"
        signal_color = "normal" if latest_signal == "BUY" else "inverse"

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Price", f"${last_price:,.2f}")
        with col2:
            st.metric(f"SMA {sma_months_tab2} Months", f"${sma.iloc[-1]:,.2f}")
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
            y=sma,
            mode='lines',
            name=f'SMA {sma_months_tab2}M',
            line=dict(width=2, dash='dash', color='#ff7f0e')
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
                look,
                use_container_width=True
            )

with tab3:
    st.header("Parameter Simulation Results")

    simulate_button = st.button("Run Parameter Simulation", use_container_width=True)

    if simulate_button:

        if 'stock_data_full' in locals() and 'irx_data_full' in locals():
            sma_months_range = [7, 8, 9, 10, 11, 12, 15, 18]
            rebalance_freqs = ["Monthly", "Bimonthly", "Quarterly"]

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
        else:
            st.error("Please run the historical backtest first to generate data for simulation.")

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

            best_by_cagr = results_df.loc[results_df['cagr'].idxmax()]
            best_by_sharpe = results_df.loc[results_df['sharpe'].idxmax()]
            best_by_sortino = results_df.loc[results_df['sortino'].idxmax()]

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                        "Best CAGR",
                        f"SMA {int(best_by_cagr['sma_months'])}M\n{best_by_cagr['timing']}",
                        f"{best_by_cagr['cagr']*100:.1f}%"
                    )

            with col2:
                st.metric(
                        "Best Sharpe Ratio",
                        f"SMA {int(best_by_sharpe['sma_months'])}M\n{best_by_sharpe['timing']}",
                        f"{best_by_sharpe['sharpe']:.2f}"
                    )

            with col3:
                st.metric(
                        "Best Sortino Ratio",
                        f"SMA {int(best_by_sortino['sma_months'])}M\n{best_by_sortino['timing']}",
                        f"{best_by_sortino['sortino']:.2f}"
                    )

            top_results = results_df.sort_values(by='cagr', ascending=False).copy()
            top_results['total_return'] = (top_results['total_return'] * 100).round(1)
            top_results['cagr'] = (top_results['cagr'] * 100).round(1)
            top_results['sharpe'] = (top_results['sharpe']).round(2)
            top_results['sortino'] = (top_results['sortino']).round(2)
            top_results['final_value'] = top_results['final_value'].round(0)

            top_results.columns = ['SMA Months', 'Rebalance Freq', 'Final Value', 'Total Return %', 'CAGR %', 'Sharpe Ratio', 'Sortino Ratio']
            st.dataframe(top_results, use_container_width=True)