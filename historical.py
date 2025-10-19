import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
import plotly.graph_objects as go

st.set_page_config(page_title="SMA Trading Backtester", layout="wide")
st.title("📈 SMA Strategy vs Buy & Hold")

ticker = st.text_input("Enter stock ticker:", "AAPL").upper()
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start Date", date(2015, 1, 1))
with col2:
    end_date = st.date_input("End Date", date.today())

sma_months = st.slider("SMA Lookback (months)", 3, 24, 11)
timing = st.selectbox(
    "Trading Frequency",
    ["At Signal", "Weekly", "Biweekly", "Monthly", "Bimonthly", "Quarterly"])

sma_lookback = relativedelta(months=sma_months)
fetch_start_date = start_date - sma_lookback

with st.spinner("Fetching data..."):
    stock_data_full = yf.download(ticker, start=fetch_start_date, end=end_date)
    irx_data_full = yf.download("^IRX", start=fetch_start_date, end=end_date)

if stock_data_full.empty or irx_data_full.empty:
    st.error("No data found for stock or ^IRX.")
    st.stop()

stock_data_full.index = pd.to_datetime(stock_data_full.index)
irx_data_full.index = pd.to_datetime(irx_data_full.index)

window_size = sma_months * 21
stock_data_full[f"{sma_months}M_SMA"] = stock_data_full["Close"].rolling(
    window=window_size, min_periods=1).mean()

stock_data = stock_data_full.loc[stock_data_full.index >= pd.to_datetime(
    start_date)].copy()
irx_data = irx_data_full.reindex(stock_data.index, method="ffill")

if stock_data[f"{sma_months}M_SMA"].isna().any():
    first_valid_sma = stock_data[f"{sma_months}M_SMA"].first_valid_index()
    if first_valid_sma is not None:
        first_valid_value = stock_data.loc[first_valid_sma,
                                           f"{sma_months}M_SMA"]
        stock_data[f"{sma_months}M_SMA"].fillna(first_valid_value,
                                                inplace=True)
    else:

        stock_data[f"{sma_months}M_SMA"] = stock_data["Close"]

initial_cash = 10000
cash = initial_cash
primary_shares = 0
portfolio_values = []
bh_values = []

dates = stock_data.index
buy_and_hold_shares = initial_cash / stock_data["Close"].iloc[0]

for i in range(len(stock_data)):
    date_i = dates[i]
    price = float(stock_data["Close"].iloc[i])
    sma = float(stock_data[f"{sma_months}M_SMA"].iloc[i])

    irx_yield = 0.0

    try:
        irx_value = irx_data.loc[date_i]

        if isinstance(irx_value, pd.DataFrame):
            irx_close = irx_value["Close"].iloc[
                0] if not irx_value.empty else 0.0
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

    irx_daily_return = (1 + irx_yield)**(1 / 252) - 1

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
            should_trade = (date_i.month != next_date.month
                            and date_i.month % 2 == 0)
    elif timing == "Quarterly":
        if i == len(stock_data) - 1:
            should_trade = (date_i.month in [3, 6, 9, 12])
        else:
            next_date = dates[i + 1]
            should_trade = (date_i.month != next_date.month
                            and date_i.month in [3, 6, 9, 12])

    if should_trade:
        if price >= sma and cash > 0:

            primary_shares = cash / price
            cash = 0
        elif price < sma and primary_shares > 0:

            cash = primary_shares * price
            primary_shares = 0

    portfolio_value = cash + (primary_shares * price)
    portfolio_values.append(portfolio_value)

    bh_value = float(buy_and_hold_shares * price)
    bh_values.append(bh_value)

final_portfolio = float(
    portfolio_values[-1]) if portfolio_values else initial_cash
final_bh = float(bh_values[-1]) if bh_values else initial_cash

fig = go.Figure()
fig.add_trace(
    go.Scatter(x=dates,
               y=portfolio_values,
               name=f"{timing} SMA Strategy",
               line=dict(width=2)))
fig.add_trace(
    go.Scatter(x=dates, y=bh_values, name="Buy & Hold", line=dict(width=2)))
fig.update_layout(title=f"{ticker} | SMA {sma_months}M Strategy vs Buy & Hold",
                  xaxis_title="Date",
                  yaxis_title="Portfolio Value ($)",
                  template="plotly_white",
                  hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

col1, col2, col3 = st.columns(3)

sma_return = (final_portfolio - initial_cash) / initial_cash * 100
bh_return = (final_bh - initial_cash) / initial_cash * 100
relative_perf = (final_portfolio / final_bh - 1) * 100

col1.metric("SMA Strategy", f"${final_portfolio:,.2f}", f"{sma_return:+.2f}%")
col2.metric("Buy & Hold", f"${final_bh:,.2f}", f"{bh_return:+.2f}%")
col3.metric("Relative Performance", f"{relative_perf:+.2f}%", "vs Buy & Hold")

st.subheader("Performance Summary")
st.write(f"**Initial Investment:** ${initial_cash:,.2f}")
st.write(f"**SMA Strategy Return:** {sma_return:+.2f}%")
st.write(f"**Buy & Hold Return:** {bh_return:+.2f}%")
st.write(f"**Relative Performance:** {relative_perf:+.2f}%")

if primary_shares > 0:
    st.info(f"Final Position: {primary_shares:.2f} shares of {ticker}")
else:
    st.info(f"Final Position: ${cash:,.2f} in cash")
