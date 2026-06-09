import pandas as pd
import numpy as np
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
import empyrical as ep

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

        if cash > 0 and primary_shares == 0:
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

