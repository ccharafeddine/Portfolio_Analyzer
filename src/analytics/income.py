"""
Dividend and income tracking analytics.

Fetches dividend data via yfinance and computes income metrics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None


def fetch_dividends(
    ticker: str,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> pd.Series:
    """
    Fetch dividend history for a single ticker.

    Returns a pd.Series indexed by ex-date with dividend amounts.
    Returns empty Series on any error.
    """
    try:
        if yf is None:
            return pd.Series(dtype=float, name=ticker)
        t = yf.Ticker(ticker)
        divs = t.dividends
        if divs.empty:
            return pd.Series(dtype=float, name=ticker)

        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        # Ensure timezone-naive comparison
        if divs.index.tz is not None:
            divs.index = divs.index.tz_localize(None)
        mask = (divs.index >= start_ts) & (divs.index <= end_ts)
        filtered = divs.loc[mask]
        filtered.name = ticker
        return filtered
    except Exception:
        return pd.Series(dtype=float, name=ticker)


def compute_income_summary(
    holdings: pd.DataFrame,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
) -> pd.DataFrame:
    """
    Compute dividend income summary for each holding.

    Parameters
    ----------
    holdings : DataFrame with Ticker, Shares, PurchasePrice columns
    start_date, end_date : date range

    Returns
    -------
    DataFrame with columns:
        Ticker, Shares, AnnualDividendPerShare, AnnualIncome,
        YieldOnCost, CurrentYield, IncomeGrowthRate
    """
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    years = max((end_ts - start_ts).days / 365.25, 0.5)

    rows = []
    for _, row in holdings.iterrows():
        ticker = row["Ticker"]
        shares = row["Shares"]
        purchase_price = row["PurchasePrice"]

        divs = fetch_dividends(ticker, start_date, end_date)

        if divs.empty:
            rows.append({
                "Ticker": ticker,
                "Shares": shares,
                "AnnualDividendPerShare": 0.0,
                "AnnualIncome": 0.0,
                "YieldOnCost": 0.0,
                "CurrentYield": 0.0,
                "IncomeGrowthRate": 0.0,
            })
            continue

        total_div_per_share = divs.sum()
        annual_div = total_div_per_share / years
        annual_income = annual_div * shares
        yield_on_cost = annual_div / purchase_price if purchase_price > 0 else 0.0

        # Current yield: most recent annualized dividend / current price
        try:
            current_price = yf.Ticker(ticker).info.get(
                "regularMarketPrice",
                yf.Ticker(ticker).info.get("previousClose", purchase_price),
            )
        except Exception:
            current_price = purchase_price
        current_yield = annual_div / current_price if current_price > 0 else 0.0

        # Income growth rate: compare first-year vs last-year dividends
        growth = 0.0
        if years >= 2:
            mid = start_ts + (end_ts - start_ts) / 2
            first_half = divs[divs.index < mid].sum()
            second_half = divs[divs.index >= mid].sum()
            if first_half > 0:
                growth = (second_half / first_half - 1.0)

        rows.append({
            "Ticker": ticker,
            "Shares": shares,
            "AnnualDividendPerShare": round(annual_div, 4),
            "AnnualIncome": round(annual_income, 2),
            "YieldOnCost": round(yield_on_cost, 4),
            "CurrentYield": round(current_yield, 4),
            "IncomeGrowthRate": round(growth, 4),
        })

    return pd.DataFrame(rows)


def portfolio_income_metrics(
    income_summary: pd.DataFrame,
    total_invested: float,
) -> dict:
    """
    Compute aggregate portfolio income metrics.

    Returns dict with:
        total_annual_income, portfolio_yield, avg_yield_on_cost, n_payers
    """
    if income_summary.empty:
        return {
            "total_annual_income": 0.0,
            "portfolio_yield": 0.0,
            "avg_yield_on_cost": 0.0,
            "n_payers": 0,
        }

    total_income = income_summary["AnnualIncome"].sum()
    portfolio_yield = total_income / total_invested if total_invested > 0 else 0.0
    avg_yoc = income_summary["YieldOnCost"].mean()
    n_payers = int((income_summary["AnnualIncome"] > 0).sum())

    return {
        "total_annual_income": round(total_income, 2),
        "portfolio_yield": round(portfolio_yield, 4),
        "avg_yield_on_cost": round(avg_yoc, 4),
        "n_payers": n_payers,
    }


def cumulative_income_series(
    holdings: pd.DataFrame,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
) -> pd.Series:
    """
    Compute cumulative dividend income over time.

    Returns a pd.Series indexed by date with cumulative dollar income.
    """
    all_income = []

    for _, row in holdings.iterrows():
        ticker = row["Ticker"]
        shares = row["Shares"]
        divs = fetch_dividends(ticker, start_date, end_date)
        if not divs.empty:
            income = divs * shares
            all_income.append(income)

    if not all_income:
        return pd.Series(dtype=float, name="CumulativeIncome")

    combined = pd.concat(all_income, axis=0).sort_index()
    # Group by date in case multiple tickers pay on same day
    combined = combined.groupby(combined.index).sum()
    cumulative = combined.cumsum()
    cumulative.name = "CumulativeIncome"
    return cumulative
