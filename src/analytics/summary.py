"""Table-shaped summaries for the redesigned screens (holdings, period returns,
monthly calendar). Pure functions over pipeline outputs; no UI imports."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analytics import metrics as M


def holdings_summary(
    holdings: pd.DataFrame,
    prices: pd.DataFrame,
    benchmark: str | None = None,
    lookback_days: int = 252,
) -> pd.DataFrame:
    """One row per position plus a ``Portfolio`` total row.

    Columns: Ticker, Shares, Price, Value, Weight, Return, Contribution,
    RiskContribution, Beta. Weights are current (drifted) market-value weights;
    Return is since purchase; Contribution = initial weight x return (sums to
    the portfolio return); RiskContribution is each name's share of portfolio
    variance over the last ``lookback_days`` (w_i * (Sigma w)_i / w'Sigma w);
    Beta is vs ``benchmark`` daily returns over the same window.
    """
    cols = ["Ticker", "Shares", "Price", "Value", "Weight", "Return",
            "Contribution", "RiskContribution", "Beta"]
    if holdings is None or holdings.empty or prices is None or prices.empty:
        return pd.DataFrame(columns=cols)
    h = holdings.copy()
    h = h[h["Ticker"].isin(prices.columns)]
    if h.empty:
        return pd.DataFrame(columns=cols)
    last = prices[h["Ticker"]].ffill().iloc[-1]
    h["Price"] = h["Ticker"].map(last)
    h["Value"] = h["Shares"] * h["Price"]
    total_value = float(h["Value"].sum())
    total_invested = float(h["Invested"].sum())
    h["Weight"] = h["Value"] / total_value if total_value else np.nan
    h["Return"] = h["Value"] / h["Invested"] - 1.0
    w0 = h["Invested"] / total_invested if total_invested else np.nan
    h["Contribution"] = w0 * h["Return"]

    rets = prices[h["Ticker"]].pct_change().iloc[1:].tail(lookback_days)
    rc = pd.Series(np.nan, index=h.index)
    beta = pd.Series(np.nan, index=h.index)
    if len(rets) > 20:
        cov = rets.cov().to_numpy()
        w = h["Weight"].to_numpy()
        var = float(w @ cov @ w)
        if var > 0:
            rc = pd.Series(w * (cov @ w) / var, index=h.index)
        if benchmark and benchmark in prices.columns:
            b = prices[benchmark].pct_change().iloc[1:].tail(lookback_days)
            bv = float(b.var())
            if bv > 0:
                beta = pd.Series([float(rets[t].cov(b)) / bv for t in h["Ticker"]],
                                 index=h.index)
    h["RiskContribution"] = rc
    h["Beta"] = beta
    out = h[cols].sort_values("Weight", ascending=False).reset_index(drop=True)
    total = {
        "Ticker": "Portfolio", "Shares": np.nan, "Price": np.nan,
        "Value": total_value, "Weight": float(out["Weight"].sum()),
        "Return": total_value / total_invested - 1.0 if total_invested else np.nan,
        "Contribution": float(out["Contribution"].sum()),
        "RiskContribution": float(out["RiskContribution"].sum(min_count=1)),
        "Beta": float((out["Weight"] * out["Beta"]).sum(min_count=1)),
    }
    return pd.concat([out, pd.DataFrame([total])], ignore_index=True)


def _window_return(v: pd.Series, start: pd.Timestamp, annualize: bool) -> float:
    s = v.loc[v.index >= start].dropna()
    prior = v.loc[v.index < start].dropna()
    if prior.empty or s.empty:
        return np.nan
    gross = float(s.iloc[-1] / prior.iloc[-1])
    if annualize:
        years = (s.index[-1] - prior.index[-1]).days / 365.25
        return gross ** (1 / years) - 1 if years > 0 else np.nan
    return gross - 1.0


def period_returns(values: pd.Series, bench: pd.Series | None = None) -> pd.DataFrame:
    """1M / 3M / YTD / 1Y / 3Y ann. / since-inception ann. for portfolio and benchmark."""
    v = values.dropna()
    if v.empty:
        return pd.DataFrame(columns=["Period", "Portfolio", "Benchmark", "Excess"])
    end = v.index[-1]
    windows = [
        ("1M", end - pd.DateOffset(months=1), False),
        ("3M", end - pd.DateOffset(months=3), False),
        ("YTD", pd.Timestamp(end.year, 1, 1), False),
        ("1Y", end - pd.DateOffset(years=1), False),
        ("3Y ann.", end - pd.DateOffset(years=3), True),
    ]
    rows = []
    for label, start, ann in windows:
        p = _window_return(v, start, ann)
        b = _window_return(bench.dropna(), start, ann) if bench is not None else np.nan
        rows.append((label, p, b))
    years = (v.index[-1] - v.index[0]).days / 365.25
    si = (float(v.iloc[-1] / v.iloc[0])) ** (1 / years) - 1 if years > 0 else np.nan
    bsi = np.nan
    if bench is not None and not bench.dropna().empty:
        bb = bench.dropna()
        by = (bb.index[-1] - bb.index[0]).days / 365.25
        bsi = float(bb.iloc[-1] / bb.iloc[0]) ** (1 / by) - 1 if by > 0 else np.nan
    rows.append(("Since incep. ann.", si, bsi))
    df = pd.DataFrame(rows, columns=["Period", "Portfolio", "Benchmark"])
    df["Excess"] = df["Portfolio"] - df["Benchmark"]
    return df


def monthly_calendar(values: pd.Series, bench: pd.Series | None = None) -> pd.DataFrame:
    """Year x month table of simple monthly returns, plus Year and benchmark-year columns."""
    v = values.dropna()
    if v.empty:
        return pd.DataFrame()
    m = v.resample("ME").last()
    mr = m.pct_change()
    mr.iloc[0] = m.iloc[0] / v.iloc[0] - 1.0  # first (possibly partial) month
    tbl = pd.DataFrame({"Year": mr.index.year, "Month": mr.index.month, "r": mr.values})
    cal = tbl.pivot(index="Year", columns="Month", values="r").reindex(columns=range(1, 13))
    cal.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    cal["Year total"] = (1 + mr).groupby(mr.index.year).prod() - 1
    if bench is not None and not bench.dropna().empty:
        b = bench.dropna()
        ye = b.resample("YE").last()
        start = pd.concat([b.iloc[:1], ye.iloc[:-1]])
        cal["Benchmark"] = pd.Series(ye.values / start.values - 1, index=ye.index.year)
    return cal


def data_quality(prices: pd.DataFrame) -> dict:
    """As-of date, number of trading days and gap count for the status bar."""
    if prices is None or prices.empty:
        return {"as_of": None, "days": 0, "gaps": 0}
    gaps = int(prices.isna().sum().sum())
    return {"as_of": prices.index[-1], "days": int(len(prices)), "gaps": gaps}


__all__ = ["holdings_summary", "period_returns", "monthly_calendar", "data_quality", "M"]
