"""
Data transformation utilities.

Pure functions that take DataFrames/Series in, return DataFrames/Series out.
No IO, no side effects, fully testable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns from price levels."""
    return prices.pct_change().iloc[1:]


def monthly_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Resample daily prices to month-end."""
    return prices.resample("ME").last()


def monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Month-over-month returns from daily prices."""
    mp = monthly_prices(prices)
    return mp.pct_change().dropna(how="all")


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns."""
    return np.log(prices / prices.shift(1)).iloc[1:]


def annualize_return(
    returns: pd.Series, periods_per_year: int = 252
) -> float:
    """Geometric annualized return from periodic simple returns."""
    r = returns.dropna()
    if r.empty:
        return np.nan
    gross = (1.0 + r).prod()
    n = len(r)
    if n == 0 or gross <= 0:
        return np.nan
    return float(gross ** (periods_per_year / n) - 1.0)


def annualize_vol(
    returns: pd.Series, periods_per_year: int = 252
) -> float:
    """Annualized volatility from periodic returns."""
    r = returns.dropna()
    if r.empty:
        return np.nan
    return float(r.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series,
    rf_annual: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Annualized Sharpe ratio."""
    r = returns.dropna()
    if r.empty:
        return np.nan
    rf_per = rf_annual / periods_per_year
    excess = r - rf_per
    vol = excess.std(ddof=1)
    if vol == 0 or np.isnan(vol):
        return np.nan
    return float(np.sqrt(periods_per_year) * excess.mean() / vol)


def max_drawdown(values: pd.Series) -> float:
    """Maximum drawdown from a value series. Returns negative decimal."""
    v = values.dropna().astype(float).sort_index()
    if v.empty:
        return np.nan
    running_max = v.cummax()
    dd = v / running_max - 1.0
    return float(dd.min())


def drawdown_series(values: pd.Series) -> pd.Series:
    """Full drawdown time series from portfolio values."""
    v = values.dropna().astype(float).sort_index()
    if v.empty:
        return pd.Series(dtype=float)
    running_max = v.cummax()
    return v / running_max - 1.0


def var_cvar(
    returns: pd.Series, alpha: float = 0.95
) -> tuple[float, float]:
    """
    Historical Value-at-Risk and Conditional VaR (Expected Shortfall).

    Returns (VaR, CVaR) at the given confidence level.
    Both are negative numbers representing losses.
    """
    r = returns.dropna().astype(float)
    if r.empty:
        return np.nan, np.nan
    p = 1.0 - alpha
    var_val = float(r.quantile(p))
    tail = r[r <= var_val]
    cvar_val = float(tail.mean()) if not tail.empty else np.nan
    return var_val, cvar_val


def gain_to_pain(returns: pd.Series) -> float | None:
    """Gain-to-Pain ratio: sum(gains) / |sum(losses)|."""
    r = returns.dropna()
    if r.empty:
        return None
    gains = r[r > 0].sum()
    losses = r[r < 0].sum()
    if losses >= 0:
        return None
    return float(gains / abs(losses))


def align_series(*series: pd.Series, how: str = "inner") -> list[pd.Series]:
    """Align multiple Series on a common DatetimeIndex."""
    if not series:
        return []
    df = pd.concat(series, axis=1, join=how).dropna()
    return [df.iloc[:, i] for i in range(df.shape[1])]


def normalize_to_base(
    series: pd.Series, base: float = 1_000_000.0
) -> pd.Series:
    """Scale a value series so its first non-NaN value equals `base`."""
    s = series.dropna()
    if s.empty:
        return series
    start_val = float(s.iloc[0])
    if start_val == 0:
        return pd.Series(np.nan, index=series.index)
    return series / start_val * base
