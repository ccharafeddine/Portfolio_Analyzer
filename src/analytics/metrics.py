"""Single source of truth for return/risk metrics.

Conventions (every function follows these; tests pin them against empyrical):

* Inputs are *simple periodic* returns (daily or monthly), never prices.
* ``periods_per_year`` is explicit: ``DAILY`` (252) or ``MONTHLY`` (12). Use
  :func:`infer_periods_per_year` when you only have a DatetimeIndex.
* Risk-free / minimum acceptable return is given **annually** and converted to a
  per-period rate geometrically: ``(1 + rf) ** (1 / periods_per_year) - 1``.
* Annualized return is geometric (CAGR). Annualized volatility is
  ``std(ddof=1) * sqrt(periods_per_year)``.
* Sharpe  = mean(r - rf_p) / std(r - rf_p, ddof=1) * sqrt(ppy).
* Sortino = mean(r - mar_p) * ppy / (downside_dev_p * sqrt(ppy)), where
  ``downside_dev_p = sqrt(mean(min(r - mar_p, 0) ** 2))`` taken over **all**
  periods (not just the losing ones). MAR defaults to the risk-free rate so
  Sharpe and Sortino share a hurdle.
* Calmar  = CAGR / |max drawdown| over the full sample supplied.
* VaR/CVaR are historical, per-period, returned as negative numbers.
* CAPM alpha/beta come from OLS on excess returns, with standard errors and
  t-stats; ``alpha_annual`` is ``alpha_p * ppy``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DAILY = 252
MONTHLY = 12
WEEKLY = 52


def infer_periods_per_year(index: pd.Index) -> int:
    """Guess the sampling frequency of a DatetimeIndex (daily/weekly/monthly)."""
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 3:
        return DAILY
    median_days = float(pd.Series(index).diff().dt.days.median())
    if median_days >= 25:
        return MONTHLY
    if median_days >= 5:
        return WEEKLY
    return DAILY


def periodic_rate(annual_rate: float, periods_per_year: int) -> float:
    """Geometric de-annualization of a rate."""
    return (1.0 + float(annual_rate)) ** (1.0 / periods_per_year) - 1.0


def _clean(returns) -> pd.Series:
    return pd.Series(returns, dtype=float).dropna()


def annualized_return(returns, periods_per_year: int = DAILY) -> float:
    r = _clean(returns)
    if r.empty:
        return np.nan
    gross = float((1.0 + r).prod())
    if gross <= 0:
        return np.nan
    return gross ** (periods_per_year / len(r)) - 1.0


def annualized_volatility(returns, periods_per_year: int = DAILY) -> float:
    r = _clean(returns)
    if len(r) < 2:
        return np.nan
    return float(r.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(returns, rf_annual: float = 0.0, periods_per_year: int = DAILY) -> float:
    r = _clean(returns)
    if len(r) < 2:
        return np.nan
    ex = r - periodic_rate(rf_annual, periods_per_year)
    sd = ex.std(ddof=1)
    if not sd or np.isnan(sd):
        return np.nan
    return float(ex.mean() / sd * np.sqrt(periods_per_year))


def downside_deviation(returns, mar_annual: float = 0.0, periods_per_year: int = DAILY) -> float:
    """Annualized downside deviation below MAR, averaged over *all* periods."""
    r = _clean(returns)
    if r.empty:
        return np.nan
    shortfall = np.minimum(r - periodic_rate(mar_annual, periods_per_year), 0.0)
    return float(np.sqrt(np.mean(shortfall ** 2)) * np.sqrt(periods_per_year))


def sortino_ratio(returns, mar_annual: float = 0.0, periods_per_year: int = DAILY) -> float:
    r = _clean(returns)
    if len(r) < 2:
        return np.nan
    dd = downside_deviation(r, mar_annual, periods_per_year)
    if not dd or np.isnan(dd):
        return np.nan
    excess = (r - periodic_rate(mar_annual, periods_per_year)).mean() * periods_per_year
    return float(excess / dd)


def drawdown_series(values: pd.Series) -> pd.Series:
    v = pd.Series(values, dtype=float).dropna().sort_index()
    if v.empty:
        return pd.Series(dtype=float)
    return v / v.cummax() - 1.0


def max_drawdown(values: pd.Series) -> float:
    """Max drawdown of a *value/level* series (negative decimal)."""
    dd = drawdown_series(values)
    return float(dd.min()) if not dd.empty else np.nan


def max_drawdown_from_returns(returns) -> float:
    r = _clean(returns)
    if r.empty:
        return np.nan
    levels = pd.concat([pd.Series([1.0]), (1.0 + r).cumprod()], ignore_index=True)
    return max_drawdown(levels)


def calmar_ratio(returns, periods_per_year: int = DAILY) -> float:
    mdd = max_drawdown_from_returns(returns)
    if not mdd or np.isnan(mdd):
        return np.nan
    return float(annualized_return(returns, periods_per_year) / abs(mdd))


def var_cvar(returns, alpha: float = 0.95) -> tuple[float, float]:
    """Historical per-period VaR and CVaR (expected shortfall), negative numbers."""
    r = _clean(returns)
    if r.empty:
        return np.nan, np.nan
    var_val = float(r.quantile(1.0 - alpha))
    tail = r[r <= var_val]
    return var_val, (float(tail.mean()) if not tail.empty else np.nan)


def tracking_error(port, bench, periods_per_year: int = MONTHLY) -> float:
    a, b = pd.Series(port).align(pd.Series(bench), join="inner")
    diff = (a - b).dropna()
    if len(diff) < 2:
        return np.nan
    return float(diff.std(ddof=1) * np.sqrt(periods_per_year))


def information_ratio(port, bench, periods_per_year: int = MONTHLY) -> float:
    a, b = pd.Series(port).align(pd.Series(bench), join="inner")
    diff = (a - b).dropna()
    te = tracking_error(a, b, periods_per_year)
    if not te or np.isnan(te):
        return np.nan
    return float(diff.mean() * periods_per_year / te)


@dataclass
class CapmStats:
    alpha: float          # per period
    alpha_annual: float   # alpha * periods_per_year
    beta: float
    alpha_se: float
    beta_se: float
    t_alpha: float
    t_beta: float
    r_squared: float
    n_obs: int
    periods_per_year: int


def capm(port, bench, rf_annual: float = 0.0, periods_per_year: int = MONTHLY) -> CapmStats:
    """OLS of (Rp - Rf) on (Rb - Rf) with classical standard errors."""
    df = pd.concat([pd.Series(port, dtype=float), pd.Series(bench, dtype=float)], axis=1).dropna()
    n = len(df)
    nan = CapmStats(np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, n, periods_per_year)
    if n < 5:
        return nan
    rf_p = periodic_rate(rf_annual, periods_per_year)
    y = df.iloc[:, 0].to_numpy() - rf_p
    x = df.iloc[:, 1].to_numpy() - rf_p
    xm = x.mean()
    sxx = float(((x - xm) ** 2).sum())
    if sxx == 0:
        return nan
    beta = float(((x - xm) * (y - y.mean())).sum() / sxx)
    alpha = float(y.mean() - beta * xm)
    resid = y - alpha - beta * x
    s2 = float((resid ** 2).sum() / (n - 2))
    se_b = np.sqrt(s2 / sxx)
    se_a = np.sqrt(s2 * (1.0 / n + xm ** 2 / sxx))
    sst = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((resid ** 2).sum()) / sst if sst > 0 else np.nan
    return CapmStats(
        alpha=alpha, alpha_annual=alpha * periods_per_year, beta=beta,
        alpha_se=float(se_a), beta_se=float(se_b),
        t_alpha=alpha / se_a if se_a > 0 else np.nan,
        t_beta=beta / se_b if se_b > 0 else np.nan,
        r_squared=r2, n_obs=n, periods_per_year=periods_per_year,
    )
