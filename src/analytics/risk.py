"""
Enhanced risk analytics.

New features beyond the original codebase:
- Stress testing against historical scenarios
- Marginal risk contribution per asset
- Rolling metrics (vol, Sharpe, beta)
- Correlation regime detection
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.data import transforms as T


# ──────────────────────────────────────────────────────────────
# Stress testing
# ──────────────────────────────────────────────────────────────

# Named historical stress scenarios: (name, start, end, description)
HISTORICAL_SCENARIOS = [
    ("COVID Crash", "2020-02-19", "2020-03-23", "Pandemic market crash"),
    ("2022 Bear Market", "2022-01-03", "2022-10-12", "Fed tightening selloff"),
    ("SVB Crisis", "2023-03-08", "2023-03-15", "Banking sector stress"),
    ("Aug 2024 Unwind", "2024-07-16", "2024-08-05", "Yen carry trade unwind"),
    ("GFC Peak-Trough", "2007-10-09", "2009-03-09", "Global Financial Crisis"),
    ("Taper Tantrum", "2013-05-22", "2013-06-24", "Fed taper announcement"),
    ("China Deval", "2015-08-10", "2015-08-25", "Chinese yuan devaluation"),
    ("VIX-mageddon", "2018-01-26", "2018-02-08", "Volatility spike"),
]


@dataclass
class StressResult:
    """Result of a stress test against one scenario."""

    name: str
    description: str
    start: str
    end: str
    portfolio_return: float
    benchmark_return: float
    max_drawdown: float
    days: int
    data_available: bool


def run_stress_tests(
    portfolio_values: pd.Series,
    benchmark_values: pd.Series,
    scenarios: list[tuple[str, str, str, str]] | None = None,
) -> list[StressResult]:
    """
    Test portfolio performance during historical stress periods.

    Parameters
    ----------
    portfolio_values : daily portfolio value series
    benchmark_values : daily benchmark value series
    scenarios : list of (name, start, end, description) tuples;
                defaults to HISTORICAL_SCENARIOS
    """
    if scenarios is None:
        scenarios = HISTORICAL_SCENARIOS

    results = []

    for name, start, end, desc in scenarios:
        try:
            start_dt = pd.Timestamp(start)
            end_dt = pd.Timestamp(end)

            # Slice portfolio
            pv = portfolio_values.loc[
                (portfolio_values.index >= start_dt)
                & (portfolio_values.index <= end_dt)
            ]
            bv = benchmark_values.loc[
                (benchmark_values.index >= start_dt)
                & (benchmark_values.index <= end_dt)
            ]

            if len(pv) < 2 or len(bv) < 2:
                results.append(StressResult(
                    name=name, description=desc, start=start, end=end,
                    portfolio_return=np.nan, benchmark_return=np.nan,
                    max_drawdown=np.nan, days=0, data_available=False,
                ))
                continue

            p_ret = float(pv.iloc[-1] / pv.iloc[0] - 1.0)
            b_ret = float(bv.iloc[-1] / bv.iloc[0] - 1.0)
            mdd = T.max_drawdown(pv)

            results.append(StressResult(
                name=name, description=desc, start=start, end=end,
                portfolio_return=p_ret, benchmark_return=b_ret,
                max_drawdown=mdd, days=len(pv), data_available=True,
            ))
        except Exception:
            results.append(StressResult(
                name=name, description=desc, start=start, end=end,
                portfolio_return=np.nan, benchmark_return=np.nan,
                max_drawdown=np.nan, days=0, data_available=False,
            ))

    return results


def stress_results_to_df(results: list[StressResult]) -> pd.DataFrame:
    """Convert stress test results to a display DataFrame."""
    rows = []
    for r in results:
        if not r.data_available:
            rows.append({
                "Scenario": r.name,
                "Period": f"{r.start} → {r.end}",
                "Portfolio": "N/A",
                "Benchmark": "N/A",
                "Max DD": "N/A",
                "Days": "—",
            })
        else:
            rows.append({
                "Scenario": r.name,
                "Period": f"{r.start} → {r.end}",
                "Portfolio": f"{r.portfolio_return:.2%}",
                "Benchmark": f"{r.benchmark_return:.2%}",
                "Max DD": f"{r.max_drawdown:.2%}",
                "Days": str(r.days),
            })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────
# Risk contribution
# ──────────────────────────────────────────────────────────────


def marginal_risk_contribution(
    weights: np.ndarray | pd.Series,
    cov: np.ndarray | pd.DataFrame,
) -> pd.Series:
    """
    Marginal contribution to risk (MCR) for each asset.

    MCR_i = (Σ w)_i / σ_p

    Returns a Series of MCR values that sum to total portfolio vol.
    """
    if isinstance(weights, pd.Series):
        names = weights.index.tolist()
        w = weights.values
    else:
        w = weights
        names = [f"Asset_{i}" for i in range(len(w))]

    if isinstance(cov, pd.DataFrame):
        C = cov.values
    else:
        C = cov

    port_vol = float(np.sqrt(w @ C @ w))
    if port_vol < 1e-12:
        return pd.Series(0.0, index=names, name="MCR")

    # MCR = w_i * (Σ w)_i / σ_p
    cov_w = C @ w
    mcr = w * cov_w / port_vol

    return pd.Series(mcr, index=names, name="MCR")


def risk_contribution_pct(
    weights: np.ndarray | pd.Series,
    cov: np.ndarray | pd.DataFrame,
) -> pd.Series:
    """Percentage contribution to total risk (sums to 100%)."""
    mcr = marginal_risk_contribution(weights, cov)
    total = mcr.sum()
    if total == 0:
        return mcr * 0
    return (mcr / total * 100).round(2)


# ──────────────────────────────────────────────────────────────
# Rolling analytics
# ──────────────────────────────────────────────────────────────


def rolling_volatility(
    returns: pd.DataFrame,
    window: int = 21,
    annualize: int = 252,
) -> pd.DataFrame:
    """Rolling annualized volatility."""
    return returns.rolling(window).std() * np.sqrt(annualize)


def rolling_sharpe(
    returns: pd.DataFrame,
    window: int = 63,
    rf_annual: float = 0.04,
    annualize: int = 252,
) -> pd.DataFrame:
    """Rolling annualized Sharpe ratio."""
    rf_per = rf_annual / annualize
    excess = returns - rf_per
    roll_mean = excess.rolling(window).mean() * annualize
    roll_vol = returns.rolling(window).std() * np.sqrt(annualize)
    return roll_mean / roll_vol


def rolling_beta(
    asset_returns: pd.Series,
    market_returns: pd.Series,
    window: int = 63,
) -> pd.Series:
    """Rolling beta of asset vs market."""
    aligned = pd.concat(
        [asset_returns.rename("a"), market_returns.rename("m")],
        axis=1,
    ).dropna()

    cov_rolling = aligned["a"].rolling(window).cov(aligned["m"])
    var_rolling = aligned["m"].rolling(window).var()

    beta = cov_rolling / var_rolling
    beta.name = f"Beta_{asset_returns.name}"
    return beta


def rolling_correlation(
    returns: pd.DataFrame,
    window: int = 63,
) -> pd.DataFrame:
    """
    Rolling average pairwise correlation.

    Returns a Series of the mean off-diagonal correlation over time.
    Useful for detecting correlation regime changes.
    """
    n = returns.shape[1]
    if n < 2:
        return pd.DataFrame()

    # For each window, compute mean pairwise correlation
    result = []
    for i in range(window, len(returns)):
        chunk = returns.iloc[i - window : i]
        corr = chunk.corr()
        # Mean of upper triangle (excluding diagonal)
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        avg_corr = float(corr.values[mask].mean())
        result.append({"Date": returns.index[i], "AvgCorrelation": avg_corr})

    return pd.DataFrame(result).set_index("Date")


# ──────────────────────────────────────────────────────────────
# Tail metrics
# ──────────────────────────────────────────────────────────────


def tail_metrics(returns: pd.Series) -> dict[str, float]:
    """
    Comprehensive tail risk metrics for a return series.
    """
    r = returns.dropna()
    if r.empty:
        return {}

    var95, cvar95 = T.var_cvar(r, 0.95)
    var99, cvar99 = T.var_cvar(r, 0.99)
    gtp = T.gain_to_pain(r)

    # Skewness and kurtosis
    skew = float(r.skew())
    kurt = float(r.kurtosis())  # excess kurtosis

    # Sortino ratio
    rf_daily = 0.0
    downside = r[r < rf_daily]
    downside_vol = float(downside.std() * np.sqrt(252)) if len(downside) > 1 else np.nan
    ann_ret = T.annualize_return(r)
    sortino = (ann_ret - 0.0) / downside_vol if downside_vol > 0 else np.nan

    # Calmar ratio
    mdd = T.max_drawdown(pd.Series((1 + r).cumprod()))
    calmar = ann_ret / abs(mdd) if mdd != 0 and not np.isnan(mdd) else np.nan

    return {
        "VaR_95": var95,
        "CVaR_95": cvar95,
        "VaR_99": var99,
        "CVaR_99": cvar99,
        "Skewness": skew,
        "Excess_Kurtosis": kurt,
        "Sortino": sortino,
        "Calmar": calmar,
        "Gain_to_Pain": gtp if gtp is not None else np.nan,
        "Max_Drawdown": mdd,
        "Worst_Day": float(r.min()),
        "Best_Day": float(r.max()),
    }


# ──────────────────────────────────────────────────────────────
# Concentration metrics
# ──────────────────────────────────────────────────────────────


def herfindahl_index(weights: np.ndarray | pd.Series) -> float:
    """Herfindahl-Hirschman Index: sum of squared weights."""
    w = weights.values if isinstance(weights, pd.Series) else np.asarray(weights)
    return float(np.sum(w ** 2))


def effective_n_bets(weights: np.ndarray | pd.Series) -> float:
    """Effective number of bets = 1 / HHI."""
    hhi = herfindahl_index(weights)
    if hhi < 1e-12:
        return 0.0
    return 1.0 / hhi


def concentration_ratio(
    weights: np.ndarray | pd.Series,
    top_n: int = 3,
) -> float:
    """Sum of the top N absolute weights."""
    w = weights.values if isinstance(weights, pd.Series) else np.asarray(weights)
    sorted_abs = np.sort(np.abs(w))[::-1]
    return float(sorted_abs[:top_n].sum())
