"""Performance-measurement suite: benchmark-relative metrics analysts expect.

Builds on the existing engine: reuses ``transforms`` (annualization, Sharpe,
drawdown), ``risk.tail_metrics`` (Sortino/Calmar), and ``regression.capm_regression``
(alpha/beta). Adds the pieces that did not exist: up/down capture, batting
average, tracking error, information ratio, and a consolidated summary computed
over both the full available history and the common window (where every asset
exists), so the inception caveat is explicit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analytics import metrics as M
from src.data import transforms as T
from src.analytics.risk import tail_metrics
from src.analytics.regression import capm_regression


def to_monthly_returns(values: pd.Series) -> pd.Series:
    """Month-end returns from a daily value series."""
    return values.resample("ME").last().pct_change().dropna()


def _align(a: pd.Series, b: pd.Series) -> tuple[pd.Series, pd.Series]:
    df = pd.concat([a.rename("p"), b.rename("b")], axis=1).dropna()
    return df["p"], df["b"]


def up_down_capture(port_ret: pd.Series, bench_ret: pd.Series,
                    periods_per_year: int = 12) -> dict:
    """Morningstar-style up/down capture on aligned periodic returns.

    For the periods where the benchmark was up (down), capture is the ratio of
    the portfolio's *annualized geometric* return to the benchmark's over those
    same periods: ``((prod(1+p))**(ppy/n) - 1) / ((prod(1+b))**(ppy/n) - 1)``.
    (Comparing cumulative products instead, as before, compounds the gap over
    every up month and blows the ratio up on long samples — e.g. 317% vs 180%.)
    Matches ``empyrical.up_capture`` / ``down_capture``.
    """
    p, b = _align(port_ret, bench_ret)
    out = {"up_capture": np.nan, "down_capture": np.nan, "capture_ratio": np.nan}

    def _geo(x: pd.Series) -> float:
        return float((1 + x).prod() ** (periods_per_year / len(x)) - 1.0)

    for key, mask in (("up_capture", b > 0), ("down_capture", b < 0)):
        if mask.any():
            bench = _geo(b[mask])
            if abs(bench) > 1e-12:
                out[key] = _geo(p[mask]) / bench
    dn = out["down_capture"]
    if not np.isnan(dn) and abs(dn) > 1e-9:
        out["capture_ratio"] = out["up_capture"] / dn
    return out


def batting_average(port_ret: pd.Series, bench_ret: pd.Series) -> float:
    """Fraction of periods the portfolio beat the benchmark."""
    p, b = _align(port_ret, bench_ret)
    if len(p) == 0:
        return np.nan
    return float((p > b).mean())


def tracking_error(port_ret: pd.Series, bench_ret: pd.Series, periods_per_year: int = 12) -> float:
    """Annualized std of active returns (see ``src.analytics.metrics``)."""
    p, b = _align(port_ret, bench_ret)
    return M.tracking_error(p, b, periods_per_year)


def information_ratio(port_ret: pd.Series, bench_ret: pd.Series, periods_per_year: int = 12) -> float:
    """Annualized mean active return / tracking error (arithmetic convention)."""
    p, b = _align(port_ret, bench_ret)
    return M.information_ratio(p, b, periods_per_year)


def blended_benchmark(prices: pd.DataFrame, blend: dict[str, float], capital: float) -> pd.Series:
    """Value series of a fixed-weight blend of benchmark tickers (e.g. 60/40)."""
    cols = [t for t in blend if t in prices.columns]
    if not cols:
        raise ValueError("No blend tickers in price data.")
    sub = prices[cols].dropna(how="any")
    w = np.array([blend[c] for c in cols], dtype=float)
    w = w / w.sum()
    units = (capital * w) / sub.iloc[0].values
    value = (sub.values * units).sum(axis=1)
    return pd.Series(value, index=sub.index, name="Blended")


def _window_metrics(active_val: pd.Series, passive_val: pd.Series, rf: float) -> dict:
    """All metrics for one time window (both series already sliced to it)."""
    a_daily = active_val.pct_change().dropna()
    p_month = to_monthly_returns(active_val)
    b_month = to_monthly_returns(passive_val)
    a_tail = tail_metrics(a_daily, rf_annual=rf, periods_per_year=M.DAILY)
    cap = up_down_capture(p_month, b_month)

    # Alpha/beta (with standard errors) on aligned monthly excess returns.
    pa, ba = _align(p_month, b_month)
    stats = M.capm(pa, ba, rf_annual=rf, periods_per_year=M.MONTHLY) if len(pa) >= 6 else None
    alpha = stats.alpha_annual if stats else np.nan
    beta = stats.beta if stats else np.nan

    corr = float(pa.corr(ba)) if len(pa) >= 2 else np.nan
    return {
        "Ann. Return": T.annualize_return(a_daily),
        "Ann. Volatility": T.annualize_vol(a_daily),
        "Sharpe": T.sharpe_ratio(a_daily, rf_annual=rf),
        "Sortino": a_tail.get("Sortino", np.nan),
        "Calmar": a_tail.get("Calmar", np.nan),
        "Max Drawdown": T.max_drawdown(active_val),
        "Tracking Error": tracking_error(p_month, b_month),
        "Information Ratio": information_ratio(p_month, b_month),
        "Up Capture": cap["up_capture"],
        "Down Capture": cap["down_capture"],
        "Batting Avg": batting_average(p_month, b_month),
        "Correlation": corr,
        "Alpha (ann.)": alpha,
        "Alpha t-stat": stats.t_alpha if stats else np.nan,
        "Beta": beta,
        "Beta SE": stats.beta_se if stats else np.nan,
    }


def performance_summary(
    active_val: pd.Series,
    passive_val: pd.Series,
    rf: float,
    effective_start=None,
) -> pd.DataFrame:
    """Consolidated benchmark-relative metrics over the full history and, when the
    portfolio's composition changed mid-window, the common window too."""
    full = _window_metrics(active_val, passive_val, rf)
    rows = {m: {"Full History": v} for m, v in full.items()}

    if effective_start is not None:
        es = pd.Timestamp(effective_start)
        a_c = active_val.loc[active_val.index >= es]
        p_c = passive_val.loc[passive_val.index >= es]
        # Only bother if the common window is materially shorter than the full one.
        if len(a_c) > 20 and a_c.index[0] > active_val.index[0]:
            common = _window_metrics(a_c, p_c, rf)
            for m, v in common.items():
                rows[m]["Common Window"] = v

    df = pd.DataFrame(rows).T
    df.index.name = "Metric"
    return df.reset_index()


def rolling_alpha_beta(
    active_val: pd.Series, passive_val: pd.Series, rf: float, window: int = 12
) -> pd.DataFrame:
    """Rolling annualized alpha and beta from monthly excess returns."""
    p, b = _align(to_monthly_returns(active_val), to_monthly_returns(passive_val))
    if len(p) < window:
        return pd.DataFrame(columns=["Alpha", "Beta"])
    rf_m = (1 + rf) ** (1 / 12) - 1
    pe, be = p - rf_m, b - rf_m
    cov = pe.rolling(window).cov(be)
    var = be.rolling(window).var()
    beta = cov / var
    alpha = (pe.rolling(window).mean() - beta * be.rolling(window).mean()) * 12
    return pd.DataFrame({"Alpha": alpha, "Beta": beta}).dropna()


def capture_metrics(active_val: pd.Series, passive_val: pd.Series) -> dict:
    """Up/down capture + batting average on monthly returns (for the stat grid)."""
    p, b = to_monthly_returns(active_val), to_monthly_returns(passive_val)
    out = up_down_capture(p, b)
    out["batting_avg"] = batting_average(p, b)
    return out
