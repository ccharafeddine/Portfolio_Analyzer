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

from src.data import transforms as T
from src.analytics.risk import tail_metrics
from src.analytics.regression import capm_regression


def to_monthly_returns(values: pd.Series) -> pd.Series:
    """Month-end returns from a daily value series."""
    return values.resample("ME").last().pct_change().dropna()


def _align(a: pd.Series, b: pd.Series) -> tuple[pd.Series, pd.Series]:
    df = pd.concat([a.rename("p"), b.rename("b")], axis=1).dropna()
    return df["p"], df["b"]


def up_down_capture(port_ret: pd.Series, bench_ret: pd.Series) -> dict:
    """Compounded up/down capture ratios (Morningstar-style) on aligned returns."""
    p, b = _align(port_ret, bench_ret)
    out = {"up_capture": np.nan, "down_capture": np.nan, "capture_ratio": np.nan}
    up, down = b > 0, b < 0
    if up.any():
        bench_up = (1 + b[up]).prod() - 1
        port_up = (1 + p[up]).prod() - 1
        if abs(bench_up) > 1e-12:
            out["up_capture"] = port_up / bench_up
    if down.any():
        bench_dn = (1 + b[down]).prod() - 1
        port_dn = (1 + p[down]).prod() - 1
        if abs(bench_dn) > 1e-12:
            out["down_capture"] = port_dn / bench_dn
    if out["down_capture"] and not np.isnan(out["down_capture"]) and abs(out["down_capture"]) > 1e-9:
        out["capture_ratio"] = out["up_capture"] / out["down_capture"]
    return out


def batting_average(port_ret: pd.Series, bench_ret: pd.Series) -> float:
    """Fraction of periods the portfolio beat the benchmark."""
    p, b = _align(port_ret, bench_ret)
    if len(p) == 0:
        return np.nan
    return float((p > b).mean())


def tracking_error(port_ret: pd.Series, bench_ret: pd.Series, periods_per_year: int = 12) -> float:
    p, b = _align(port_ret, bench_ret)
    if len(p) < 2:
        return np.nan
    return float((p - b).std(ddof=1) * np.sqrt(periods_per_year))


def information_ratio(port_ret: pd.Series, bench_ret: pd.Series, periods_per_year: int = 12) -> float:
    p, b = _align(port_ret, bench_ret)
    te = tracking_error(p, b, periods_per_year)
    if not te or np.isnan(te) or te == 0:
        return np.nan
    active_ann = T.annualize_return(p, periods_per_year) - T.annualize_return(b, periods_per_year)
    return float(active_ann / te)


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
    a_tail = tail_metrics(a_daily)
    cap = up_down_capture(p_month, b_month)

    # Alpha/beta on aligned monthly returns.
    rf_m = (1 + rf) ** (1 / 12) - 1
    pa, ba = _align(p_month, b_month)
    alpha = beta = np.nan
    if len(pa) >= 6:
        reg = capm_regression(pa, ba, rf_periodic=rf_m)
        alpha, beta = reg.alpha * 12, float(reg.beta)

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
        "Beta": beta,
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
