import os
from typing import Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------


def _to_returns_from_values(values: pd.Series) -> pd.Series:
    """
    Convert a portfolio value series into simple returns.

    Assumes 'values' is a time series of portfolio values (e.g., dollars).
    """
    v = values.dropna().astype(float)
    v = v.sort_index()
    rets = v.pct_change().dropna()
    return rets


def _annualize_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Annualized geometric mean return from period returns.
    """
    if returns.empty:
        return np.nan
    gross = (1.0 + returns).prod()
    n = len(returns)
    if n == 0 or gross <= 0:
        return np.nan
    return float(gross ** (periods_per_year / n) - 1.0)


def _annualize_vol(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Annualized volatility.
    """
    if returns.empty:
        return np.nan
    return float(returns.std(ddof=1) * np.sqrt(periods_per_year))


def _sharpe_ratio(
    returns: pd.Series,
    rf_annual: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Sharpe ratio using an annual risk-free rate.
    """
    if returns.empty:
        return np.nan
    rf_per = rf_annual / periods_per_year
    excess = returns - rf_per
    vol = excess.std(ddof=1)
    if vol == 0 or np.isnan(vol):
        return np.nan
    mean_excess = excess.mean()
    return float(np.sqrt(periods_per_year) * mean_excess / vol)


def _max_drawdown(values: pd.Series) -> float:
    """
    Maximum drawdown from a portfolio value series (in decimal, e.g. -0.25).
    """
    v = values.dropna().astype(float).sort_index()
    if v.empty:
        return np.nan
    cum_max = v.cummax()
    drawdown = v / cum_max - 1.0
    return float(drawdown.min())


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------


def compute_active_stats_table(
    active_values: pd.Series,
    benchmark_values: pd.Series,
    rf: float,
    outdir: str,
) -> pd.DataFrame:
    """
    Compute summary statistics for the active portfolio vs benchmark and
    save them to active_stats_table.csv.

    Parameters
    ----------
    active_values : Series
        Time series of active portfolio values (e.g., dollars).
    benchmark_values : Series
        Time series of benchmark portfolio values.
    rf : float
        Annual risk-free rate (e.g., 0.04 for 4%).
    outdir : str
        Output directory (typically 'outputs').

    Returns
    -------
    DataFrame
        Summary statistics table.
    """
    os.makedirs(outdir, exist_ok=True)

    # Ensure aligned index for fair comparison
    df_vals = pd.concat(
        [active_values.rename("Active"), benchmark_values.rename("Benchmark")],
        axis=1,
    ).dropna()

    active_vals = df_vals["Active"]
    bench_vals = df_vals["Benchmark"]

    # Period returns from values
    active_rets = _to_returns_from_values(active_vals)
    bench_rets = _to_returns_from_values(bench_vals)

    # Align returns
    rets = pd.concat(
        [active_rets.rename("Active"), bench_rets.rename("Benchmark")],
        axis=1,
    ).dropna()
    active_rets = rets["Active"]
    bench_rets = rets["Benchmark"]

    # Basic stats
    act_ann_ret = _annualize_return(active_rets)
    act_ann_vol = _annualize_vol(active_rets)
    act_sharpe = _sharpe_ratio(active_rets, rf_annual=rf)

    bench_ann_ret = _annualize_return(bench_rets)
    bench_ann_vol = _annualize_vol(bench_rets)
    bench_sharpe = _sharpe_ratio(bench_rets, rf_annual=rf)

    # Active vs benchmark
    active_excess_ann = act_ann_ret - bench_ann_ret
    active_diff_rets = active_rets - bench_rets
    tracking_error = _annualize_vol(active_diff_rets)
    if tracking_error and not np.isnan(tracking_error) and tracking_error != 0:
        information_ratio = float(active_excess_ann / tracking_error)
    else:
        information_ratio = np.nan

    corr = float(active_rets.corr(bench_rets)) if len(rets) > 1 else np.nan

    # Drawdowns from value series
    act_mdd = _max_drawdown(active_vals)
    bench_mdd = _max_drawdown(bench_vals)

    # Build table
    data = {
        "Portfolio": ["Active", "Benchmark", "Active vs Benchmark"],
        "Ann. Return": [act_ann_ret, bench_ann_ret, active_excess_ann],
        "Ann. Volatility": [act_ann_vol, bench_ann_vol, tracking_error],
        "Sharpe": [act_sharpe, bench_sharpe, np.nan],
        "Max Drawdown": [act_mdd, bench_mdd, np.nan],
        "Correlation (Active, Benchmark)": [corr, corr, corr],
        "Information Ratio": [np.nan, np.nan, information_ratio],
    }
    stats_df = pd.DataFrame(data)

    # Save
    out_path = os.path.join(outdir, "active_stats_table.csv")
    stats_df.to_csv(out_path, index=False)

    return stats_df


def compute_active_stats(
    active_values: pd.Series,
    benchmark_values: pd.Series,
    rf: float,
    outdir: str,
) -> Tuple[pd.DataFrame, dict]:
    """
    Convenience wrapper around compute_active_stats_table.

    Returns both the stats table and a small dict of key metrics that other
    modules (or the Streamlit app) could use if needed.
    """
    stats_df = compute_active_stats_table(
        active_values=active_values,
        benchmark_values=benchmark_values,
        rf=rf,
        outdir=outdir,
    )

    # Pull a few headline numbers into a dict (optional, but handy)
    key_metrics = {}

    try:
        active_row = stats_df.loc[stats_df["Portfolio"] == "Active"].iloc[0]
        bench_row = stats_df.loc[stats_df["Portfolio"] == "Benchmark"].iloc[0]
        rel_row = stats_df.loc[
            stats_df["Portfolio"] == "Active vs Benchmark"
        ].iloc[0]

        key_metrics = {
            "active_ann_return": float(active_row["Ann. Return"]),
            "active_ann_vol": float(active_row["Ann. Volatility"]),
            "active_sharpe": float(active_row["Sharpe"]),
            "benchmark_ann_return": float(bench_row["Ann. Return"]),
            "benchmark_ann_vol": float(bench_row["Ann. Volatility"]),
            "benchmark_sharpe": float(bench_row["Sharpe"]),
            "active_excess_ann": float(rel_row["Ann. Return"]),
            "tracking_error": float(rel_row["Ann. Volatility"]),
            "information_ratio": float(rel_row["Information Ratio"]),
            "corr_active_benchmark": float(
                active_row["Correlation (Active, Benchmark)"]
            ),
        }
    except Exception:
        # If anything goes wrong, just return an empty dict
        key_metrics = {}

    return stats_df, key_metrics
