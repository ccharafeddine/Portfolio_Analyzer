"""
Brinson–Fachler performance attribution.

Decomposes active return into Allocation, Selection, and Interaction effects.
Self-contained: no dependency on old performance_attribution.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def brinson_fachler(
    active_weights: pd.Series,
    benchmark_weights: pd.Series,
    active_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> pd.DataFrame:
    """
    Single-period Brinson–Fachler attribution.

    Parameters
    ----------
    active_weights : portfolio weights by asset (must sum to ~1)
    benchmark_weights : benchmark weights by asset (must sum to ~1)
    active_returns : period return per asset in the active portfolio
    benchmark_returns : period return per asset in the benchmark

    Returns
    -------
    DataFrame with columns: Asset, Allocation, Selection, Interaction, Total
    """
    # Align all series to common index
    assets = sorted(
        set(active_weights.index)
        | set(benchmark_weights.index)
    )

    w_p = active_weights.reindex(assets, fill_value=0.0)
    w_b = benchmark_weights.reindex(assets, fill_value=0.0)
    r_p = active_returns.reindex(assets, fill_value=0.0)
    r_b = benchmark_returns.reindex(assets, fill_value=0.0)

    # Overall benchmark return
    R_b = float((w_b * r_b).sum())

    # Brinson-Fachler decomposition
    allocation = (w_p - w_b) * (r_b - R_b)
    selection = w_b * (r_p - r_b)
    interaction = (w_p - w_b) * (r_p - r_b)
    total = allocation + selection + interaction

    return pd.DataFrame({
        "Asset": assets,
        "Allocation": allocation.values,
        "Selection": selection.values,
        "Interaction": interaction.values,
        "Total": total.values,
    })


def multi_period_attribution(
    portfolio_weights_ts: pd.DataFrame,
    benchmark_weights_ts: pd.DataFrame,
    portfolio_returns_ts: pd.DataFrame,
    benchmark_returns_ts: pd.DataFrame,
) -> pd.DataFrame:
    """
    Multi-period attribution by summing single-period BF effects.

    Each input DataFrame has DatetimeIndex rows and asset columns.
    Weights DataFrames have the weight at the start of each period.
    Returns DataFrames have the return during each period.

    Returns aggregated attribution across all periods.
    """
    # Get common dates
    dates = sorted(
        set(portfolio_weights_ts.index)
        & set(benchmark_weights_ts.index)
        & set(portfolio_returns_ts.index)
        & set(benchmark_returns_ts.index)
    )

    if not dates:
        return pd.DataFrame(columns=["Asset", "Allocation", "Selection", "Interaction", "Total"])

    # Sum attribution effects
    total_alloc = None

    for dt in dates:
        bf = brinson_fachler(
            active_weights=portfolio_weights_ts.loc[dt],
            benchmark_weights=benchmark_weights_ts.loc[dt],
            active_returns=portfolio_returns_ts.loc[dt],
            benchmark_returns=benchmark_returns_ts.loc[dt],
        )
        bf = bf.set_index("Asset")

        if total_alloc is None:
            total_alloc = bf[["Allocation", "Selection", "Interaction", "Total"]].copy()
        else:
            total_alloc = total_alloc.add(bf[["Allocation", "Selection", "Interaction", "Total"]], fill_value=0.0)

    if total_alloc is None:
        return pd.DataFrame(columns=["Asset", "Allocation", "Selection", "Interaction", "Total"])

    total_alloc = total_alloc.reset_index()
    return total_alloc


def simple_attribution_from_holdings(
    holdings: pd.DataFrame,
    monthly_returns: pd.DataFrame,
    benchmark: str,
    benchmark_weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Simplified attribution when we only have holdings + returns.

    Uses realized weights from holdings table and equal benchmark weights
    (or provided benchmark_weights) as the comparison.

    Parameters
    ----------
    holdings : DataFrame with 'Ticker' and 'RealizedWeight' columns
    monthly_returns : asset monthly returns
    benchmark : benchmark ticker name
    benchmark_weights : optional explicit benchmark weights; if None,
                        uses equal-weight across portfolio assets

    Returns
    -------
    Attribution DataFrame
    """
    tickers = holdings["Ticker"].tolist()
    w_col = "RealizedWeight" if "RealizedWeight" in holdings.columns else "TargetWeight"
    active_w = pd.Series(
        holdings[w_col].values,
        index=tickers,
    )

    # Benchmark weights: equal-weight if not provided
    if benchmark_weights is None:
        n = len(tickers)
        bench_w = pd.Series(1.0 / n, index=tickers)
    else:
        bench_w = pd.Series(benchmark_weights)

    # Use total-period returns
    available = [t for t in tickers if t in monthly_returns.columns]
    if not available:
        return pd.DataFrame(columns=["Asset", "Allocation", "Selection", "Interaction", "Total"])

    total_ret = (1 + monthly_returns[available]).prod() - 1

    # Active returns = asset returns
    # Benchmark returns = also asset returns (BF compares weighting decisions)
    return brinson_fachler(
        active_weights=active_w.reindex(available, fill_value=0.0),
        benchmark_weights=bench_w.reindex(available, fill_value=0.0),
        active_returns=total_ret,
        benchmark_returns=total_ret,
    )
