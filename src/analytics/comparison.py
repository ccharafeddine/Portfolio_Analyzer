"""Lightweight multi-portfolio comparison engine.

Given several ``(label, PortfolioConfig)`` pairs, compute just what a side-by-side
comparison needs — prices → active value series (via the backtest engine) → risk/return
metrics, portfolio-level alpha/beta, concentration, and (best-effort) sector weights.
Deliberately skips optimization / tax / planning / Monte Carlo and writes no files, so
comparing a handful of portfolios stays fast. Qt-free and unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

from src.analytics import risk
from src.analytics.backtest import build_backtest
from src.data import fetcher
from src.data import transforms as T

try:
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None

ProgressCallback = Callable[[str, float], None]


@dataclass
class ComparisonResult:
    label: str
    values: pd.Series                       # daily portfolio value
    daily_returns: pd.Series
    ann_return: float = float("nan")
    ann_vol: float = float("nan")
    sharpe: float = float("nan")
    sortino: float = float("nan")
    max_dd: float = float("nan")
    var95: float = float("nan")
    cvar95: float = float("nan")
    alpha: float = float("nan")             # annualized
    beta: float = float("nan")
    weights: dict = field(default_factory=dict)
    hhi: float = float("nan")
    effective_n: float = float("nan")
    top3: float = float("nan")
    sector_weights: Optional[dict] = None   # sector -> weight
    tickers: list = field(default_factory=list)
    effective_start: Optional[str] = None
    error: Optional[str] = None


def _alpha_beta(port_ret: pd.Series, bench_ret: pd.Series, rf_annual: float):
    """Annualized alpha and beta of the portfolio vs the benchmark (daily OLS)."""
    df = pd.concat([port_ret, bench_ret], axis=1).dropna()
    if len(df) < 30:
        return float("nan"), float("nan")
    rf_d = rf_annual / 252.0
    y = df.iloc[:, 0] - rf_d
    x = df.iloc[:, 1] - rf_d
    var_x = float(x.var(ddof=1))
    if var_x <= 0:
        return float("nan"), float("nan")
    beta = float(x.cov(y) / var_x)
    alpha_daily = float(y.mean() - beta * x.mean())
    return alpha_daily * 252.0, beta


def _sector_weights(weights: dict, sector_cache: dict) -> Optional[dict]:
    """Group portfolio weights by sector, fetching any missing tickers' sector once
    into ``sector_cache`` (shared across portfolios). Best-effort; None on failure."""
    if yf is None:
        return None
    try:
        out: dict[str, float] = {}
        for tk, w in weights.items():
            if tk not in sector_cache:
                try:
                    sector_cache[tk] = (yf.Ticker(tk).info or {}).get("sector") or "Other"
                except Exception:
                    sector_cache[tk] = "Other"
            out[sector_cache[tk]] = out.get(sector_cache[tk], 0.0) + float(w)
        return out or None
    except Exception:
        return None


def _one(label: str, cfg, sector_cache: dict, include_sectors: bool) -> ComparisonResult:
    tickers = list(cfg.tickers)
    blend = getattr(cfg, "benchmark_weights", None) or {}
    bench_tickers = list(blend) if blend else [cfg.benchmark]
    prices = fetcher.fetch_prices(tickers + bench_tickers, cfg.start_date, cfg.end_date)
    bt = build_backtest(
        prices, cfg.weights, cfg.capital, cfg.risk_free_rate,
        inception_mode=cfg.backtest.inception_mode,
        rebalance_frequency=cfg.backtest.rebalance_frequency,
        cost_bps=cfg.backtest.transaction_cost_bps,
    )
    values = bt.values
    rets = values.pct_change().dropna()

    bench_ret = pd.Series(dtype=float)
    if blend:
        try:
            from src.analytics.performance import blended_benchmark
            bench_ret = blended_benchmark(prices, blend, cfg.capital).pct_change().dropna()
        except Exception:
            pass
    elif cfg.benchmark in prices.columns:
        bench_ret = prices[cfg.benchmark].pct_change().dropna()

    tail = risk.tail_metrics(rets)
    var95, cvar95 = T.var_cvar(rets, 0.95)
    alpha, beta = _alpha_beta(rets, bench_ret, cfg.risk_free_rate)
    w = pd.Series(cfg.weights, dtype=float)

    return ComparisonResult(
        label=label,
        values=values,
        daily_returns=rets,
        ann_return=T.annualize_return(rets),
        ann_vol=T.annualize_vol(rets),
        sharpe=T.sharpe_ratio(rets, rf_annual=cfg.risk_free_rate),
        sortino=tail.get("Sortino", float("nan")),
        max_dd=T.max_drawdown(values),
        var95=var95,
        cvar95=cvar95,
        alpha=alpha,
        beta=beta,
        weights=dict(cfg.weights),
        hhi=risk.herfindahl_index(w),
        effective_n=risk.effective_n_bets(w),
        top3=risk.concentration_ratio(w, 3),
        sector_weights=_sector_weights(cfg.weights, sector_cache) if include_sectors else None,
        tickers=tickers,
        effective_start=(bt.effective_start.date().isoformat() if bt.effective_start is not None else None),
    )


def compare_portfolios(
    named_configs, progress: Optional[ProgressCallback] = None,
    include_sectors: bool = True,
) -> list[ComparisonResult]:
    """Run the fast comparison for each ``(label, PortfolioConfig)``.

    Errors on one portfolio yield a sparse result (with ``error`` set) rather than
    aborting the whole comparison.
    """
    results: list[ComparisonResult] = []
    sector_cache: dict = {}
    n = max(1, len(named_configs))
    for i, (label, cfg) in enumerate(named_configs):
        if progress:
            progress(f"Analyzing {label}…", i / n)
        try:
            results.append(_one(label, cfg, sector_cache, include_sectors))
        except Exception as e:
            results.append(ComparisonResult(
                label=label, values=pd.Series(dtype=float),
                daily_returns=pd.Series(dtype=float),
                weights=dict(getattr(cfg, "weights", {})),
                tickers=list(getattr(cfg, "tickers", [])), error=str(e),
            ))
    if progress:
        progress("Done", 1.0)
    return results


def returns_correlation(results: list[ComparisonResult]) -> pd.DataFrame:
    """Correlation matrix of the portfolios' daily-return streams (aligned by date)."""
    cols = {r.label: r.daily_returns for r in results
            if r.daily_returns is not None and not r.daily_returns.empty}
    if len(cols) < 2:
        return pd.DataFrame()
    return pd.DataFrame(cols).dropna().corr()
