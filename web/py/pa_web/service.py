"""Run the shared analytics pipeline for the web demo and turn the results into JSON.

Runs in the browser under Pyodide (see web/static/engine.worker.js) and in CPython
for tests. Only the steps the demo displays are run (no optimizer, Monte Carlo,
retirement, reports or disk outputs). Nothing here imports Qt.
"""

from __future__ import annotations

import math
import tempfile
from typing import Any

import numpy as np
import pandas as pd

from src.analytics import metrics as M
from src.analytics.summary import holdings_summary, period_returns
from src.config.models import PortfolioConfig
from src.pipeline import AnalysisPipeline, AnalysisResults

from .schemas import AnalyzeRequest

MAX_POINTS = 1300  # beyond this, chart series are resampled to weekly


class DemoPipeline(AnalysisPipeline):
    """The desktop pipeline restricted to the steps the web demo shows."""

    def run(self, progress=None) -> AnalysisResults:  # noqa: D401 - same contract
        steps = [
            ("Fetching price data", self._fetch_data),
            ("Building active portfolio", self._build_active),
            ("Building passive portfolio", self._build_passive),
            ("Running CAPM regressions", self._run_capm),
            ("Measuring performance", self._measure_performance),
            ("Computing risk metrics", self._compute_risk),
            ("VaR models & drawdowns", self._compute_tail_risk),
            ("Running stress tests", self._run_stress_tests),
            ("Running attribution", self._run_attribution),
        ]
        import time
        for label, fn in steps:
            t0 = time.perf_counter()
            try:
                fn()
            except Exception as e:  # same non-fatal contract as the desktop run
                self._warn(f"Step '{label}' failed", e)
            self.results.timings[label] = round(time.perf_counter() - t0, 3)
        return self.results


def build_config(req: AnalyzeRequest) -> PortfolioConfig:
    w = req.normalized_weights()
    return PortfolioConfig(
        tickers=list(w), weights=w, benchmark=req.benchmark,
        start_date=req.start_date, end_date=req.end_date, capital=req.capital,
        risk_free_source="tbill", risk_free_rate=0.04,
        include_orp=False, include_complete=False,
    )


class AnalysisError(Exception):
    """The run produced nothing usable (e.g. no price data for the tickers)."""


def run_analysis(req: AnalyzeRequest) -> dict[str, Any]:
    cfg = build_config(req)
    with tempfile.TemporaryDirectory(prefix="pa_web_") as out:  # pipeline mkdirs it
        res = DemoPipeline(cfg, output_dir=out).run()
    if res.active is None or res.passive is None:
        msg = "; ".join(res.warnings[:3]) or "No price data returned"
        raise AnalysisError(f"Could not build the portfolio: {msg}")
    return serialize(res)


# ── serialization ──────────────────────────────────────────────────────────


def _f(v) -> float | None:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) or math.isinf(v) else round(v, 6)


def _series(s: pd.Series) -> dict[str, list]:
    s = s.dropna()
    if len(s) > MAX_POINTS:
        s = s.resample("W-FRI").last().dropna()
    return {"x": [d.strftime("%Y-%m-%d") for d in s.index], "y": [_f(v) for v in s.values]}


def _records(df: pd.DataFrame | None) -> list[dict]:
    if df is None or df.empty:
        return []
    df = df.loc[:, ~df.columns.duplicated()]
    out = []
    for row in df.to_dict(orient="records"):
        rec = {}
        for k, v in row.items():
            if isinstance(v, (pd.Timestamp,)) or hasattr(v, "strftime"):
                rec[str(k)] = None if pd.isna(v) else v.strftime("%Y-%m-%d")
            elif isinstance(v, (int, float, np.floating, np.integer)) and not isinstance(v, bool):
                rec[str(k)] = _f(v)
            else:
                rec[str(k)] = None if v is None or (isinstance(v, float) and math.isnan(v)) else str(v)
        out.append(rec)
    return out


def serialize(res: AnalysisResults) -> dict[str, Any]:
    cfg = res.config
    a, p = res.active, res.passive
    bench = res.config.benchmark
    summ = {}
    if res.performance_summary is not None and not res.performance_summary.empty:
        summ = dict(zip(res.performance_summary["Metric"], res.performance_summary["Full History"]))
    tr = res.tail_risk or {}
    nav = float(a.values.iloc[-1])
    day = float(a.values.iloc[-1] - a.values.iloc[-2]) if len(a.values) > 1 else 0.0
    kpis = {
        "nav": _f(nav), "nav_day": _f(day),
        "cagr": _f(a.ann_return), "bench_cagr": _f(p.ann_return),
        "excess": _f(a.ann_return - p.ann_return),
        "alpha": _f(summ.get("Alpha (ann.)")), "alpha_t": _f(summ.get("Alpha t-stat")),
        "vol": _f(a.ann_vol), "bench_vol": _f(p.ann_vol),
        "sharpe": _f(a.sharpe), "sortino": _f(tr.get("Sortino")),
        "max_dd": _f(a.max_dd), "bench_max_dd": _f(p.max_dd),
        "beta": _f(summ.get("Beta")), "beta_se": _f(summ.get("Beta SE")),
        "tracking_error": _f(summ.get("Tracking Error")),
        "info_ratio": _f(summ.get("Information Ratio")),
    }
    hold = holdings_summary(res.holdings, res.prices, benchmark=bench)
    periods = period_returns(a.values, p.values)
    sa_meta = res.sector_attribution_meta or {}
    rf = res.risk_free
    prices = res.prices
    return {
        "meta": {
            "tickers": list(cfg.tickers), "benchmark": bench,
            "start": str(a.values.index[0].date()), "end": str(a.values.index[-1].date()),
            "capital": cfg.capital, "trading_days": int(len(a.values)),
            "risk_free": {"rate": _f(rf.rate), "source": rf.source} if rf else None,
            "data_source": "Yahoo Finance via price proxy",
            "offline": bool(getattr(prices, "attrs", {}).get("offline")) if prices is not None else False,
        },
        "kpis": kpis,
        "growth": {"portfolio": _series(a.values), "benchmark": _series(p.values)},
        "drawdown": {"portfolio": _series(M.drawdown_series(a.values)),
                     "benchmark": _series(M.drawdown_series(p.values))},
        "rolling_sharpe": _series(M.rolling_sharpe(a.daily_returns, 252, cfg.risk_free_rate)),
        "periods": _records(periods),
        "holdings": _records(hold),
        "performance": _records(res.performance_summary),
        "capture": {k: _f(v) for k, v in (res.capture_metrics or {}).items()},
        "risk": {
            "var": _records(res.var_table),
            "var_backtest": _records(res.var_backtest),
            "drawdowns": _records(res.drawdown_table),
            "stress": _records(res.stress_df),
            "tail": {k: _f(v) for k, v in tr.items()},
        },
        "attribution": {
            "sectors": _records(res.sector_attribution),
            "source": sa_meta.get("source"),
            "port_return": _f(sa_meta.get("port_return")),
            "bench_return": _f(sa_meta.get("bench_return")),
            "active_return": _f(sa_meta.get("active_return")),
            "sector_map": sa_meta.get("sectors") or {},
        },
        "warnings": list(res.warnings),
        "timings": res.timings,
        "missing": sorted(set(getattr(res.prices, "attrs", {}).get("missing", []) or [])) if res.prices is not None else [],
    }
