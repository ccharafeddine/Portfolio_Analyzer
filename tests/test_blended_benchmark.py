"""Tests for blended (multi-asset) benchmarks in the pipeline."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from src import pipeline as pl
from src.config.models import PortfolioConfig


def _prices(cols, n=260, seed=0):
    idx = pd.bdate_range("2022-01-03", periods=n)
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {c: 100 * np.cumprod(1 + rng.normal(0.0004, 0.01, n)) for c in cols}, index=idx
    )


def _install(monkeypatch):
    def fake(tickers, start, end, **kw):
        return _prices(sorted(set(tickers)))

    monkeypatch.setattr(pl, "fetch_prices", fake)


def test_blended_benchmark_injected(monkeypatch, tmp_path):
    _install(monkeypatch)
    cfg = PortfolioConfig(
        tickers=["AAPL", "MSFT"], weights={"AAPL": 0.5, "MSFT": 0.5},
        benchmark="60/40", benchmark_weights={"SPY": 0.6, "AGG": 0.4},
        start_date=date(2022, 1, 3), end_date=date(2022, 12, 30),
    )
    p = pl.AnalysisPipeline(cfg, output_dir=str(tmp_path))
    p._fetch_data()
    assert p._bench_label == "60/40"
    assert "60/40" in p.results.prices.columns
    assert "60/40" in p.results.monthly_returns.columns

    p._build_active()
    p._build_passive()
    assert p.results.passive is not None
    # A blended benchmark is a value series seeded at `capital`.
    assert abs(p.results.passive.values.iloc[0] - cfg.capital) < cfg.capital * 1e-3


def test_blend_label_collision_avoided(monkeypatch, tmp_path):
    _install(monkeypatch)
    cfg = PortfolioConfig(
        tickers=["AAPL", "MSFT"], weights={"AAPL": 0.5, "MSFT": 0.5},
        benchmark="SPY", benchmark_weights={"SPY": 0.6, "AGG": 0.4},
        start_date=date(2022, 1, 3), end_date=date(2022, 12, 30),
    )
    p = pl.AnalysisPipeline(cfg, output_dir=str(tmp_path))
    p._fetch_data()
    # Real "SPY" column must be preserved; the blend uses a distinct label.
    assert p._bench_label == "SPY (blend)"
    assert "SPY" in p.results.prices.columns
    assert "SPY (blend)" in p.results.prices.columns


def test_single_benchmark_unchanged(monkeypatch, tmp_path):
    _install(monkeypatch)
    cfg = PortfolioConfig(
        tickers=["AAPL", "MSFT"], weights={"AAPL": 0.5, "MSFT": 0.5},
        benchmark="SPY",
        start_date=date(2022, 1, 3), end_date=date(2022, 12, 30),
    )
    p = pl.AnalysisPipeline(cfg, output_dir=str(tmp_path))
    p._fetch_data()
    assert p._bench_label == "SPY"
    assert p.config.benchmark_weights == {}
