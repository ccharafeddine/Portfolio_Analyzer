"""Tests for Black-Litterman posterior returns and their effect on the ORP."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from src import pipeline as pl
from src.analytics.optimization import black_litterman_posterior
from src.config.models import BLConfig, BLView, PortfolioConfig


COV = np.diag([0.04, 0.04, 0.04])
W = np.array([1 / 3, 1 / 3, 1 / 3])
RF, DELTA = 0.03, 3.0


def _pi():
    return RF + DELTA * (COV @ W)


def test_no_views_returns_equilibrium():
    post = black_litterman_posterior(COV, W, [], RF, tau=0.05, delta=DELTA)
    assert np.allclose(post, _pi())


def test_absolute_view_raises_that_asset():
    pi = _pi()
    view = (np.array([1.0, 0.0, 0.0]), 0.20, 0.35)  # bullish on asset 0, high conf
    post = black_litterman_posterior(COV, W, [view], RF, tau=0.05, delta=DELTA)
    assert post[0] > pi[0]
    assert post[0] > post[1]


def test_relative_view_widens_spread():
    view = (np.array([1.0, -1.0, 0.0]), 0.10, 1.0)  # asset0 beats asset1 by 10%
    post = black_litterman_posterior(COV, W, [view], RF, tau=0.05, delta=DELTA)
    assert post[0] - post[1] > 0.0


# ── Pipeline effect ──

def _prices(cols, n=400, seed=7):
    idx = pd.bdate_range("2021-01-04", periods=n)
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {c: 100 * np.cumprod(1 + rng.normal(0.0004, 0.011, n)) for c in cols}, index=idx
    )


def _orp_weights(monkeypatch, tmp_path, bl):
    monkeypatch.setattr(pl, "fetch_prices",
                        lambda tickers, start, end, **kw: _prices(sorted(set(tickers))))
    cfg = PortfolioConfig(
        tickers=["AAPL", "MSFT", "NVDA"],
        weights={"AAPL": 0.34, "MSFT": 0.33, "NVDA": 0.33},
        benchmark="SPY", black_litterman=bl,
        start_date=date(2021, 1, 4), end_date=date(2022, 8, 1),
    )
    p = pl.AnalysisPipeline(cfg, output_dir=str(tmp_path))
    p._fetch_data()
    p._build_active()
    p._optimize()
    return p.results.orp_optimization.weights


def test_bullish_view_tilts_orp(monkeypatch, tmp_path):
    base = _orp_weights(monkeypatch, tmp_path, BLConfig(enabled=False))
    # Baseline max-Sharpe corners into the best in-sample asset; a strong bullish view
    # on an otherwise-ignored asset should pull weight toward it.
    tilted = _orp_weights(
        monkeypatch, tmp_path,
        BLConfig(enabled=True, views=[
            BLView(type="absolute", asset="AAPL", q=0.80, confidence="high")
        ]),
    )
    assert tilted["AAPL"] > base["AAPL"] + 1e-3
