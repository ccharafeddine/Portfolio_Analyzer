"""Unit tests for the multi-portfolio comparison engine (src/analytics/comparison.py).

Prices are mocked (no network); sector lookups are disabled.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.analytics import comparison as cmp
from src.config.models import PortfolioConfig


def _cfg(tickers, weights, benchmark="SPY"):
    return PortfolioConfig(
        tickers=tickers, weights=weights, benchmark=benchmark,
        start_date=date(2022, 1, 3), end_date=date(2023, 1, 3),
    )


def _install_prices(monkeypatch, n=260, seed=1):
    idx = pd.bdate_range("2022-01-03", periods=n)

    def fake_fetch(tickers, start, end, cache_dir=None, force_refresh=False):
        rng = np.random.default_rng(seed)
        data = {}
        for j, t in enumerate(tickers):
            r = rng.normal(0.0005 + 0.0001 * j, 0.011, n)
            data[t] = 100.0 * np.cumprod(1.0 + r)
        return pd.DataFrame(data, index=idx)

    monkeypatch.setattr(cmp.fetcher, "fetch_prices", fake_fetch)
    monkeypatch.setattr(cmp, "yf", None)  # skip sector network lookups


def test_compare_produces_metrics(monkeypatch):
    _install_prices(monkeypatch)
    named = [
        ("Growth", _cfg(["AAPL", "MSFT"], {"AAPL": 0.5, "MSFT": 0.5})),
        ("Barbell", _cfg(["AAPL", "TLT"], {"AAPL": 0.7, "TLT": 0.3})),
    ]
    results = cmp.compare_portfolios(named)
    assert [r.label for r in results] == ["Growth", "Barbell"]
    for r in results:
        assert r.error is None
        assert np.isfinite(r.ann_return) and np.isfinite(r.ann_vol)
        assert np.isfinite(r.sharpe) and np.isfinite(r.beta)
        assert r.max_dd <= 0.0
        assert r.sector_weights is None  # yf disabled


def test_concentration(monkeypatch):
    _install_prices(monkeypatch)
    r = cmp.compare_portfolios([("Equal", _cfg(["A", "B"], {"A": 0.5, "B": 0.5}))])[0]
    assert r.hhi == pytest.approx(0.5)
    assert r.effective_n == pytest.approx(2.0)
    assert r.top3 == pytest.approx(1.0)


def test_holdings_overlap_sets(monkeypatch):
    _install_prices(monkeypatch)
    named = [
        ("P1", _cfg(["AAPL", "MSFT", "NVDA"], {"AAPL": 0.34, "MSFT": 0.33, "NVDA": 0.33})),
        ("P2", _cfg(["MSFT", "NVDA", "TLT"], {"MSFT": 0.34, "NVDA": 0.33, "TLT": 0.33})),
    ]
    results = cmp.compare_portfolios(named)
    s1, s2 = set(results[0].tickers), set(results[1].tickers)
    assert s1 & s2 == {"MSFT", "NVDA"}
    assert s1 | s2 == {"AAPL", "MSFT", "NVDA", "TLT"}


def test_returns_correlation(monkeypatch):
    _install_prices(monkeypatch)
    named = [
        ("P1", _cfg(["AAPL", "MSFT"], {"AAPL": 0.5, "MSFT": 0.5})),
        ("P2", _cfg(["AAPL", "NVDA"], {"AAPL": 0.5, "NVDA": 0.5})),
    ]
    corr = cmp.returns_correlation(cmp.compare_portfolios(named))
    assert list(corr.columns) == ["P1", "P2"]
    assert corr.loc["P1", "P1"] == pytest.approx(1.0, abs=1e-9)
    assert -1.0 <= corr.loc["P1", "P2"] <= 1.0


def test_bad_portfolio_is_isolated(monkeypatch):
    # Prices only contain the good portfolio's tickers; the bad one's are absent.
    idx = pd.bdate_range("2022-01-03", periods=120)

    def fake_fetch(tickers, start, end, cache_dir=None, force_refresh=False):
        cols = [t for t in tickers if t in ("AAPL", "MSFT", "SPY")]
        rng = np.random.default_rng(0)
        return pd.DataFrame({c: 100 * np.cumprod(1 + rng.normal(0.0004, 0.01, 120))
                             for c in cols}, index=idx)

    monkeypatch.setattr(cmp.fetcher, "fetch_prices", fake_fetch)
    monkeypatch.setattr(cmp, "yf", None)
    named = [
        ("Good", _cfg(["AAPL", "MSFT"], {"AAPL": 0.5, "MSFT": 0.5})),
        ("Bad", _cfg(["ZZZZ"], {"ZZZZ": 1.0})),
    ]
    results = cmp.compare_portfolios(named)
    assert results[0].error is None
    assert results[1].error is not None  # build_backtest had no usable prices
