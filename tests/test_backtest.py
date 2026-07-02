"""Tests for the inception-aware backtest engine (src/analytics/backtest.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analytics.backtest import build_backtest


def _synthetic_prices():
    """3 assets; C is a 'late IPO' (no data until day 300)."""
    idx = pd.bdate_range("2021-01-01", periods=500)
    rng = np.random.default_rng(0)

    def walk(n, drift):
        return 100 * np.cumprod(1 + rng.normal(drift, 0.01, n))

    a = walk(500, 0.0004)
    b = walk(500, 0.0003)
    c = np.full(500, np.nan)
    c[300:] = walk(200, 0.0006)
    return pd.DataFrame({"A": a, "B": b, "C": c}, index=idx)


WEIGHTS = {"A": 0.4, "B": 0.3, "C": 0.3}


def test_rescale_uses_full_window_not_truncated():
    prices = _synthetic_prices()
    r = build_backtest(prices, WEIGHTS, 1_000_000, inception_mode="rescale")
    # The series spans the whole window, not just C's inception onward.
    assert len(r.values) == 500
    assert r.values.index[0] == prices.index[0]
    # C has zero weight before it exists, positive after.
    assert r.weight_history["C"].iloc[299] == 0.0
    assert r.weight_history["C"].iloc[-1] > 0.0


def test_rescale_weights_sum_to_one():
    prices = _synthetic_prices()
    r = build_backtest(prices, WEIGHTS, 1_000_000, inception_mode="rescale")
    wsum = r.weight_history[["A", "B", "C"]].sum(axis=1)
    assert wsum.min() == pytest.approx(1.0, abs=1e-6)
    assert wsum.max() == pytest.approx(1.0, abs=1e-6)


def test_cash_mode_reserves_weight_until_inception():
    prices = _synthetic_prices()
    r = build_backtest(prices, WEIGHTS, 1_000_000, rf_annual=0.03, inception_mode="cash")
    assert "Cash" in r.weight_history.columns
    # Before C exists, ~its 0.3 weight sits in cash; after, cash ~0.
    assert r.weight_history["Cash"].iloc[299] > 0.25
    assert r.weight_history["Cash"].iloc[-1] < 0.02


def test_coverage_and_effective_start():
    prices = _synthetic_prices()
    r = build_backtest(prices, WEIGHTS, 1_000_000)
    assert r.coverage["A"] == prices.index[0]
    assert r.coverage["C"] == prices.index[300]
    assert r.effective_start == prices.index[300]  # date all assets present


def test_costs_reduce_terminal_value_and_log_trades():
    prices = _synthetic_prices()
    free = build_backtest(prices, WEIGHTS, 1_000_000, rebalance_frequency="quarterly", cost_bps=0)
    paid = build_backtest(prices, WEIGHTS, 1_000_000, rebalance_frequency="quarterly", cost_bps=50)
    assert paid.values.iloc[-1] < free.values.iloc[-1]
    assert paid.total_costs > 0
    assert not paid.trades.empty
    assert {"Date", "Ticker", "SharesDelta", "Price", "Value"}.issubset(paid.trades.columns)


def test_buy_and_hold_only_rebalances_on_entry():
    prices = _synthetic_prices()
    r = build_backtest(prices, WEIGHTS, 1_000_000, rebalance_frequency="none")
    # One rebalance at inception + one when C enters.
    assert len(r.rebalance_dates) == 2


def test_no_staggering_all_assets_from_start():
    prices = _synthetic_prices()[["A", "B"]]
    r = build_backtest(prices, {"A": 0.5, "B": 0.5}, 1_000_000, rebalance_frequency="none")
    assert len(r.rebalance_dates) == 1  # only the opening allocation
    assert list(r.initial_holdings["Ticker"]) == ["A", "B"]
