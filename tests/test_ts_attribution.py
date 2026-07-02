"""Tests for time-series (benchmark-relative) attribution."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analytics.attribution import time_series_attribution


def _returns():
    idx = pd.period_range("2022-01", periods=6, freq="M").to_timestamp("M")
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "AAPL": rng.normal(0.01, 0.05, 6),
        "MSFT": rng.normal(0.008, 0.05, 6),
        "SPY": rng.normal(0.006, 0.03, 6),
    }, index=idx)


def test_contributions_sum_to_policy_active_return():
    r = _returns()
    weights = {"AAPL": 0.6, "MSFT": 0.4}
    contrib = time_series_attribution(r, weights, "SPY")
    assert list(contrib.columns) == ["AAPL", "MSFT"]
    # Each period: sum of contributions == sum_i w_i (r_i - r_b).
    expected = 0.6 * (r["AAPL"] - r["SPY"]) + 0.4 * (r["MSFT"] - r["SPY"])
    np.testing.assert_allclose(contrib.sum(axis=1).values, expected.values, atol=1e-12)


def test_single_asset_contribution_value():
    r = _returns()
    contrib = time_series_attribution(r, {"AAPL": 1.0}, "SPY")
    # One holding at 100% -> contribution == its excess vs benchmark each period.
    np.testing.assert_allclose(
        contrib["AAPL"].values, (r["AAPL"] - r["SPY"]).values, atol=1e-12
    )


def test_weights_normalized_and_benchmark_excluded():
    r = _returns()
    # Raw weights (sum 2) including the benchmark ticker (should be dropped).
    contrib = time_series_attribution(r, {"AAPL": 1.0, "MSFT": 1.0, "SPY": 0.5}, "SPY")
    assert "SPY" not in contrib.columns
    # Normalized to 0.5/0.5.
    expected_aapl = 0.5 * (r["AAPL"] - r["SPY"])
    np.testing.assert_allclose(contrib["AAPL"].values, expected_aapl.values, atol=1e-12)


def test_missing_benchmark_returns_empty():
    r = _returns()
    assert time_series_attribution(r, {"AAPL": 1.0}, "MISSING").empty
