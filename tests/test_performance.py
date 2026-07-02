"""Tests for the performance-measurement suite (src/analytics/performance.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analytics import performance as perf

# Simple deterministic monthly returns with known answers.
_IDX = pd.date_range("2022-01-31", periods=4, freq="ME")
PORT = pd.Series([0.02, -0.01, 0.03, 0.00], index=_IDX)
BENCH = pd.Series([0.01, -0.02, 0.02, 0.01], index=_IDX)


def test_batting_average():
    # port > bench in 3 of 4 months.
    assert perf.batting_average(PORT, BENCH) == pytest.approx(0.75)


def test_up_down_capture():
    cap = perf.up_down_capture(PORT, BENCH)
    # Up months (bench>0): 0.02,0.03,0.00 vs 0.01,0.02,0.01.
    port_up = 1.02 * 1.03 * 1.00 - 1
    bench_up = 1.01 * 1.02 * 1.01 - 1
    assert cap["up_capture"] == pytest.approx(port_up / bench_up, rel=1e-6)
    # Down month (bench<0): -0.01 vs -0.02.
    assert cap["down_capture"] == pytest.approx(0.5, rel=1e-6)


def test_tracking_error():
    # diff = [0.01,0.01,0.01,-0.01], std(ddof=1)=0.01, annualized ×sqrt(12).
    assert perf.tracking_error(PORT, BENCH) == pytest.approx(0.01 * np.sqrt(12), rel=1e-6)


def test_information_ratio_positive_when_outperforming():
    ir = perf.information_ratio(PORT, BENCH)
    assert ir > 0  # portfolio beats benchmark on average


def test_blended_benchmark_60_40():
    idx = pd.bdate_range("2022-01-03", periods=10)
    prices = pd.DataFrame({"SPY": np.linspace(100, 110, 10), "BND": np.linspace(50, 50, 10)}, index=idx)
    blended = perf.blended_benchmark(prices, {"SPY": 0.6, "BND": 0.4}, 1_000_000)
    assert blended.iloc[0] == pytest.approx(1_000_000, rel=1e-6)
    # SPY +10%, BND flat -> ~+6% blended.
    assert blended.iloc[-1] == pytest.approx(1_060_000, rel=1e-3)


def test_performance_summary_shape():
    vals_a = pd.Series(np.linspace(1_000_000, 1_200_000, 400),
                       index=pd.bdate_range("2022-01-03", periods=400))
    vals_b = pd.Series(np.linspace(1_000_000, 1_100_000, 400), index=vals_a.index)
    df = perf.performance_summary(vals_a, vals_b, rf=0.04)
    assert "Metric" in df.columns and "Full History" in df.columns
    assert {"Ann. Return", "Tracking Error", "Up Capture", "Beta"}.issubset(set(df["Metric"]))
