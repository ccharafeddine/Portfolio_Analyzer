"""Tests for retirement/withdrawal planning (src/analytics/simulation.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analytics.simulation import _recenter, run_retirement_plan


def _returns(seed=1):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0006, 0.011, 1200))


def test_recenter_keeps_vol_changes_mean():
    r = _returns().values
    adj = _recenter(r, 0.07)
    target_daily = (1.07) ** (1 / 252) - 1
    assert abs(adj.mean() - target_daily) < 1e-9      # mean shifted to target
    assert abs(np.std(adj) - np.std(r)) < 1e-12       # volatility (shape) preserved


def test_higher_withdrawal_increases_depletion():
    r = _returns()
    low = run_retirement_plan(1_000_000, r, horizon_years=30, annual_withdrawal=30_000,
                              expected_return=0.06, compute_swr=False)
    high = run_retirement_plan(1_000_000, r, horizon_years=30, annual_withdrawal=90_000,
                               expected_return=0.06, compute_swr=False)
    assert high.depletion_prob > low.depletion_prob
    assert high.success_prob < low.success_prob


def test_contributions_grow_portfolio():
    r = _returns()
    plan = run_retirement_plan(500_000, r, horizon_years=20, annual_contribution=24_000,
                               expected_return=0.07, compute_swr=False)
    assert plan.median_terminal > 500_000
    assert plan.depletion_prob == 0.0  # only adding money, never depletes


def test_goal_probability():
    r = _returns()
    plan = run_retirement_plan(500_000, r, horizon_years=20, annual_contribution=24_000,
                               goal_amount=2_000_000, expected_return=0.07, compute_swr=False)
    assert 0.0 <= plan.goal_prob <= 1.0
    assert plan.success_prob == plan.goal_prob  # goal set -> success == goal prob


def test_plan_result_shape():
    r = _returns()
    plan = run_retirement_plan(1_000_000, r, horizon_years=10, annual_withdrawal=40_000,
                               expected_return=0.06)
    assert plan.paths.shape == (1000, 10 * 252)
    assert plan.safe_withdrawal_rate is not None
    assert 0.0 <= plan.safe_withdrawal_rate <= 0.15
