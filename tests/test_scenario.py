"""Tests for the what-if scenario model (betas + drivers)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analytics.scenario import MACRO_FACTORS, PRESETS, build_scenario_model


def _macro_prices(n=400, seed=0):
    idx = pd.bdate_range("2021-01-04", periods=n)
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {t: 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, n)) for _, t in MACRO_FACTORS},
        index=idx,
    )


def test_holdings_use_weight_as_beta():
    macro = _macro_prices()
    port = pd.Series(
        np.random.default_rng(1).normal(0.0004, 0.011, len(macro) - 1),
        index=macro.index[1:],
    )
    model = build_scenario_model(port, macro, {"AAPL": 0.6, "MSFT": 0.4}, 1_000_000)
    holds = {d["name"]: d for d in model["drivers"] if d["group"] == "Holding"}
    assert holds["AAPL"]["beta"] == 0.6
    assert holds["MSFT"]["beta"] == 0.4
    assert model["value"] == 1_000_000
    assert model["presets"] == PRESETS


def test_weights_normalized():
    macro = _macro_prices()
    port = pd.Series(np.zeros(len(macro) - 1), index=macro.index[1:])
    model = build_scenario_model(port, macro, {"A": 2.0, "B": 2.0}, 500_000)
    holds = {d["name"]: d for d in model["drivers"] if d["group"] == "Holding"}
    assert holds["A"]["beta"] == 0.5 and holds["B"]["beta"] == 0.5


def test_macro_beta_recovered():
    # Build a portfolio that is exactly 1.3x the SPY factor return -> beta ~1.3 on SPY.
    macro = _macro_prices(seed=3)
    spy_ret = macro["SPY"].pct_change().dropna()
    port = 1.3 * spy_ret + np.random.default_rng(9).normal(0, 1e-4, len(spy_ret))
    model = build_scenario_model(port, macro, {"AAPL": 1.0}, 1_000_000)
    macro_drivers = {d["name"]: d for d in model["drivers"] if d["group"] == "Macro"}
    assert "Equity (S&P 500)" in macro_drivers
    assert abs(macro_drivers["Equity (S&P 500)"]["beta"] - 1.3) < 0.1


def test_no_macro_when_prices_missing():
    idx = pd.bdate_range("2021-01-04", periods=100)
    port = pd.Series(np.random.default_rng(2).normal(0, 0.01, 100), index=idx)
    model = build_scenario_model(port, pd.DataFrame(), {"AAPL": 1.0}, 1000)
    assert all(d["group"] == "Holding" for d in model["drivers"])
    assert len(model["drivers"]) == 1
