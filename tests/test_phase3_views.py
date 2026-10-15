"""View-model helpers for the Phase 3 panels (no Qt rendering)."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.ui.tabs.attribution_tab import _factor_display
from src.ui.tabs.optimization_tab import OptimizationTab


def _ps(name, end_value):
    return SimpleNamespace(name=name, ann_return=0.1, ann_vol=0.2, sharpe=0.5, max_dd=-0.2,
                           values=pd.Series([1.0, end_value]))


def test_strategy_table_turnover_and_cost():
    res = SimpleNamespace(
        config=SimpleNamespace(weights={"A": 0.5, "B": 0.5},
                               backtest=SimpleNamespace(transaction_cost_bps=10.0)),
        trade_recos=pd.DataFrame({"Ticker": ["A", "B"], "CurrentWeight": [0.7, 0.3]}),
        orp_optimization=SimpleNamespace(weights=pd.Series({"A": 0.0, "B": 1.0})),
        hrp_weights=None, active=_ps("Active", 100_000.0), passive=_ps("Passive", 1.0),
        orp=_ps("ORP", 1.0), hrp=None, rebalanced=None, complete=None,
    )
    t = OptimizationTab._strategy_table(res).set_index("Strategy")
    assert t.loc["Active", "Turnover"] == pytest.approx(0.2)
    assert t.loc["Active", "Cost"] == pytest.approx(2 * 0.2 * 100_000 * 10 / 1e4)
    assert t.loc["ORP", "Turnover"] == pytest.approx(0.7)
    assert np.isnan(t.loc["Passive", "Turnover"])


def test_factor_display_formats_coef_with_t():
    df = pd.DataFrame({"Asset": ["P"], "const_coef": [0.0001], "const_t": [2.5],
                       "Mkt-RF_coef": [1.05], "Mkt-RF_t": [44.6], "R2": [0.9], "N": [500]})
    out, cols = _factor_display(df)
    assert out.loc[0, "Mkt-RF"] == "1.05 (44.6)"
    assert out.loc[0, "Alpha"].endswith("(2.5)")
    assert [c.key for c in cols] == ["Asset", "Alpha", "Mkt-RF", "R2", "N"]
