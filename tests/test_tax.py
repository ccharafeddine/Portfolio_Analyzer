"""Tests for tax-aware analysis (src/analytics/tax.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analytics.backtest import build_backtest
from src.analytics.tax import build_tax_analysis
from src.config.models import TaxConfig


def _prices_gain_and_loss():
    idx = pd.bdate_range("2022-01-03", periods=260)  # ~1 year
    a = np.linspace(100, 150, 260)  # winner
    b = np.linspace(100, 80, 260)   # loser
    return pd.DataFrame({"A": a, "B": b}, index=idx)


def test_unrealized_gain_and_loss_split():
    prices = _prices_gain_and_loss()
    bt = build_backtest(prices, {"A": 0.5, "B": 0.5}, 1_000_000, rebalance_frequency="none")
    tax_cfg = TaxConfig(enabled=True)
    metrics, detail = build_tax_analysis(bt, prices, {"A": 100.0, "B": 100.0}, tax_cfg)

    assert len(detail) == 2
    a_row = detail.set_index("Ticker").loc["A"]
    b_row = detail.set_index("Ticker").loc["B"]
    assert a_row["UnrealizedGain"] > 0     # A above cost
    assert b_row["UnrealizedGain"] < 0     # B below cost
    assert metrics["n_harvest"] == 1       # B is the harvest candidate
    assert metrics["harvest_potential"] < 0
    assert metrics["cost_basis_source"] == "user"


def test_inferred_cost_basis_when_empty():
    prices = _prices_gain_and_loss()
    bt = build_backtest(prices, {"A": 0.5, "B": 0.5}, 1_000_000, rebalance_frequency="none")
    metrics, detail = build_tax_analysis(bt, prices, {}, TaxConfig())
    # With cost basis = opening price, A is still a gain, B still a loss.
    assert metrics["cost_basis_source"].startswith("inferred")
    assert detail.set_index("Ticker").loc["A"]["AvgCost"] > 0


def test_realized_tax_from_rebalancing():
    prices = _prices_gain_and_loss()
    # Quarterly rebalancing trims the winner (A) -> realized gains -> tax owed.
    bt = build_backtest(prices, {"A": 0.5, "B": 0.5}, 1_000_000, rebalance_frequency="quarterly")
    metrics, _ = build_tax_analysis(bt, prices, {"A": 100.0, "B": 100.0}, TaxConfig(enabled=True))
    assert metrics["estimated_tax"] >= 0.0
    # A was trimmed at a profit, so some realized gain should exist.
    assert (metrics["realized_short"] + metrics["realized_long"]) > 0
