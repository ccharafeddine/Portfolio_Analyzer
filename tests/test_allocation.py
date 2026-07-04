"""Shares → weights + capital conversion, and the config ``shares`` field.

Qt-free: the helper and the pydantic model import no PySide6.
"""

from datetime import date

import pytest

from src.config.models import PortfolioConfig
from src.ui.allocation import shares_to_weights_and_capital


def test_shares_to_weights_and_capital_basic():
    shares = {"AAPL": 10, "MSFT": 5}
    prices = {"AAPL": 100.0, "MSFT": 200.0}  # 1000 + 1000 = 2000, 50/50
    weights, capital = shares_to_weights_and_capital(shares, prices)
    assert capital == 2000.0
    assert weights == {"AAPL": 0.5, "MSFT": 0.5}
    assert abs(sum(weights.values()) - 1.0) < 1e-9


def test_shares_value_weighting_is_price_based():
    shares = {"A": 10, "B": 10}
    prices = {"A": 300.0, "B": 100.0}  # A worth 3x B -> 0.75 / 0.25
    weights, capital = shares_to_weights_and_capital(shares, prices)
    assert capital == 4000.0
    assert abs(weights["A"] - 0.75) < 1e-9 and abs(weights["B"] - 0.25) < 1e-9


def test_missing_price_and_zero_shares_skipped():
    shares = {"A": 10, "B": 5, "C": 0}
    prices = {"A": 100.0}  # B has no price, C has no shares
    weights, capital = shares_to_weights_and_capital(shares, prices)
    assert capital == 1000.0
    assert set(weights) == {"A"} and weights["A"] == 1.0


def test_no_priceable_holdings_returns_empty():
    assert shares_to_weights_and_capital({"A": 10}, {}) == ({}, 0.0)
    assert shares_to_weights_and_capital({}, {"A": 1.0}) == ({}, 0.0)


def test_case_insensitive_tickers():
    weights, capital = shares_to_weights_and_capital({"aapl": 2}, {"AAPL": 50.0})
    assert capital == 100.0 and weights == {"AAPL": 1.0}


def _cfg(**kw):
    base = dict(
        tickers=["AAPL", "MSFT"],
        weights={"AAPL": 0.5, "MSFT": 0.5},
        start_date=date(2021, 1, 1),
        end_date=date(2024, 1, 1),
    )
    base.update(kw)
    return PortfolioConfig(**base)


def test_config_shares_defaults_empty_and_roundtrips():
    assert _cfg().shares == {}
    c = _cfg(shares={"aapl": 10, "msft": 5})
    assert c.shares == {"AAPL": 10.0, "MSFT": 5.0}  # uppercased, floats
    # Survives a JSON dump/load round-trip.
    reloaded = PortfolioConfig.model_validate(c.model_dump(mode="json"))
    assert reloaded.shares == {"AAPL": 10.0, "MSFT": 5.0}


def test_config_without_shares_still_valid():
    # Older saved files have no shares key — must load fine.
    data = _cfg().model_dump(mode="json")
    data.pop("shares", None)
    assert PortfolioConfig.model_validate(data).shares == {}
