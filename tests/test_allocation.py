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


def test_config_cash_defaults_zero_and_roundtrips():
    assert _cfg().cash == 0.0
    c = _cfg(cash=10_000.0)
    assert c.cash == 10_000.0
    reloaded = PortfolioConfig.model_validate(c.model_dump(mode="json"))
    assert reloaded.cash == 10_000.0
    # Older files without a cash key still load.
    data = _cfg().model_dump(mode="json")
    data.pop("cash", None)
    assert PortfolioConfig.model_validate(data).cash == 0.0


def test_config_cash_cannot_be_negative():
    with pytest.raises(Exception):
        _cfg(cash=-5.0)


def test_active_allocation_weights_adds_cash_slice():
    from src.ui.allocation import active_allocation_weights

    # capital is TOTAL: $100k total with $50k cash -> invested $50k, y = 0.5.
    out = active_allocation_weights({"AAPL": 0.5, "MSFT": 0.5}, 100_000, 50_000)
    assert abs(out["AAPL"] - 0.25) < 1e-9 and abs(out["MSFT"] - 0.25) < 1e-9
    assert abs(out["Cash"] - 0.5) < 1e-9
    assert abs(sum(out.values()) - 1.0) < 1e-9


def test_active_allocation_no_cash_passthrough():
    from src.ui.allocation import active_allocation_weights

    w = {"AAPL": 0.6, "MSFT": 0.4}
    out = active_allocation_weights(w, 100_000, 0)
    assert out == w and "Cash" not in out


def test_blend_cash_dilutes_return_and_vol():
    import numpy as np
    import pandas as pd

    from src.data import transforms as T

    idx = pd.date_range("2022-01-01", periods=252, freq="B")
    # A volatile risky sleeve starting at $10k.
    rng = np.random.default_rng(0)
    rets = rng.normal(0.0008, 0.02, len(idx))
    values = pd.Series(10_000 * np.cumprod(1 + rets), index=idx, name="Active")

    blended = T.blend_cash(values, cash=10_000, rf_annual=0.04)
    # Starts at invested + cash.
    assert abs(blended.iloc[0] - (values.iloc[0] + 10_000)) < 1.0
    # Cash sleeve halves the risky exposure -> lower volatility.
    assert T.annualize_vol(blended.pct_change().dropna()) < T.annualize_vol(
        values.pct_change().dropna()
    )
    # No cash -> unchanged.
    assert T.blend_cash(values, 0, 0.04) is values
