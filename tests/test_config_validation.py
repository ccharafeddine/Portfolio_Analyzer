from datetime import date

import pytest
from pydantic import ValidationError

from src.config.models import PortfolioConfig

BASE = dict(tickers=["aapl", "MSFT"], weights={"AAPL": 0.5, "MSFT": 0.5},
            start_date=date(2020, 1, 1), end_date=date(2024, 1, 1))


def cfg(**kw):
    return PortfolioConfig(**{**BASE, **kw})


def test_dedupes_and_uppercases():
    assert cfg(tickers=["aapl", " AAPL ", "msft", ""]).tickers == ["AAPL", "MSFT"]


@pytest.mark.parametrize("kw, msg", [
    (dict(weights={"AAPL": 1.5, "MSFT": -0.5}), "short sales"),
    (dict(weights={"AAPL": float("nan"), "MSFT": 0.5}), "finite"),
    (dict(end_date=date(2020, 2, 1)), "60 days"),
    (dict(start_date=date(1960, 1, 1)), "1970"),
    (dict(benchmark_weights={"spy": 0.6, "agg": 0.3}), "sum to"),
    (dict(benchmark_weights={"SPY": 1.2, "AGG": -0.2}), "non-negative"),
    (dict(tickers=["AAPL", "MS<FT"]), "Invalid symbol"),
])
def test_rejects_bad_input(kw, msg):
    with pytest.raises(ValidationError, match=msg):
        cfg(**kw)


def test_shorts_allowed_when_enabled():
    c = cfg(weights={"AAPL": 1.5, "MSFT": -0.5}, short_sales=True)
    assert c.weights["MSFT"] == -0.5
    assert cfg(benchmark_weights={"spy": 0.6, "agg": 0.4}).benchmark_weights == {"SPY": 0.6, "AGG": 0.4}
