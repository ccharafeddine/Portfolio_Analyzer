"""Unit tests for the fundamentals data layer (src/data/fundamentals.py).

yfinance and FMP are fully mocked; caching is disabled for determinism.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import pytest

from src.data import fundamentals as fu


_AAPL_INFO = {
    "longName": "Apple Inc.",
    "sector": "Technology",
    "industry": "Consumer Electronics",
    "marketCap": 4_000_000_000_000,
    "currentPrice": 250.0,
    "trailingPE": 35.6,
    "forwardPE": 30.6,
    "priceToBook": 40.5,
    "priceToSalesTrailing12Months": 9.5,
    "pegRatio": 2.3,
    "enterpriseToEbitda": 27.1,
    "grossMargins": 0.478,
    "operatingMargins": 0.322,
    "profitMargins": 0.271,
    "returnOnEquity": 1.41,
    "returnOnAssets": 0.262,
    "revenueGrowth": 0.166,
    "earningsGrowth": 0.218,
    "debtToEquity": 79.5,
    "currentRatio": 1.07,
    "dividendYield": 0.37,
    "payoutRatio": 0.126,
    "beta": 1.09,
    "trailingEps": 8.27,
    "forwardEps": 9.6,
}

_AAPL_CAL = {
    "Earnings Date": [date(2026, 7, 30)],
    "Ex-Dividend Date": date(2026, 5, 10),
}


def _install_yf(monkeypatch, info, cal):
    fake = MagicMock()
    tk = MagicMock()
    tk.info = info
    tk.calendar = cal
    fake.Ticker.return_value = tk
    monkeypatch.setattr(fu, "yf", fake)


def test_yfinance_mapping(monkeypatch):
    _install_yf(monkeypatch, _AAPL_INFO, _AAPL_CAL)
    items = fu.fetch_fundamentals(["AAPL"], use_cache=False)
    assert len(items) == 1
    f = items[0]
    assert f.name == "Apple Inc."
    assert f.sector == "Technology"
    assert f.pe == pytest.approx(35.6)
    assert f.net_margin == pytest.approx(0.271)
    assert f.roe == pytest.approx(1.41)
    assert f.dividend_yield == pytest.approx(0.37)
    assert f.beta == pytest.approx(1.09)
    assert f.market_cap == pytest.approx(4_000_000_000_000)


def test_calendar_dates(monkeypatch):
    _install_yf(monkeypatch, _AAPL_INFO, _AAPL_CAL)
    f = fu.fetch_fundamentals(["AAPL"], use_cache=False)[0]
    assert f.next_earnings == "2026-07-30"
    assert f.ex_dividend == "2026-05-10"


def test_missing_fields_are_none(monkeypatch):
    _install_yf(monkeypatch, {"longName": "Sparse Co"}, {})
    f = fu.fetch_fundamentals(["XYZ"], use_cache=False)[0]
    assert f.name == "Sparse Co"
    assert f.pe is None and f.roe is None and f.next_earnings is None


def test_yfinance_failure_is_graceful(monkeypatch):
    fake = MagicMock()
    fake.Ticker.side_effect = RuntimeError("network down")
    monkeypatch.setattr(fu, "yf", fake)
    items = fu.fetch_fundamentals(["AAPL"], use_cache=False)
    assert len(items) == 1 and items[0].ticker == "AAPL" and items[0].pe is None


def test_fmp_dcf_enrichment(monkeypatch):
    _install_yf(monkeypatch, _AAPL_INFO, _AAPL_CAL)
    fake_requests = MagicMock()
    fake_requests.get.return_value.json.return_value = [
        {"symbol": "AAPL", "dcf": 210.5, "price": 250.0, "sector": "Technology"}
    ]
    monkeypatch.setattr(fu, "requests", fake_requests)
    f = fu.fetch_fundamentals(["AAPL"], fmp_key="KEY", use_cache=False)[0]
    assert f.dcf == pytest.approx(210.5)


def test_roundtrip_serialization(monkeypatch):
    _install_yf(monkeypatch, _AAPL_INFO, _AAPL_CAL)
    f = fu.fetch_fundamentals(["AAPL"], use_cache=False)[0]
    restored = fu.Fundamentals.from_dict(f.to_dict())
    assert restored.name == f.name and restored.pe == f.pe


def test_fetch_statements(monkeypatch):
    import pandas as pd

    periods = [pd.Timestamp("2025-09-30"), pd.Timestamp("2024-09-30")]
    income = pd.DataFrame(
        {periods[0]: [400e9, 100e9, 6.1], periods[1]: [383e9, 97e9, 6.0]},
        index=["Total Revenue", "Net Income", "Diluted EPS"],
    )
    balance = pd.DataFrame(
        {periods[0]: [360e9], periods[1]: [352e9]}, index=["Total Assets"]
    )
    cashflow = pd.DataFrame(
        {periods[0]: [110e9], periods[1]: [99e9]}, index=["Free Cash Flow"]
    )
    rec = pd.DataFrame([{"period": "0m", "strongBuy": 6, "buy": 22, "hold": 16,
                         "sell": 1, "strongSell": 2}])

    tk = MagicMock()
    tk.income_stmt = income
    tk.balance_sheet = balance
    tk.cashflow = cashflow
    tk.analyst_price_targets = {"current": 294.0, "high": 400.0, "low": 215.0,
                                "mean": 315.0, "median": 315.0}
    tk.recommendations_summary = rec
    fake = MagicMock()
    fake.Ticker.return_value = tk
    monkeypatch.setattr(fu, "yf", fake)

    data = fu.fetch_statements("AAPL", use_cache=False)
    assert data["income"]["periods"] == ["2025", "2024"]
    assert data["income"]["rows"]["Total Revenue"][0] == pytest.approx(400e9)
    assert data["balance"]["rows"]["Total Assets"][1] == pytest.approx(352e9)
    assert data["cashflow"]["rows"]["Free Cash Flow"][0] == pytest.approx(110e9)
    assert data["target"]["mean"] == pytest.approx(315.0)
    assert data["recommendation"]["strongBuy"] == 6
