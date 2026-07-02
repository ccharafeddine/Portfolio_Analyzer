"""Unit tests for the News & Macro data layer (src/data/market_data.py).

Network is fully mocked: yfinance via a fake Ticker, Alpha Vantage / FRED via a
fake requests.get. Caching is disabled so tests are deterministic.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.data import market_data as md


# ── yfinance news ──

def _yf_item(title, url, ts="2026-07-01T21:10:00Z", publisher="Yahoo"):
    return {
        "id": url,
        "content": {
            "title": title,
            "provider": {"displayName": publisher},
            "clickThroughUrl": {"url": url},
            "pubDate": ts,
            "summary": f"<p>About {title}</p>",
            "thumbnail": {"originalUrl": "https://img/x.png"},
        },
    }


def _install_yf(monkeypatch, mapping):
    fake_yf = MagicMock()

    def ticker(sym):
        t = MagicMock()
        t.news = mapping.get(sym, [])
        return t

    fake_yf.Ticker.side_effect = ticker
    monkeypatch.setattr(md, "yf", fake_yf)


def test_yf_news_normalization(monkeypatch):
    _install_yf(monkeypatch, {"AAPL": [_yf_item("Apple soars", "https://n/a")]})
    items = md.fetch_ticker_news(["AAPL"], use_cache=False)
    assert len(items) == 1
    it = items[0]
    assert it.title == "Apple soars"
    assert it.publisher == "Yahoo"
    assert it.url == "https://n/a"
    assert it.tickers == ["AAPL"]
    assert it.summary == "About Apple soars"  # HTML stripped
    assert it.published is not None and it.published.tzinfo is not None


def test_dedupe_merges_tickers(monkeypatch):
    # Same URL surfaced under two tickers → one item, merged tickers.
    _install_yf(monkeypatch, {
        "AAPL": [_yf_item("Big tech rallies", "https://n/shared")],
        "MSFT": [_yf_item("Big tech rallies", "https://n/shared")],
    })
    items = md.fetch_ticker_news(["AAPL", "MSFT"], use_cache=False)
    assert len(items) == 1
    assert set(items[0].tickers) == {"AAPL", "MSFT"}


def test_news_sorted_newest_first(monkeypatch):
    _install_yf(monkeypatch, {"AAPL": [
        _yf_item("Old", "https://n/old", ts="2026-06-01T00:00:00Z"),
        _yf_item("New", "https://n/new", ts="2026-07-01T00:00:00Z"),
    ]})
    items = md.fetch_ticker_news(["AAPL"], use_cache=False)
    assert [i.title for i in items] == ["New", "Old"]


def test_yf_failure_is_graceful(monkeypatch):
    fake_yf = MagicMock()
    fake_yf.Ticker.side_effect = RuntimeError("network down")
    monkeypatch.setattr(md, "yf", fake_yf)
    # Should swallow the error and return an empty list, not raise.
    assert md.fetch_ticker_news(["AAPL"], use_cache=False) == []


# ── Alpha Vantage sentiment ──

def test_alpha_vantage_sentiment(monkeypatch):
    _install_yf(monkeypatch, {"AAPL": []})
    av_payload = {"feed": [{
        "title": "Analysts bullish on Apple",
        "url": "https://av/apple",
        "time_published": "20260701T120000",
        "summary": "Upgrades roll in.",
        "source": "Reuters",
        "overall_sentiment_score": 0.42,
        "overall_sentiment_label": "Bullish",
        "ticker_sentiment": [{"ticker": "AAPL"}],
    }]}
    fake_requests = MagicMock()
    fake_requests.get.return_value.json.return_value = av_payload
    monkeypatch.setattr(md, "requests", fake_requests)

    items = md.fetch_ticker_news(["AAPL"], av_key="KEY", use_cache=False)
    assert len(items) == 1
    it = items[0]
    assert it.sentiment == pytest.approx(0.42)
    assert it.sentiment_label == "Bullish"
    assert it.publisher == "Reuters"
    assert it.tickers == ["AAPL"]


# ── FRED macro ──

def _fred_response(series_id):
    data = {
        "DGS3MO": [("2026-06-01", "5.10"), ("2025-06-01", "5.30")],
        "DGS2": [("2026-06-01", "4.60"), ("2025-06-01", "4.80")],
        "DGS10": [("2026-06-01", "4.20"), ("2025-06-01", "3.90")],
        "DGS30": [("2026-06-01", "4.40"), ("2025-06-01", "4.10")],
        "FEDFUNDS": [("2026-06-01", "5.00"), ("2025-06-01", "5.25")],
        "UNRATE": [("2026-06-01", "4.10"), ("2025-06-01", "3.90")],
        "CPIAUCSL": [("2026-06-01", "320.0"), ("2025-06-01", "310.0")],
    }
    return {"observations": [{"date": d, "value": v} for d, v in data.get(series_id, [])]}


def _install_fred(monkeypatch):
    fake_requests = MagicMock()

    def get(url, params=None, timeout=None):
        resp = MagicMock()
        resp.json.return_value = _fred_response((params or {}).get("series_id"))
        return resp

    fake_requests.get.side_effect = get
    monkeypatch.setattr(md, "requests", fake_requests)


def test_macro_requires_key():
    assert md.fetch_macro(None) is None


def test_macro_parsing(monkeypatch):
    _install_fred(monkeypatch)
    macro = md.fetch_macro("FREDKEY", use_cache=False)
    assert macro is not None
    # Curve uses the latest observation per tenor.
    assert macro.curve["10Y"] == pytest.approx(4.20)
    assert macro.curve["3M"] == pytest.approx(5.10)
    # Rates table present for every series.
    names = {r["name"] for r in macro.rates}
    assert "10-Year Treasury" in names and "CPI (YoY)" in names
    # CPI converted to YoY %: 320/310 - 1 ≈ 3.23%.
    cpi = next(r for r in macro.rates if r["name"] == "CPI (YoY)")
    assert cpi["value"] == pytest.approx((320.0 / 310.0 - 1) * 100, rel=1e-3)
    # 1-yr change for the 10Y: 4.20 - 3.90 = +0.30.
    ten = next(r for r in macro.rates if r["name"] == "10-Year Treasury")
    assert ten["change_1y"] == pytest.approx(0.30, abs=1e-6)
    # History series retained for the chart.
    assert "DGS10" in macro.series and "FEDFUNDS" in macro.series


def test_macro_roundtrip_serialization(monkeypatch):
    _install_fred(monkeypatch)
    macro = md.fetch_macro("FREDKEY", use_cache=False)
    restored = md.MacroData.from_dict(macro.to_dict())
    assert restored.curve == macro.curve
    assert len(restored.rates) == len(macro.rates)
    assert "DGS10" in restored.series
