"""Real-time quote providers + resolver.

Qt-free: patches ``market_data.requests`` (and yfinance) so no network is hit;
``settings.realtime_provider`` is exercised with ``get_api_key`` patched.
"""

from types import SimpleNamespace
from unittest.mock import patch

import src.data.market_data as md


class _Resp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def json(self):
        return self._payload


def _requests(get):
    return SimpleNamespace(get=get)


def test_finnhub_parsing_and_realtime_flag():
    def fake_get(url, params=None, headers=None, timeout=6):
        assert "finnhub" in url and params["symbol"] == "AAPL"
        return _Resp({"c": 150.0, "pc": 148.0, "h": 151.0, "l": 147.0})

    with patch.object(md, "requests", _requests(fake_get)):
        out = md._finnhub_quotes(["AAPL"], "key")
    qt = out["AAPL"]
    assert qt.last == 150.0 and qt.prev_close == 148.0
    assert qt.source == "finnhub" and qt.realtime is True
    assert abs(qt.change_pct - (2.0 / 148.0)) < 1e-9


def test_polygon_parsing():
    def fake_get(url, params=None, headers=None, timeout=6):
        return _Resp({"ticker": {"day": {"c": 90.0, "h": 91.0, "l": 89.0, "v": 1e6},
                                 "prevDay": {"c": 88.0},
                                 "lastTrade": {"p": 90.5}}})

    with patch.object(md, "requests", _requests(fake_get)):
        out = md._polygon_quotes(["MSFT"], "key")
    qt = out["MSFT"]
    assert qt.last == 90.5 and qt.prev_close == 88.0 and qt.realtime is True
    assert qt.source == "polygon"


def test_alpaca_needs_key_and_secret():
    def fake_get(url, params=None, headers=None, timeout=6):
        assert headers["APCA-API-KEY-ID"] == "k"
        return _Resp({"trades": {"TSLA": {"p": 240.0}}})

    with patch.object(md, "requests", _requests(fake_get)):
        out = md._alpaca_quotes(["TSLA"], ("k", "s"))
        assert out["TSLA"].last == 240.0 and out["TSLA"].realtime is True
        # Missing secret => no request, empty result.
        assert md._alpaca_quotes(["TSLA"], "k") == {}


def test_fetch_quotes_uses_provider_then_falls_back_to_yfinance():
    def fake_get(url, params=None, headers=None, timeout=6):
        if params.get("symbol") == "AAPL":
            return _Resp({"c": 150.0, "pc": 148.0})
        return _Resp({"c": 0})  # Finnhub has nothing for MSFT

    yf_infos = {"MSFT": SimpleNamespace(last_price=90.0, previous_close=100.0)}
    mock_yf = SimpleNamespace(Ticker=lambda t: SimpleNamespace(fast_info=yf_infos[t]))

    with patch.object(md, "requests", _requests(fake_get)), patch.object(md, "yf", mock_yf):
        out = md.fetch_quotes(["AAPL", "MSFT"], use_cache=False, provider="finnhub", creds="key")

    assert out["AAPL"].source == "finnhub" and out["AAPL"].realtime is True
    assert out["MSFT"].source == "yfinance" and out["MSFT"].realtime is False
    assert out["MSFT"].last == 90.0


def test_provider_exception_falls_back_entirely():
    def boom(url, params=None, headers=None, timeout=6):
        raise RuntimeError("provider down")

    yf_infos = {"AAPL": SimpleNamespace(last_price=100.0, previous_close=99.0)}
    mock_yf = SimpleNamespace(Ticker=lambda t: SimpleNamespace(fast_info=yf_infos[t]))
    with patch.object(md, "requests", _requests(boom)), patch.object(md, "yf", mock_yf):
        out = md.fetch_quotes(["AAPL"], use_cache=False, provider="finnhub", creds="key")
    assert out["AAPL"].source == "yfinance" and out["AAPL"].last == 100.0


def test_realtime_provider_resolver_priority():
    from src.ui import settings as S

    keys: dict = {}
    with patch.object(S, "get_api_key", lambda n: keys.get(n)):
        assert S.realtime_provider() == (None, None)

        keys["POLYGON_API_KEY"] = "p"
        assert S.realtime_provider() == ("polygon", "p")

        keys["FINNHUB_API_KEY"] = "f"      # finnhub outranks polygon
        assert S.realtime_provider() == ("finnhub", "f")

        keys.clear()
        keys["ALPACA_API_KEY"] = "k"       # alpaca needs both key + secret
        assert S.realtime_provider() == (None, None)
        keys["ALPACA_API_SECRET"] = "s"
        assert S.realtime_provider() == ("alpaca", ("k", "s"))
