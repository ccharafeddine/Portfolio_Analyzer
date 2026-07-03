"""Delayed-quote fetch + Live Market Watch formatting.

`fetch_quotes` is the data spine for the ticker strip and Live Market Watch.
yfinance is patched at module level (the fetch uses `fast_info`), so these run
offline. The formatting helpers are pure and imported directly.
"""

from types import SimpleNamespace
from unittest.mock import patch

from src.data import market_data
from src.data.market_data import Quote, fetch_quotes


def _fast_info(**kw):
    return SimpleNamespace(**kw)


def test_fetch_quotes_computes_change_and_pct():
    infos = {
        "AAPL": _fast_info(last_price=110.0, previous_close=100.0, day_high=111.0,
                           day_low=99.0, last_volume=1_000_000, currency="USD"),
        "MSFT": _fast_info(last_price=90.0, previous_close=100.0, day_high=101.0,
                           day_low=89.0, last_volume=2_000_000, currency="USD"),
    }
    mock_yf = SimpleNamespace(Ticker=lambda t: SimpleNamespace(fast_info=infos[t]))
    with patch.object(market_data, "yf", mock_yf):
        out = fetch_quotes(["AAPL", "MSFT"], use_cache=False)

    assert set(out) == {"AAPL", "MSFT"}
    up = out["AAPL"]
    assert up.ok and up.last == 110.0 and up.prev_close == 100.0
    assert up.change == 10.0
    assert abs(up.change_pct - 0.10) < 1e-9
    assert up.day_high == 111.0 and up.day_low == 99.0
    assert up.volume == 1_000_000 and up.currency == "USD"
    assert up.as_of  # ISO timestamp stamped

    down = out["MSFT"]
    assert down.change == -10.0 and abs(down.change_pct + 0.10) < 1e-9


def test_fetch_quotes_bad_symbol_is_not_ok_but_does_not_raise():
    def ticker(t):
        if t == "GOOD":
            return SimpleNamespace(fast_info=_fast_info(last_price=50.0, previous_close=48.0))
        raise RuntimeError("no such symbol")

    mock_yf = SimpleNamespace(Ticker=ticker)
    with patch.object(market_data, "yf", mock_yf):
        out = fetch_quotes(["GOOD", "ZZZZ"], use_cache=False)

    assert out["GOOD"].ok is True
    assert out["ZZZZ"].ok is False  # empty quote, no exception
    assert out["ZZZZ"].last is None


def test_fetch_quotes_handles_zero_prev_close():
    mock_yf = SimpleNamespace(
        Ticker=lambda t: SimpleNamespace(fast_info=_fast_info(last_price=10.0, previous_close=0.0))
    )
    with patch.object(market_data, "yf", mock_yf):
        out = fetch_quotes(["X"], use_cache=False)
    q = out["X"]
    assert q.last == 10.0
    assert q.change is None and q.change_pct is None  # no divide-by-zero


def test_fetch_quotes_without_yfinance_returns_empty_quotes():
    with patch.object(market_data, "yf", None):
        out = fetch_quotes(["AAPL"], use_cache=False)
    assert isinstance(out["AAPL"], Quote) and out["AAPL"].ok is False


def test_fetch_quotes_empty_universe():
    assert fetch_quotes([], use_cache=False) == {}
    assert fetch_quotes(None, use_cache=False) == {}


def test_fetch_quotes_dedupes_and_uppercases():
    seen = []

    def ticker(t):
        seen.append(t)
        return SimpleNamespace(fast_info=_fast_info(last_price=1.0, previous_close=1.0))

    with patch.object(market_data, "yf", SimpleNamespace(Ticker=ticker)):
        out = fetch_quotes(["aapl", "AAPL", "msft"], use_cache=False)
    assert set(out) == {"AAPL", "MSFT"}
    assert sorted(seen) == ["AAPL", "MSFT"]  # deduped, uppercased


def test_live_watch_formatting_helpers():
    from src.ui.live_watch_view import _fmt_price, _fmt_signed, _fmt_volume, _fmt_pct

    assert _fmt_price(1234.5) == "1,234.50"
    assert _fmt_price(None) == "—"
    assert _fmt_signed(2.5) == "+2.50"
    assert _fmt_signed(-2.5) == "-2.50"
    assert _fmt_signed(0.0123, pct=True) == "+1.23%"
    assert _fmt_signed(-0.0123, pct=True) == "-1.23%"
    assert _fmt_signed(float("nan")) == "—"
    assert _fmt_volume(2_500_000) == "2.50M"
    assert _fmt_volume(1_200_000_000) == "1.20B"
    assert _fmt_volume(None) == "—"
    assert _fmt_pct(0.25) == "25.00%"
