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


def _intraday_df():
    import pandas as pd

    idx = pd.date_range("2026-07-03 09:30", periods=5, freq="1min")
    return pd.DataFrame(
        {"Open": [100, 101, 102, 101, 103], "High": [101, 102, 103, 102, 104],
         "Low": [99, 100, 101, 100, 102], "Close": [100.5, 101.5, 102.5, 101.0, 103.5],
         "Volume": [1000, 1100, 900, 1200, 1300]},
        index=idx,
    )


def test_fetch_intraday_returns_frame():
    df = _intraday_df()
    mock_yf = SimpleNamespace(Ticker=lambda t: SimpleNamespace(history=lambda **kw: df))
    with patch.object(market_data, "yf", mock_yf):
        out = market_data.fetch_intraday("AAPL", use_cache=False)
    assert out is not None and not out.empty and "Close" in out.columns


def test_fetch_intraday_none_on_empty_or_error():
    import pandas as pd

    empty_yf = SimpleNamespace(Ticker=lambda t: SimpleNamespace(history=lambda **kw: pd.DataFrame()))
    with patch.object(market_data, "yf", empty_yf):
        assert market_data.fetch_intraday("AAPL", use_cache=False) is None

    def boom(**kw):
        raise RuntimeError("network")

    err_yf = SimpleNamespace(Ticker=lambda t: SimpleNamespace(history=boom))
    with patch.object(market_data, "yf", err_yf):
        assert market_data.fetch_intraday("AAPL", use_cache=False) is None

    with patch.object(market_data, "yf", None):
        assert market_data.fetch_intraday("AAPL", use_cache=False) is None


def test_intraday_chart_builds_figure():
    from src.charts import plotly_charts as charts

    fig = charts.intraday_chart(_intraday_df(), ticker="AAPL", prev_close=100.0)
    assert fig is not None and len(fig.data) >= 1
    assert charts.intraday_chart(None) is None


def test_holdings_treemap_builds_and_skips_zero_weight():
    from src.charts import plotly_charts as charts

    tickers = ["A", "B", "C"]
    weights = {"A": 0.5, "B": 0.5, "C": 0.0}   # C excluded (zero weight)
    changes = {"A": 0.02, "B": -0.01, "C": 0.0}
    fig = charts.holdings_treemap(tickers, weights, changes)
    assert fig is not None
    tm = fig.data[0]
    assert set(tm.labels) == {"A", "B"}         # C dropped
    assert charts.holdings_treemap([], {}, {}) is None


def test_day_change_heatmap_builds_for_normal_input():
    from src.charts import plotly_charts as charts

    symbols = ["AAPL", "MSFT", "NVDA", "TSLA", "SPY"]
    changes = {"AAPL": 0.012, "MSFT": -0.008, "NVDA": 0.031, "TSLA": -0.024, "SPY": 0.003}
    fig = charts.day_change_heatmap(symbols, changes)
    assert fig is not None and len(fig.data) == 1
    hm = fig.data[0]
    # Every symbol appears in the tile text; the grid is roughly square (5 -> 3x2).
    flat_text = " ".join(t for row in hm.text for t in row)
    for s in symbols:
        assert s in flat_text
    assert len(hm.z) == 2 and len(hm.z[0]) == 3          # rows x cols
    # Color bound is symmetric and clamped to the [±1%, ±5%] window.
    assert hm.zmax == 3.1 and hm.zmin == -3.1


def test_day_change_heatmap_handles_partial_and_missing_quotes():
    from src.charts import plotly_charts as charts

    # A symbol with no quote still tiles (neutral), as long as *someone* has a move.
    fig = charts.day_change_heatmap(["AAPL", "ZZZZ"], {"AAPL": 0.01})
    assert fig is not None
    flat_text = " ".join(t for row in fig.data[0].text for t in row)
    assert "AAPL" in flat_text and "ZZZZ" in flat_text


def test_day_change_heatmap_none_on_empty_or_all_failed():
    from src.charts import plotly_charts as charts

    assert charts.day_change_heatmap([], {}) is None            # empty universe
    assert charts.day_change_heatmap(None, None) is None        # nothing at all
    # All quotes failed (no numeric change anywhere) -> nothing to shade.
    assert charts.day_change_heatmap(["AAPL", "MSFT"], {"AAPL": None}) is None
    assert charts.day_change_heatmap(["AAPL"], {"AAPL": float("nan")}) is None


def test_live_watch_formatting_helpers():
    # Qt-free module (CI's headless runner can't import PySide6.QtWidgets).
    from src.ui.quote_format import fmt_price, fmt_signed, fmt_volume, fmt_pct

    assert fmt_price(1234.5) == "1,234.50"
    assert fmt_price(None) == "—"
    assert fmt_signed(2.5) == "+2.50"
    assert fmt_signed(-2.5) == "-2.50"
    assert fmt_signed(0.0123, pct=True) == "+1.23%"
    assert fmt_signed(-0.0123, pct=True) == "-1.23%"
    assert fmt_signed(float("nan")) == "—"
    assert fmt_volume(2_500_000) == "2.50M"
    assert fmt_volume(1_200_000_000) == "1.20B"
    assert fmt_volume(None) == "—"
    assert fmt_pct(0.25) == "25.00%"
