"""Watchlist store + symbol normalization + name-enriched quotes.

All Qt-free: the store is exercised through an injected dict-backed settings
fake (no QSettings), and yfinance is patched at module level. Mirrors the
existing market-data test style in ``tests/test_quotes.py``.
"""

from types import SimpleNamespace
from unittest.mock import patch

from src.data import market_data
from src.data.market_data import normalize_symbol
from src.ui.watchlist import STARTERS, WatchlistStore


# ── A dict-backed stand-in for QSettings (value/setValue), like test_samples ──
class FakeSettings:
    def __init__(self):
        self.store = {}

    def value(self, key, default=None, type=None):  # noqa: A002 (mirror QSettings)
        return self.store.get(key, default)

    def setValue(self, key, val):
        self.store[key] = val


# ── normalize_symbol ──────────────────────────────────────────────
def test_normalize_symbol_crypto_and_passthrough():
    assert normalize_symbol("btc") == "BTC-USD"        # bare crypto shorthand
    assert normalize_symbol("BTCUSD") == "BTC-USD"     # USD-suffixed shorthand
    assert normalize_symbol("BTC-USD") == "BTC-USD"    # already canonical
    assert normalize_symbol(" eth ") == "ETH-USD"      # trims + maps
    assert normalize_symbol("aapl") == "AAPL"          # equity passthrough (upcased)
    assert normalize_symbol("BRK-B") == "BRK-B"        # dashed equity is not crypto
    assert normalize_symbol("^GSPC") == "^GSPC"        # index passthrough


def test_normalize_symbol_junk_and_empty():
    assert normalize_symbol("") == ""
    assert normalize_symbol("   ") == ""
    assert normalize_symbol("!!!") == ""               # no alphanumerics -> junk
    assert normalize_symbol(None) == ""


# ── Quotes: names + per-symbol failure ────────────────────────────
def _fast_info(**kw):
    return SimpleNamespace(**kw)


def test_fetch_quotes_with_names_populates_name():
    market_data._NAME_CACHE.clear()
    fast = _fast_info(last_price=110.0, previous_close=100.0)
    mock_yf = SimpleNamespace(
        Ticker=lambda t: SimpleNamespace(fast_info=fast, info={"shortName": "Apple Inc."})
    )
    with patch.object(market_data, "yf", mock_yf):
        out = market_data.fetch_quotes(["AAPL"], use_cache=False, with_names=True)
    assert out["AAPL"].ok
    assert out["AAPL"].name == "Apple Inc."


def test_fetch_quotes_with_names_tolerates_per_symbol_failure():
    market_data._NAME_CACHE.clear()

    def ticker(t):
        if t == "GOOD":
            return SimpleNamespace(
                fast_info=_fast_info(last_price=5.0, previous_close=4.0),
                info={"longName": "Good Corp"},
            )
        raise RuntimeError("no such symbol")

    with patch.object(market_data, "yf", SimpleNamespace(Ticker=ticker)):
        out = market_data.fetch_quotes(["GOOD", "BAD"], use_cache=False, with_names=True)

    assert out["GOOD"].ok and out["GOOD"].name == "Good Corp"
    # A bad symbol degrades to a sparse, flagged record — no exception, no name.
    assert out["BAD"].ok is False
    assert out["BAD"].last is None
    assert out["BAD"].name is None
    assert out["BAD"].error  # error flag set


# ── Watchlist store round-trip ────────────────────────────────────
def test_watchlist_add_dedupe_remove_and_persist():
    s = FakeSettings()
    store = WatchlistStore(settings=s, seed=False)
    assert store.symbols() == []

    assert store.add("aapl") == "AAPL"          # normalized + stored
    assert store.add("AAPL") is None            # dedupe (already present)
    assert store.add("btc") == "BTC-USD"        # crypto normalized
    assert store.add("!!!") is None             # junk not persisted
    assert store.symbols() == ["AAPL", "BTC-USD"]

    # Persists: a fresh store over the same settings reloads the list.
    reloaded = WatchlistStore(settings=s, seed=False)
    assert reloaded.symbols() == ["AAPL", "BTC-USD"]

    assert reloaded.remove("aapl") is True      # normalize on remove too
    assert reloaded.remove("aapl") is False     # already gone
    assert reloaded.symbols() == ["BTC-USD"]

    again = WatchlistStore(settings=s, seed=False)
    assert again.symbols() == ["BTC-USD"]


def test_watchlist_reorder():
    s = FakeSettings()
    store = WatchlistStore(settings=s, seed=False)
    for sym in ("AAPL", "MSFT", "NVDA"):
        store.add(sym)
    store.reorder(["nvda", "aapl"])             # partial + un-normalized input
    # Reordered symbols first (normalized), the omitted one kept at the end.
    assert store.symbols() == ["NVDA", "AAPL", "MSFT"]
    assert WatchlistStore(settings=s, seed=False).symbols() == ["NVDA", "AAPL", "MSFT"]


def test_watchlist_seeds_first_run_only():
    s = FakeSettings()
    store = WatchlistStore(settings=s, seed=True)
    assert store.symbols() == STARTERS          # seeded on genuine first run

    # User clears the whole list.
    for sym in list(store.symbols()):
        store.remove(sym)
    assert store.symbols() == []

    # Re-opening must NOT re-seed an existing (now empty) watchlist.
    reopened = WatchlistStore(settings=s, seed=True)
    assert reopened.symbols() == []


def test_watchlist_seed_respects_preexisting_list():
    s = FakeSettings()
    # Simulate a saved watchlist from before the seed flag existed.
    s.setValue("watchlist/symbols", ["TSLA"])
    store = WatchlistStore(settings=s, seed=True)
    assert store.symbols() == ["TSLA"]          # not overwritten by starters
