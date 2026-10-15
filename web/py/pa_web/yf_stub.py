"""A minimal stand-in for the ``yfinance`` module inside the browser.

The shared analytics only touch yfinance in a few best-effort places: sector
lookups (``Ticker(t).info['sector']``) and fund sector weights (``funds_data``),
both of which already fall back gracefully. This stub answers ``info`` from the
proxy's profile endpoint and makes everything else raise, so those fallbacks run
(e.g. the bundled SPY sector snapshot).
"""

from __future__ import annotations

import sys
import types

_client = None


class _Unavailable(Exception):
    pass


class Ticker:
    def __init__(self, symbol: str):
        self.ticker = symbol

    @property
    def info(self) -> dict:
        if _client is None:
            raise _Unavailable("no proxy client")
        p = _client.profile(self.ticker)
        return {"sector": p.get("sector"), "quoteType": p.get("quoteType"),
                "longName": p.get("name")}

    def __getattr__(self, name):
        raise _Unavailable(f"yfinance.Ticker.{name} is not available in the web demo")


def download(*_a, **_k):
    raise _Unavailable("use pa_web.fetcher.ProxyClient.fetch_prices")


def install(client) -> types.ModuleType:
    """Register this module as ``yfinance`` and bind it to a ProxyClient."""
    global _client
    _client = client
    mod = sys.modules[__name__]
    sys.modules["yfinance"] = mod
    return mod
