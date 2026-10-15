"""Price/profile fetcher that talks to the demo's Cloudflare Worker proxy instead of
yfinance. It is a drop-in for ``src.data.fetcher.fetch_prices``.

Under Pyodide the transport is a synchronous XHR (``pyodide.http.open_url``), which
browsers allow inside a Web Worker; tests inject their own transport.
"""

from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from typing import Callable, Iterable, Optional
from urllib.parse import urlencode

import pandas as pd

log = logging.getLogger(__name__)

MAX_SYMBOLS_PER_CALL = 25  # mirrors the Worker's cap

Transport = Callable[[str], str]


def _pyodide_transport(url: str) -> str:
    from pyodide.http import open_url  # only importable under Pyodide

    return open_url(url).read()


class ProxyClient:
    """Fetch adjusted daily closes and basic profiles through the Worker."""

    def __init__(self, base_url: str = "", transport: Optional[Transport] = None):
        self.base_url = base_url.rstrip("/")
        self.transport = transport or _pyodide_transport
        self._series: dict[tuple[str, str, str], pd.Series] = {}
        self._profiles: dict[str, dict] = {}
        self.calls = 0

    def _get(self, path: str, params: dict) -> dict:
        self.calls += 1
        raw = self.transport(f"{self.base_url}{path}?{urlencode(params)}")
        try:
            data = json.loads(raw)
        except ValueError:
            raise ValueError("Price proxy returned an invalid response (is it reachable?)") from None
        if isinstance(data, dict) and data.get("error") and "data" not in data and "profiles" not in data:
            raise ValueError(f"Price proxy error: {data['error']}")
        return data

    # ── prices ────────────────────────────────────────────────────────────
    def fetch_prices(self, tickers: Iterable[str], start: date, end: date, **_) -> pd.DataFrame:
        tickers = list(dict.fromkeys(str(t) for t in tickers))
        s, e = _iso(start), _iso(end)
        need = [t for t in tickers if (t, s, e) not in self._series]
        errors: dict[str, str] = {}
        for i in range(0, len(need), MAX_SYMBOLS_PER_CALL):
            chunk = need[i:i + MAX_SYMBOLS_PER_CALL]
            data = self._get("/v1/chart", {"symbols": ",".join(chunk), "start": s, "end": e})
            errors.update(data.get("errors") or {})
            for sym, payload in (data.get("data") or {}).items():
                ser = parse_series(payload)
                if ser is not None:
                    self._series[(sym, s, e)] = ser
        cols = {t: self._series[(t, s, e)] for t in tickers if (t, s, e) in self._series}
        missing = sorted(set(tickers) - set(cols))
        if not cols:
            detail = "; ".join(f"{k}: {v}" for k, v in list(errors.items())[:3])
            raise ValueError(f"No price data for {tickers} between {s} and {e}"
                             + (f" ({detail})" if detail else ""))
        prices = pd.DataFrame(cols).sort_index()
        prices = prices.loc[(prices.index >= pd.Timestamp(s)) & (prices.index <= pd.Timestamp(e))]
        prices.index.name = "Date"
        if missing:
            log.warning("No price data for: %s", ", ".join(missing))
            prices.attrs["missing"] = missing
        return prices

    # ── profiles (sector) ─────────────────────────────────────────────────
    def profile(self, symbol: str) -> dict:
        if symbol not in self._profiles:
            data = self._get("/v1/profile", {"symbols": symbol})
            self._profiles[symbol] = (data.get("profiles") or {}).get(symbol) or {}
        return self._profiles[symbol]


def parse_series(payload: dict) -> Optional[pd.Series]:
    """``{"d": ["2024-01-02", ...], "c": [187.1, ...]}`` -> float Series (nulls dropped)."""
    if not payload:
        return None
    d, c = payload.get("d") or [], payload.get("c") or []
    if not d or len(d) != len(c):
        return None
    ser = pd.Series(pd.to_numeric(pd.Series(c, dtype="object"), errors="coerce").to_numpy(dtype=float),
                    index=pd.DatetimeIndex(pd.to_datetime(d)), dtype=float).dropna()
    ser = ser[~ser.index.duplicated(keep="last")].sort_index()
    return ser if len(ser) else None


def _iso(d) -> str:
    if isinstance(d, str):
        return d[:10]
    if hasattr(d, "date") and not isinstance(d, date):
        d = d.date()
    return d.isoformat()


def end_inclusive(end: date) -> date:  # kept for parity with the yfinance fetcher
    return end + timedelta(days=1)
