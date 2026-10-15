"""Small on-disk cache for slow per-ticker metadata (sector from yfinance
``Ticker.info``). One JSON file in the price cache directory; entries expire
after ``TTL_DAYS``. Network failures fall back to the cached value, then to
``None``."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

from src.data.fetcher import DEFAULT_CACHE_DIR

log = logging.getLogger(__name__)
TTL_DAYS = 30


def _path(cache_dir: Optional[Path]) -> Path:
    return (cache_dir or DEFAULT_CACHE_DIR) / "ticker_meta.json"


def _load(p: Path) -> dict:
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def get_sectors(tickers, cache_dir: Optional[Path] = None, fetch=None) -> dict[str, Optional[str]]:
    """{ticker: sector or None}. ``fetch(ticker) -> dict`` defaults to yfinance info."""
    p = _path(cache_dir)
    db = _load(p)
    now = time.time()
    out, dirty = {}, False
    for t in tickers:
        e = db.get(t)
        if e and now - e.get("ts", 0) < TTL_DAYS * 86400:
            out[t] = e.get("sector")
            continue
        try:
            if fetch is None:
                import yfinance as yf
                info = yf.Ticker(t).info or {}
            else:
                info = fetch(t) or {}
            sec = info.get("sector") or None
            db[t] = {"sector": sec, "ts": now}
            dirty = True
            out[t] = sec
        except Exception as ex:
            log.warning("Sector lookup failed for %s: %s", t, ex)
            out[t] = e.get("sector") if e else None
    if dirty:
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(db))
        except Exception as ex:
            log.warning("Metadata cache write failed: %s", ex)
    return out
