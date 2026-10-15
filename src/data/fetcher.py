"""
Data fetching layer — yfinance only, with local parquet caching.

Design:
- yfinance handles stocks, ETFs, crypto (BTC-USD), and indices (^GSPC) natively
- Local parquet cache avoids re-downloading on every run
- Cache is date-range aware: fetches only missing data
- Clean, typed API surface for the pipeline
"""

from __future__ import annotations

import hashlib
import os
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

import logging

log = logging.getLogger(__name__)
import yfinance as yf


# ──────────────────────────────────────────────────────────────
# Cache config
# ──────────────────────────────────────────────────────────────

# The desktop app sets PORTFOLIO_ANALYZER_CACHE_DIR (via src.ui.paths) so the
# GUI and CLI share one cache location. Falls back to the original home-dir
# location when unset, keeping standalone/CLI behavior unchanged.
DEFAULT_CACHE_DIR = Path(
    os.getenv("PORTFOLIO_ANALYZER_CACHE_DIR")
    or (Path.home() / ".portfolio_analyzer_cache")
)
CACHE_TTL_DAYS = 1  # Re-fetch if data is older than this


def _cache_key(tickers: list[str], start: date, end: date) -> str:
    """Deterministic cache key from the request parameters."""
    raw = f"{'|'.join(sorted(tickers))}_{start}_{end}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def _cache_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / f"prices_{key}.parquet"


def _is_cache_fresh(path: Path, ttl_days: int = CACHE_TTL_DAYS) -> bool:
    """Check if cached file exists and is younger than TTL."""
    if not path.exists():
        return False
    import time

    age_seconds = time.time() - path.stat().st_mtime
    return age_seconds < (ttl_days * 86400)


# ──────────────────────────────────────────────────────────────
# Core fetcher
# ──────────────────────────────────────────────────────────────


def fetch_prices(
    tickers: list[str],
    start: date,
    end: date,
    cache_dir: Optional[Path] = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Download adjusted close prices for a list of tickers via yfinance.

    Returns a DataFrame with DatetimeIndex and one column per ticker.
    Uses local parquet cache to avoid redundant API calls.

    Parameters
    ----------
    tickers : list[str]
        Ticker symbols (e.g., ["AAPL", "BTC-USD", "^GSPC"]).
    start : date
        Start date (inclusive).
    end : date
        End date (inclusive). We add 1 day for yfinance's exclusive end.
    cache_dir : Path, optional
        Cache directory. Defaults to ~/.portfolio_analyzer_cache.
    force_refresh : bool
        If True, skip cache and re-download.

    Returns
    -------
    pd.DataFrame
        Adjusted close prices, columns = tickers, index = Date.

    Raises
    ------
    ValueError
        If no data could be fetched for any ticker.
    """
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    key = _cache_key(tickers, start, end)
    path = _cache_path(cache_dir, key)

    # Try cache first
    if not force_refresh and _is_cache_fresh(path):
        try:
            cached = pd.read_parquet(path)
            # Verify all requested tickers are present
            if set(tickers).issubset(set(cached.columns)):
                return cached[tickers].sort_index()
        except Exception as e:  # cache corrupted -> re-download
            log.warning("Price cache unreadable (%s); re-downloading", e)

    # Fetch from yfinance
    # yfinance end is exclusive, so add 1 day
    end_fetch = end + timedelta(days=1)

    try:
        raw = yf.download(
            tickers,
            start=start.isoformat(),
            end=end_fetch.isoformat(),
            auto_adjust=True,
            progress=False,
            threads=True,
        )
    except Exception as e:  # network down, DNS, rate limit ...
        log.warning("Price download failed: %s", e)
        raw = pd.DataFrame()

    if raw.empty:
        stale = _stale_cache(cache_dir, path, tickers, start, end)
        if stale is not None:
            return stale
        raise ValueError(
            f"yfinance returned no data for {tickers} "
            f"between {start} and {end}, and no cached copy is available. "
            "Check the tickers, the dates and your internet connection."
        )

    # yf.download returns MultiIndex columns for multiple tickers:
    # level 0 = Price type (Close, High, etc.), level 1 = Ticker
    # For single ticker it returns flat columns.
    if isinstance(raw.columns, pd.MultiIndex):
        # Extract Close prices
        if "Close" in raw.columns.get_level_values(0):
            prices = raw["Close"]
        else:
            # Fallback: take first price level
            prices = raw.iloc[:, raw.columns.get_level_values(0) == raw.columns.get_level_values(0)[0]]
            prices.columns = prices.columns.droplevel(0)
    else:
        # Single ticker case
        prices = raw[["Close"]].copy()
        prices.columns = tickers[:1]

    # Ensure column names match requested tickers
    prices.columns = [str(c) for c in prices.columns]
    prices.index.name = "Date"
    prices = prices.sort_index()

    # Report missing tickers
    fetched = set(prices.columns)
    requested = set(tickers)
    missing = requested - fetched
    if missing:
        log.warning("No price data for: %s", ", ".join(sorted(missing)))
        prices.attrs["missing"] = sorted(missing)

    if prices.empty:
        raise ValueError("No price data returned after processing.")

    # Cache the result
    try:
        prices.to_parquet(path)
    except Exception as e:
        log.warning("Price cache write failed (non-fatal): %s", e)

    return prices


def _stale_cache(cache_dir: Path, path: Path, tickers: list[str], start: date,
                 end: date) -> Optional[pd.DataFrame]:
    """Offline fallback: the exact cached request regardless of age, else any
    cached frame that holds every ticker and covers the window. The result is
    tagged ``attrs['offline'] = True`` and ``attrs['cache_age_days']``."""
    import time

    candidates = [path] if path.exists() else []
    candidates += sorted((p for p in cache_dir.glob("prices_*.parquet") if p != path),
                         key=lambda p: -p.stat().st_mtime)
    for p in candidates:
        try:
            df = pd.read_parquet(p)
        except Exception:
            continue
        if not set(tickers).issubset(df.columns):
            continue
        df = df.sort_index()
        df = df.loc[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end)), tickers]
        if p != path and (df.empty or df.index[0] > pd.Timestamp(start) + pd.Timedelta(days=7)):
            continue
        if df.empty:
            continue
        df = df.copy()
        df.attrs["offline"] = True
        df.attrs["cache_age_days"] = round((time.time() - p.stat().st_mtime) / 86400, 1)
        log.warning("Offline: using cached prices from %s (%.1f days old)", p.name,
                    df.attrs["cache_age_days"])
        return df
    return None


def fetch_single_ticker(
    ticker: str,
    start: date,
    end: date,
    cache_dir: Optional[Path] = None,
) -> pd.Series:
    """Convenience wrapper for a single ticker. Returns a Series."""
    df = fetch_prices([ticker], start, end, cache_dir=cache_dir)
    if ticker not in df.columns:
        raise ValueError(f"No data for {ticker}")
    s = df[ticker].dropna()
    s.name = ticker
    return s


# ──────────────────────────────────────────────────────────────
# Cache management
# ──────────────────────────────────────────────────────────────


def clear_cache(cache_dir: Optional[Path] = None) -> int:
    """Remove all cached parquet files. Returns count of files removed."""
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR
    if not cache_dir.exists():
        return 0
    count = 0
    for f in cache_dir.glob("prices_*.parquet"):
        f.unlink()
        count += 1
    return count


def cache_info(cache_dir: Optional[Path] = None) -> dict:
    """Return info about the current cache state."""
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR
    if not cache_dir.exists():
        return {"files": 0, "total_size_mb": 0.0, "path": str(cache_dir)}

    files = list(cache_dir.glob("prices_*.parquet"))
    total_bytes = sum(f.stat().st_size for f in files)
    return {
        "files": len(files),
        "total_size_mb": round(total_bytes / (1024 * 1024), 2),
        "path": str(cache_dir),
    }
