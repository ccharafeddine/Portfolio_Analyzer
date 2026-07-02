"""Fama-French factor data from the Ken French Data Library.

Downloads the daily research factors (3-factor, 5-factor, momentum) directly from
Dartmouth, parses their fixed CSV format, and caches to parquet — mirroring the price
fetcher's cache pattern. All returns are decimals (the source is in percent).

Best-effort: network failures raise, and callers (the pipeline factor step) wrap this so
the rest of the analysis is unaffected when Dartmouth is unreachable.
"""

from __future__ import annotations

import io
import re
import time
import urllib.request
import zipfile
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

from src.data.fetcher import DEFAULT_CACHE_DIR

_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
_FILES = {
    "ff3": "F-F_Research_Data_Factors_daily_CSV.zip",
    "ff5": "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
    "mom": "F-F_Momentum_Factor_daily_CSV.zip",
}
_TTL_DAYS = 7  # factor history is stable; refresh weekly


def _parse_ff_csv(text: str) -> pd.DataFrame:
    """Parse a Ken French daily CSV (preamble + date-indexed table + footer)."""
    lines = text.splitlines()
    # The real header is the first line that STARTS with a comma (empty date column)
    # and names a factor — this avoids matching prose like "...Momentum Factor..."
    header_idx = next(
        (i for i, ln in enumerate(lines)
         if ln.strip().startswith(",") and re.search(r"Mkt-RF|Mom|SMB|RMW", ln)),
        None,
    )
    if header_idx is None:
        raise ValueError("Fama-French factor header not found.")
    header = [h.strip() for h in lines[header_idx].split(",")]
    body = header[1:] if header and header[0] == "" else header
    cols = [c for c in body if c]  # drop empty names (files can have trailing commas)

    idx, rows = [], []
    for ln in lines[header_idx + 1:]:
        parts = [p.strip() for p in ln.split(",")]
        if not parts or not re.fullmatch(r"\d{8}", parts[0] or ""):
            continue  # skip footer / blank / annual rows
        vals = parts[1:len(cols) + 1]
        if len(vals) != len(cols):
            continue
        try:
            fv = [float(v) / 100.0 for v in vals]
        except ValueError:
            continue
        idx.append(pd.to_datetime(parts[0], format="%Y%m%d"))
        rows.append(fv)
    if not rows:
        raise ValueError("No Fama-French data rows parsed.")
    df = pd.DataFrame(rows, index=pd.DatetimeIndex(idx), columns=cols)
    return df.rename(columns={"Mom": "MOM", "Mom   ": "MOM"})


def _download(key: str, cache_dir: Path) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"ff_{key}.parquet"
    if cache.exists() and (time.time() - cache.stat().st_mtime) < _TTL_DAYS * 86400:
        return pd.read_parquet(cache)

    req = urllib.request.Request(
        _BASE + _FILES[key], headers={"User-Agent": "Mozilla/5.0"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        blob = resp.read()
    zf = zipfile.ZipFile(io.BytesIO(blob))
    text = zf.read(zf.namelist()[0]).decode("latin-1")
    df = _parse_ff_csv(text)
    try:
        df.to_parquet(cache)
    except Exception:
        pass
    return df


def fetch_ff_factors(
    start: date, end: date, model: str, cache_dir: Optional[Path] = None
) -> Optional[pd.DataFrame]:
    """Daily factor returns (decimals) for a model, sliced to [start, end].

    Includes an ``RF`` column. Returns None for an unknown model. ``model`` is one of
    ``FF3``, ``Carhart 4-Factor``, ``FF5``.
    """
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    if model in ("FF3", "Carhart 4-Factor"):
        df = _download("ff3", cache_dir).copy()
        if model == "Carhart 4-Factor":
            mom = _download("mom", cache_dir)
            df = df.join(mom[["MOM"]], how="inner")
    elif model == "FF5":
        df = _download("ff5", cache_dir).copy()
    else:
        return None

    mask = (df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))
    return df.loc[mask]
