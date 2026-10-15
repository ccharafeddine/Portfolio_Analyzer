"""Risk-free rate resolution shared by every metric in a pipeline run.

Source: Yahoo Finance ``^IRX`` (13-week US Treasury bill). Yahoo quotes it as
the **discount yield in percent**; we convert each daily close to a
bond-equivalent yield, ``BEY = 365 * d / (360 - 91 * d)``, and average it over
the analysis window. The result is an annual rate used everywhere (Sharpe,
Sortino MAR, CAPM excess returns, cash sleeve, optimization).

Fallback: if ``^IRX`` cannot be fetched or has no data in the window, the
user-supplied ``risk_free_rate`` from the config is used and a warning is
returned so the UI can say so. Set ``risk_free_source = "manual"`` in the
config to always use the configured rate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Callable, Optional

import pandas as pd

log = logging.getLogger(__name__)

TBILL_TICKER = "^IRX"
TBILL_DAYS = 91


@dataclass
class RiskFreeRate:
    rate: float              # annual, decimal
    source: str              # "^IRX period average", "manual", "fallback (manual)"
    warning: Optional[str] = None


def discount_to_bey(discount_pct: pd.Series) -> pd.Series:
    d = discount_pct.astype(float) / 100.0
    return 365.0 * d / (360.0 - TBILL_DAYS * d)


def resolve_risk_free_rate(
    start: date,
    end: date,
    fallback: float,
    source: str = "tbill",
    fetch: Optional[Callable[..., pd.DataFrame]] = None,
) -> RiskFreeRate:
    if source == "manual":
        return RiskFreeRate(float(fallback), "manual")
    try:
        if fetch is None:
            from src.data.fetcher import fetch_prices as fetch
        df = fetch([TBILL_TICKER], start, end)
        col = df[TBILL_TICKER] if TBILL_TICKER in getattr(df, "columns", []) else None
        series = col.dropna() if col is not None else pd.Series(dtype=float)
        if series.empty:
            raise ValueError("no ^IRX data in the analysis window")
        rate = float(discount_to_bey(series).mean())
        if not (0.0 <= rate <= 0.25):
            raise ValueError(f"implausible T-bill rate {rate:.4f}")
        return RiskFreeRate(rate, f"{TBILL_TICKER} period average")
    except Exception as e:  # network, cache, or data problem -> documented fallback
        msg = (f"Could not load the 13-week T-bill rate ({TBILL_TICKER}): {e}. "
               f"Using the configured risk-free rate {float(fallback):.2%} instead.")
        log.warning(msg)
        return RiskFreeRate(float(fallback), "fallback (manual)", msg)
