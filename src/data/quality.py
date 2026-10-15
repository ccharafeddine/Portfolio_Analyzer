"""Price data-quality checks.

``quality_report`` inspects an adjusted-close frame (DatetimeIndex x tickers)
and returns one row per ticker plus a list of human-readable issues:

* **Inception** - first valid price later than the requested start (the
  backtest has to phase the holding in).
* **Stale** - last valid price more than ``stale_days`` business days before
  the frame's last date (delisted, halted, or a feed problem).
* **Gaps** - runs of ``gap_days`` or more consecutive missing prices *inside*
  the ticker's live span (pre-inception NaNs are not gaps).
* **Split suspects** - single-day moves whose price ratio sits within 3% of a
  common split ratio (2, 3, 4, 5, 10, 20 or their inverses) and exceed 45% in
  size. Adjusted prices should never show these, so one means the feed did not
  back-adjust.
* **Flat runs** - ``flat_days`` or more identical consecutive prices
  (often a stale feed carried forward).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

_SPLIT_RATIOS = (2, 3, 4, 5, 10, 20)


@dataclass
class QualityReport:
    table: pd.DataFrame
    issues: list[str] = field(default_factory=list)
    as_of: pd.Timestamp | None = None
    days: int = 0

    @property
    def gaps(self) -> int:
        return int(self.table["Gaps"].sum()) if not self.table.empty else 0

    @property
    def ok(self) -> bool:
        return not self.issues


def _runs(mask: pd.Series) -> list[tuple[int, int]]:
    """(start, length) of True runs."""
    out, start = [], None
    vals = mask.to_numpy()
    for i, v in enumerate(vals):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i - start))
            start = None
    if start is not None:
        out.append((start, len(vals) - start))
    return out


def _split_like(ratio: float, tol: float = 0.03) -> int | None:
    for r in _SPLIT_RATIOS:
        for target in (r, 1.0 / r):
            if abs(ratio / target - 1.0) <= tol:
                return r
    return None


def quality_report(prices: pd.DataFrame, start=None, stale_days: int = 5,
                   gap_days: int = 3, flat_days: int = 10) -> QualityReport:
    if prices is None or prices.empty:
        return QualityReport(pd.DataFrame(columns=["Ticker", "First", "Last", "Missing", "Gaps",
                                                   "Longest gap", "Splits", "Flat run", "Status"]))
    px = prices.sort_index()
    last_date = px.index[-1]
    start = pd.Timestamp(start) if start is not None else px.index[0]
    rows, issues = [], []
    for t in px.columns:
        s = px[t]
        valid = s.dropna()
        if valid.empty:
            rows.append({"Ticker": t, "First": pd.NaT, "Last": pd.NaT, "Missing": len(s),
                         "Gaps": 0, "Longest gap": 0, "Splits": 0, "Flat run": 0,
                         "Status": "No data"})
            issues.append(f"{t}: no price data")
            continue
        first, last = valid.index[0], valid.index[-1]
        span = s.loc[first:last]
        miss = span.isna()
        gap_runs = [n for _, n in _runs(miss) if n >= gap_days]
        ratio = (valid / valid.shift(1)).dropna()
        splits = [(d, _split_like(r)) for d, r in ratio.items()
                  if abs(np.log(r)) > np.log(1.45) and _split_like(r)]
        same = valid.diff().eq(0)
        flat = max((n for _, n in _runs(same)), default=0) + (1 if same.any() else 0)
        stale = int(np.busday_count(last.date(), last_date.date())) if last < last_date else 0
        status = []
        if first > start + pd.Timedelta(days=7):
            status.append("Inception")
            issues.append(f"{t}: history starts {first:%d %b %Y}, after the requested start")
        if stale > stale_days:
            status.append("Stale")
            issues.append(f"{t}: last price {last:%d %b %Y} ({stale} business days stale)")
        if gap_runs:
            status.append("Gaps")
            issues.append(f"{t}: {len(gap_runs)} gap(s) of {gap_days}+ days, longest {max(gap_runs)}")
        if splits:
            status.append("Split?")
            d, r = splits[0]
            issues.append(f"{t}: {len(splits)} unadjusted split-like move(s), e.g. {d:%d %b %Y} ~{r}:1")
        if flat >= flat_days:
            status.append("Flat")
            issues.append(f"{t}: price unchanged for {flat} consecutive days")
        rows.append({"Ticker": t, "First": first, "Last": last, "Missing": int(miss.sum()),
                     "Gaps": len(gap_runs), "Longest gap": max(gap_runs, default=0),
                     "Splits": len(splits), "Flat run": flat,
                     "Status": ", ".join(status) or "OK"})
    return QualityReport(pd.DataFrame(rows), issues, last_date, int(len(px)))
