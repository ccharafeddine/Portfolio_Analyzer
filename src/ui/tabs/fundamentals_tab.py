"""Fundamentals tab — per-holding company fundamentals.

yfinance baseline (valuation, profitability, growth, health, dividends, upcoming
earnings/ex-dividend dates), optionally enriched with FMP's DCF fair value. Fetched
on a background thread (each run + on demand). Not included in exported reports.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from .. import settings
from ..worker import FundamentalsWorker
from .refreshable_tab import RefreshableWebTab


def _money(v) -> str:
    if v is None:
        return "—"
    a = abs(v)
    if a >= 1e12:
        return f"${v / 1e12:.2f}T"
    if a >= 1e9:
        return f"${v / 1e9:.2f}B"
    if a >= 1e6:
        return f"${v / 1e6:.2f}M"
    return f"${v:,.0f}"


def _ratio(v) -> str:
    return "—" if v is None else f"{v:.2f}"


def _pct_frac(v) -> str:
    """Format a fraction (0.27 -> 27.0%)."""
    return "—" if v is None else f"{v * 100:.1f}%"


def _pct_raw(v) -> str:
    """Format a value already expressed in percent (yfinance dividendYield)."""
    return "—" if v is None else f"{v:.2f}%"


def _date(s) -> str:
    return s or "—"


class FundamentalsTab(RefreshableWebTab):
    def __init__(self) -> None:
        super().__init__()
        self._items: list = []
        self._last_updated: Optional[datetime] = None

    def refresh(self) -> None:
        if self._fetching or not self._tickers:
            return
        self._set_status("Fetching fundamentals…")
        worker = FundamentalsWorker(self._tickers, fmp_key=settings.get_api_key("FMP_API_KEY"))
        self._start(worker, self._on_data)

    def _on_data(self, items) -> None:
        self._items = items or []
        self._last_updated = datetime.now(timezone.utc)
        stamp = self._last_updated.astimezone().strftime("%I:%M %p").lstrip("0")
        self._set_status(f"Updated {stamp}  ·  {len(self._items)} holdings")
        self.mark_dirty()
        self.ensure_populated()

    # ── Rendering ──
    def _populate(self, results) -> None:
        items = self._items
        if not items:
            self.add_heading("Fundamentals", explain="fundamentals")
            msg = ("Fetching company fundamentals for your holdings…" if self._fetching
                   else "No fundamentals yet. Click Refresh, or run an analysis.")
            self.add_interpretation(msg)
            return

        has_dcf = any(i.dcf for i in items)

        val_cols = {
            "Ticker": [i.ticker for i in items],
            "P/E": [_ratio(i.pe) for i in items],
            "Fwd P/E": [_ratio(i.forward_pe) for i in items],
            "P/B": [_ratio(i.pb) for i in items],
            "P/S": [_ratio(i.ps) for i in items],
            "PEG": [_ratio(i.peg) for i in items],
            "EV/EBITDA": [_ratio(i.ev_ebitda) for i in items],
        }
        if has_dcf:
            val_cols["DCF Value"] = [_money(i.dcf) if i.dcf else "—" for i in items]
            val_cols["Upside"] = [
                (f"{(i.dcf / i.price - 1) * 100:+.0f}%" if (i.dcf and i.price) else "—")
                for i in items
            ]
        self.add_heading("Valuation", explain="fundamentals")
        self.add_table(pd.DataFrame(val_cols))

        self.add_heading("Profitability")
        self.add_table(pd.DataFrame({
            "Ticker": [i.ticker for i in items],
            "Gross Margin": [_pct_frac(i.gross_margin) for i in items],
            "Operating Margin": [_pct_frac(i.operating_margin) for i in items],
            "Net Margin": [_pct_frac(i.net_margin) for i in items],
            "ROE": [_pct_frac(i.roe) for i in items],
            "ROA": [_pct_frac(i.roa) for i in items],
        }))

        self.add_heading("Growth & Financial Health")
        self.add_table(pd.DataFrame({
            "Ticker": [i.ticker for i in items],
            "Revenue Growth": [_pct_frac(i.revenue_growth) for i in items],
            "Earnings Growth": [_pct_frac(i.earnings_growth) for i in items],
            "Debt/Equity": [_ratio(i.debt_to_equity) for i in items],
            "Current Ratio": [_ratio(i.current_ratio) for i in items],
        }))

        self.add_heading("Dividends & Profile")
        self.add_table(pd.DataFrame({
            "Ticker": [i.ticker for i in items],
            "Div Yield": [_pct_raw(i.dividend_yield) for i in items],
            "Payout": [_pct_frac(i.payout_ratio) for i in items],
            "Beta": [_ratio(i.beta) for i in items],
            "Market Cap": [_money(i.market_cap) for i in items],
            "Sector": [i.sector or "—" for i in items],
        }))

        self.add_heading("Upcoming Earnings & Ex-Dividend Dates")
        self.add_table(pd.DataFrame({
            "Ticker": [i.ticker for i in items],
            "Company": [i.name or i.ticker for i in items],
            "Next Earnings": [_date(i.next_earnings) for i in items],
            "Ex-Dividend": [_date(i.ex_dividend) for i in items],
        }))
