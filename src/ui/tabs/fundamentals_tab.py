"""Fundamentals tab.

Two clearly separated zones:

- **Company Financials** (top) — a single-holding drill-down chosen with the ticker
  chips in that section's header: a revenue/net-income trend, the income / balance /
  cash-flow history, and analyst estimates. Statements are fetched lazily and cached.
- **Full Portfolio - Individual Asset Comparison** (below) — every holding side by
  side (valuation, profitability, growth, health, dividends, upcoming dates) plus
  visual comparisons.

All from yfinance (FMP DCF when keyed), on background threads; excluded from reports.
"""

from __future__ import annotations

import html as _html
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
from PySide6.QtCore import QThread

from src.charts import plotly_charts as charts

from .. import settings
from ..worker import FundamentalsWorker, StatementsWorker
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


def _big(v) -> str:
    """Signed large-number money format for statement line items."""
    if v is None:
        return "—"
    n, s = abs(v), ("-" if v < 0 else "")
    if n >= 1e12:
        return f"{s}${n / 1e12:.2f}T"
    if n >= 1e9:
        return f"{s}${n / 1e9:.2f}B"
    if n >= 1e6:
        return f"{s}${n / 1e6:.2f}M"
    return f"{s}${n:,.0f}"


def _ratio(v) -> str:
    return "—" if v is None else f"{v:.2f}"


def _pct_frac(v) -> str:
    return "—" if v is None else f"{v * 100:.1f}%"


def _pct_raw(v) -> str:
    return "—" if v is None else f"{v:.2f}%"


def _px(v) -> str:
    return "—" if v is None else f"${v:.2f}"


def _date(s) -> str:
    return s or "—"


def _stmt_val(row_name: str, v) -> str:
    if v is None:
        return "—"
    if "EPS" in row_name:
        return f"${v:.2f}"
    return _big(v)


class FundamentalsTab(RefreshableWebTab):
    def __init__(self) -> None:
        super().__init__()
        self._items: list = []
        self._last_updated: Optional[datetime] = None

        # Per-ticker statements drill-down state.
        self._statements: dict[str, dict] = {}
        self._selected: Optional[str] = None
        self._stmt_fetching: Optional[str] = None
        self._stmt_thread: Optional[QThread] = None
        self._stmt_worker = None

        # In-page ticker chips route through app:// links (see refreshable_tab).
        self._page.app_link_handler = self._on_app_link

    # ── Comparison fetch ──
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

        tickers = [i.ticker for i in self._items]
        if self._selected not in tickers:
            self._selected = tickers[0] if tickers else None
        self.mark_dirty()
        self.ensure_populated()
        if self._selected:
            self._fetch_statements(self._selected)

    # ── Statements (lazy, per ticker) ──
    def _on_app_link(self, url: str) -> None:
        # url is like "app://select/AAPL"
        ticker = url.rstrip("/").rsplit("/", 1)[-1].upper()
        if ticker:
            self._on_ticker_selected(ticker)

    def _on_ticker_selected(self, ticker: str) -> None:
        if not ticker or ticker == self._selected:
            return
        self._selected = ticker
        self.mark_dirty()
        self.ensure_populated()
        if ticker not in self._statements:
            self._fetch_statements(ticker)

    def _fetch_statements(self, ticker: str) -> None:
        if ticker in self._statements or self._stmt_fetching:
            return
        self._stmt_fetching = ticker
        self._stmt_thread = QThread(self)
        self._stmt_worker = StatementsWorker(ticker)
        self._stmt_worker.moveToThread(self._stmt_thread)
        self._stmt_thread.started.connect(self._stmt_worker.run)
        self._stmt_worker.done.connect(self._on_statements)
        self._stmt_worker.failed.connect(self._on_statements_failed)
        self._stmt_worker.done.connect(self._stmt_thread.quit)
        self._stmt_worker.failed.connect(self._stmt_thread.quit)
        self._stmt_thread.finished.connect(self._stmt_worker.deleteLater)
        self._stmt_thread.start()

    def _on_statements(self, payload) -> None:
        ticker, data = payload
        self._statements[ticker] = data or {}
        self._stmt_fetching = None
        if ticker == self._selected:
            self.mark_dirty()
            self.ensure_populated()
        elif self._selected and self._selected not in self._statements:
            self._fetch_statements(self._selected)

    def _on_statements_failed(self, _msg: str) -> None:
        self._stmt_fetching = None

    def shutdown(self) -> None:
        """Stop the background statements-fetch thread cleanly (called on app close)."""
        if self._stmt_thread is not None and self._stmt_thread.isRunning():
            self._stmt_thread.quit()
            self._stmt_thread.wait(3000)

    # ── Rendering ──
    def _populate(self, results) -> None:
        items = self._items
        if not items:
            self.add_heading("Fundamentals", explain="fundamentals")
            msg = ("Fetching company fundamentals for your holdings…" if self._fetching
                   else "No fundamentals yet. Click Refresh, or run an analysis.")
            self.add_interpretation(msg)
            return

        self._render_statements()     # single-company drill-down on top
        self._render_comparison(items)  # full-portfolio comparison below

    # ── Company Financials zone ──
    def _render_statements(self) -> None:
        sel = self._selected
        if not sel:
            return
        company = next((i.name for i in self._items if i.ticker == sel), sel)
        subtitle = f"{company} ({sel})" if company and company != sel else sel
        self.add_zone("Company Financials", subtitle)
        self._add_ticker_chips(sel)

        data = self._statements.get(sel)
        if data is None:
            self.add_heading("Financial Statements", explain="financial_statements")
            self.add_interpretation(f"Loading statements for {sel}…")
            return
        if not any(data.get(k) for k in ("income", "balance", "cashflow", "target", "recommendation")):
            self.add_heading("Financial Statements", explain="financial_statements")
            self.add_interpretation(f"No statement data available for {sel}.")
            return

        inc = data.get("income")
        if inc:
            series = {}
            if "Total Revenue" in inc["rows"]:
                series["Revenue"] = inc["rows"]["Total Revenue"]
            if "Net Income" in inc["rows"]:
                series["Net Income"] = inc["rows"]["Net Income"]
            if series:
                self.add_heading("Revenue & Net Income Trend", explain="financial_statements")
                self.add_chart(
                    charts.statement_trend_chart(inc["periods"], series,
                                                 f"{sel}: Revenue & Net Income"),
                    height=320,
                )

        stmts = [("Income Statement", data.get("income")),
                 ("Balance Sheet", data.get("balance")),
                 ("Cash Flow", data.get("cashflow"))]
        slots = max((1 + len(s["periods"]) for _, s in stmts if s), default=0)
        for title, stmt in stmts:
            self._stmt_table(title, stmt, slots)

        tgt = data.get("target")
        if tgt:
            cur, mean = tgt.get("current"), tgt.get("mean")
            upside = f"{(mean / cur - 1) * 100:+.0f}%" if (mean and cur) else "—"
            self.add_heading("Analyst Price Target")
            self.add_stat_grid([
                ("Current", _px(cur)),
                ("Mean Target", _px(mean)),
                ("High", _px(tgt.get("high"))),
                ("Low", _px(tgt.get("low"))),
                ("Upside", upside),
            ], columns=5)

        rec = data.get("recommendation")
        if rec:
            self.add_heading("Analyst Recommendations")
            self.add_table(pd.DataFrame({
                "Strong Buy": [rec.get("strongBuy", 0)],
                "Buy": [rec.get("buy", 0)],
                "Hold": [rec.get("hold", 0)],
                "Sell": [rec.get("sell", 0)],
                "Strong Sell": [rec.get("strongSell", 0)],
            }))

    def _add_ticker_chips(self, selected: str) -> None:
        chips = []
        for i in self._items:
            cls = "chip active" if i.ticker == selected else "chip"
            chips.append(
                f"<a class='{cls}' href='app://select/{_html.escape(i.ticker)}'>"
                f"{_html.escape(i.ticker)}</a>"
            )
        self.add_html(f"<div class='chips'>{''.join(chips)}</div>")

    def _stmt_table(self, title: str, stmt: Optional[dict], slots: int = 0) -> None:
        if not stmt:
            return
        rows, periods = stmt["rows"], stmt["periods"]
        data = {"Line Item": list(rows.keys())}
        for i, p in enumerate(periods):
            data[p] = [
                _stmt_val(name, vals[i] if i < len(vals) else None)
                for name, vals in rows.items()
            ]
        self.add_heading(title)
        self.add_table(pd.DataFrame(data), slots=(slots or None))

    # ── Full-portfolio comparison zone ──
    def _render_comparison(self, items) -> None:
        self.add_zone("Full Portfolio - Individual Asset Comparison")
        tk = [i.ticker for i in items]

        # Build every matrix first so we can size all columns to the widest one.
        tables: list[tuple[str, Optional[str], pd.DataFrame]] = [
            ("Valuation", "fundamentals", pd.DataFrame({
                "Ticker": tk,
                "P/E": [_ratio(i.pe) for i in items],
                "Fwd P/E": [_ratio(i.forward_pe) for i in items],
                "P/B": [_ratio(i.pb) for i in items],
                "P/S": [_ratio(i.ps) for i in items],
                "PEG": [_ratio(i.peg) for i in items],
                "EV/EBITDA": [_ratio(i.ev_ebitda) for i in items],
            })),
            ("Profitability", None, pd.DataFrame({
                "Ticker": tk,
                "Gross Margin": [_pct_frac(i.gross_margin) for i in items],
                "Operating Margin": [_pct_frac(i.operating_margin) for i in items],
                "Net Margin": [_pct_frac(i.net_margin) for i in items],
                "ROE": [_pct_frac(i.roe) for i in items],
                "ROA": [_pct_frac(i.roa) for i in items],
            })),
            ("Growth & Financial Health", None, pd.DataFrame({
                "Ticker": tk,
                "Revenue Growth": [_pct_frac(i.revenue_growth) for i in items],
                "Earnings Growth": [_pct_frac(i.earnings_growth) for i in items],
                "Debt/Equity": [_ratio(i.debt_to_equity) for i in items],
                "Current Ratio": [_ratio(i.current_ratio) for i in items],
            })),
            ("Dividends & Profile", None, pd.DataFrame({
                "Ticker": tk,
                "Div Yield": [_pct_raw(i.dividend_yield) for i in items],
                "Payout": [_pct_frac(i.payout_ratio) for i in items],
                "Beta": [_ratio(i.beta) for i in items],
                "Market Cap": [_money(i.market_cap) for i in items],
                "Sector": [i.sector or "—" for i in items],
            })),
        ]
        if any(i.dcf for i in items):
            tables.append(("Fair Value (DCF)", "fundamentals", pd.DataFrame({
                "Ticker": tk,
                "Price": [_px(i.price) for i in items],
                "DCF Value": [_px(i.dcf) for i in items],
                "Upside": [(f"{(i.dcf / i.price - 1) * 100:+.0f}%" if (i.dcf and i.price) else "—")
                           for i in items],
            })))
        tables.append(("Upcoming Earnings & Ex-Dividend Dates", None, pd.DataFrame({
            "Ticker": tk,
            "Company": [i.name or i.ticker for i in items],
            "Next Earnings": [_date(i.next_earnings) for i in items],
            "Ex-Dividend": [_date(i.ex_dividend) for i in items],
        })))

        slots = max(df.shape[1] for _, _, df in tables)
        for heading, explain, df in tables:
            self.add_heading(heading, explain=explain)
            self.add_table(df, slots=slots)

        if any(i.net_margin is not None for i in items) or any(i.pe is not None for i in items):
            self.add_heading("Visual Comparison")
            self.add_chart_row([
                charts.metric_bar_chart(tk, [i.net_margin for i in items],
                                        "Net Profit Margin", pct=True),
                charts.metric_bar_chart(tk, [i.pe for i in items], "P/E Ratio"),
            ], height=320)
