"""News tab — recent headlines for the analysis's holdings.

yfinance headlines (no key) enriched with Alpha Vantage sentiment when a key is set.
Fetched on a background thread: on every run and on demand via Refresh. Not included
in exported reports.
"""

from __future__ import annotations

import html as _html
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from .. import settings, theme
from ..worker import NewsWorker
from .refreshable_tab import RefreshableWebTab

_SENTIMENT_COLORS = {
    "bullish": "#10B981",
    "somewhat-bullish": "#34D399",
    "neutral": "#94A3B8",
    "somewhat-bearish": "#F59E0B",
    "bearish": "#EF4444",
}


def _relative_time(dt: Optional[datetime]) -> str:
    if dt is None:
        return ""
    secs = (datetime.now(timezone.utc) - dt).total_seconds()
    if secs < 0:
        return "just now"
    if secs < 3600:
        return f"{int(secs // 60)}m ago"
    if secs < 86400:
        return f"{int(secs // 3600)}h ago"
    if secs < 86400 * 30:
        return f"{int(secs // 86400)}d ago"
    return dt.strftime("%b %d, %Y")


class NewsTab(RefreshableWebTab):
    def __init__(self) -> None:
        super().__init__()
        self._news: list = []
        self._events: list = []
        self._last_updated: Optional[datetime] = None

    def refresh(self) -> None:
        if self._fetching or not self._tickers:
            return
        self._set_status("Fetching latest news…")
        worker = NewsWorker(self._tickers, av_key=settings.get_api_key("ALPHAVANTAGE_API_KEY"))
        self._start(worker, self._on_news)

    def _on_news(self, payload) -> None:
        news, events = payload
        self._news = news or []
        self._events = events or []
        self._last_updated = datetime.now(timezone.utc)
        stamp = self._last_updated.astimezone().strftime("%I:%M %p").lstrip("0")
        self._set_status(f"Updated {stamp}  ·  {len(self._news)} stories")
        self.mark_dirty()
        self.ensure_populated()

    # ── Rendering ──
    def _populate(self, results) -> None:
        self.add_html(self._extra_css())

        # Upcoming earnings / ex-dividend dates across the holdings.
        self.add_heading("Upcoming Earnings & Dividends", explain="earnings_calendar")
        if self._events:
            self.add_table(pd.DataFrame({
                "Date": [self._fmt_event_date(e.date) for e in self._events],
                "Ticker": [e.ticker for e in self._events],
                "Event": [e.kind for e in self._events],
                "Detail": [e.detail or "—" for e in self._events],
            }))
        else:
            self.add_interpretation(
                "No upcoming earnings or ex-dividend dates were found for your holdings."
            )

        self.add_heading("Latest News", explain="news_feed")
        if not self._news:
            if self._fetching:
                self.add_interpretation("Fetching the latest headlines for your holdings…")
            else:
                self.add_interpretation(
                    "No news available yet. Click Refresh, or check your connection. "
                    "Add an Alpha Vantage key in Settings for more articles and sentiment."
                )
            return
        cards = "".join(self._news_card(item) for item in self._news)
        self.add_html(f"<div class='news-list'>{cards}</div>")

    @staticmethod
    def _fmt_event_date(iso: str) -> str:
        try:
            d = datetime.fromisoformat(iso).date()
        except Exception:
            return iso
        days = (d - datetime.now(timezone.utc).date()).days
        rel = "today" if days == 0 else ("tomorrow" if days == 1 else f"in {days}d")
        return f"{d.strftime('%b %d, %Y')}  ({rel})"

    def _news_card(self, item) -> str:
        t = theme.ACTIVE
        title = _html.escape(item.title or "")
        url = _html.escape(item.url or "", quote=True)
        publisher = _html.escape(item.publisher or "")
        summary = _html.escape(item.summary or "")
        when = _relative_time(item.published)
        tickers = ", ".join(_html.escape(x) for x in (item.tickers or []))
        meta = "  ·  ".join(b for b in (publisher, when, tickers) if b)

        pill = ""
        if item.sentiment_label:
            color = _SENTIMENT_COLORS.get(item.sentiment_label.strip().lower(), t.text_muted)
            pill = (
                f"<span class='news-pill' style='color:{color};border-color:{color}'>"
                f"{_html.escape(item.sentiment_label)}</span>"
            )

        title_html = (
            f"<a class='news-title' href='{url}' target='_blank'>{title}</a>"
            if url else f"<span class='news-title'>{title}</span>"
        )
        summary_html = f"<div class='news-summary'>{summary}</div>" if summary else ""
        return (
            "<div class='news-card'>"
            f"<div class='news-head'>{title_html}{pill}</div>"
            f"<div class='news-meta'>{meta}</div>"
            f"{summary_html}"
            "</div>"
        )

    def _extra_css(self) -> str:
        t = theme.ACTIVE
        return f"""<style>
  .news-list{{display:flex;flex-direction:column;gap:10px}}
  .news-card{{background:{t.card};border:1px solid {t.border};
    border-radius:{max(6, t.radius - 2)}px;padding:12px 14px}}
  .news-head{{display:flex;align-items:flex-start;gap:10px;justify-content:space-between}}
  .news-title{{color:{t.text};font-weight:600;font-size:{t.base_pt + 1}px;
    text-decoration:none;line-height:1.35}}
  .news-title:hover{{color:{t.accent};text-decoration:underline}}
  .news-meta{{color:{t.text_muted};font-size:{t.base_pt - 2}px;margin-top:4px}}
  .news-summary{{color:{t.text_slate};font-size:{t.base_pt - 1}px;margin-top:6px;line-height:1.5}}
  .news-pill{{flex:none;font-size:{t.base_pt - 3}px;font-weight:700;text-transform:uppercase;
    letter-spacing:.04em;border:1px solid currentColor;border-radius:999px;padding:2px 8px;
    white-space:nowrap}}
</style>"""
