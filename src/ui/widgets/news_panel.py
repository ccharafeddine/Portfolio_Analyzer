"""Per-symbol news panel for the Live Market Watch cockpit.

Shows recent headlines for the *currently charted* symbol — the panel follows the
same ticker the price chart is showing. A native, scrollable list (no web view):
each headline is a clickable link that opens in the system browser, with
publisher · relative-time meta and an optional Alpha Vantage sentiment pill.

Self-contained: it owns a background :class:`SymbolNewsWorker` thread and
coalesces rapid symbol changes, mirroring the watchlist/quote-fetch pattern.
"""

from __future__ import annotations

import html as _html
from datetime import datetime, timezone
from typing import Optional

from PySide6.QtCore import Qt, QThread
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from .. import theme
from ..explanations import tooltip_html
from ..worker import SymbolNewsWorker
from .info_label import InfoLabel

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


class NewsPanel(QWidget):
    """A scrollable list of recent headlines for one symbol."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._symbol: Optional[str] = None
        self._items: list = []
        self._thread: Optional[QThread] = None
        self._worker: Optional[SymbolNewsWorker] = None
        self._in_flight = False
        self._pending: Optional[str] = None

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)

        head = QHBoxLayout()
        head.setSpacing(6)
        self._title = QLabel("News")
        self._title.setObjectName("muted")
        head.addWidget(self._title)
        head.addWidget(InfoLabel(tooltip_html("news_feed")))
        head.addStretch(1)
        root.addLayout(head)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.NoFrame)
        self._body = QWidget()
        self._body_layout = QVBoxLayout(self._body)
        self._body_layout.setContentsMargins(0, 0, 0, 0)
        self._body_layout.setSpacing(8)
        self._body_layout.addStretch(1)
        self._scroll.setWidget(self._body)
        root.addWidget(self._scroll, 1)

        self._placeholder = QLabel("Select a symbol to see its news.")
        self._placeholder.setObjectName("muted")
        self._placeholder.setWordWrap(True)
        self._insert_card(self._placeholder)

    # ── Public API ──
    def set_symbol(self, symbol: str) -> None:
        """Load news for ``symbol`` (the charted ticker). Coalesces rapid changes."""
        sym = (symbol or "").strip().upper()
        if not sym or sym == self._symbol:
            return
        self._symbol = sym
        self._title.setText(f"News · {sym}")
        self._show_message(f"Loading news for {sym}…")
        if self._in_flight:
            self._pending = sym
            return
        self._fetch(sym)

    def clear(self) -> None:
        self._symbol = None
        self._items = []
        self._title.setText("News")
        self._show_message("Select a symbol to see its news.")

    # ── Fetch (worker thread) ──
    def _fetch(self, sym: str) -> None:
        self._in_flight = True
        self._thread = QThread(self)
        self._worker = SymbolNewsWorker(sym)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.done.connect(self._on_news)
        self._worker.failed.connect(self._on_failed)
        self._worker.done.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup)
        self._thread.start()

    def _on_news(self, payload) -> None:
        sym, items = payload
        if sym != self._symbol:
            return  # a newer symbol was selected mid-flight; drop this reply
        self._items = items or []
        self._render()

    def _on_failed(self, _message: str) -> None:
        if self._symbol:
            self._show_message(f"Couldn't load news for {self._symbol}.")

    def _cleanup(self) -> None:
        if self._worker is not None:
            self._worker.deleteLater()
        if self._thread is not None:
            self._thread.deleteLater()
        self._worker = None
        self._thread = None
        self._in_flight = False
        if self._pending:
            nxt, self._pending = self._pending, None
            if nxt != self._symbol:
                self._symbol = None  # force set_symbol to re-run
                self.set_symbol(nxt)
            else:
                self._fetch(nxt)

    # ── Rendering ──
    def _clear_cards(self) -> None:
        while self._body_layout.count() > 1:  # keep the trailing stretch
            item = self._body_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

    def _insert_card(self, widget: QWidget) -> None:
        # Insert before the trailing stretch (last item).
        self._body_layout.insertWidget(self._body_layout.count() - 1, widget)

    def _show_message(self, text: str) -> None:
        self._clear_cards()
        lbl = QLabel(text)
        lbl.setObjectName("muted")
        lbl.setWordWrap(True)
        self._insert_card(lbl)

    def _render(self) -> None:
        self._clear_cards()
        if not self._items:
            self._show_message(
                f"No recent news for {self._symbol}." if self._symbol else "No news."
            )
            return
        for item in self._items:
            self._insert_card(self._make_card(item))

    def _make_card(self, item) -> QWidget:
        t = theme.ACTIVE
        card = QFrame()
        card.setObjectName("newsCard")
        card.setStyleSheet(
            f"QFrame#newsCard {{ background:{t.card}; border:1px solid {t.border};"
            f" border-radius:{max(6, t.radius - 2)}px; }}"
        )
        lay = QVBoxLayout(card)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(4)

        title = _html.escape(item.title or "")
        # Only treat http(s) as a clickable link (setOpenExternalLinks hands the
        # href straight to the OS); drop javascript:/data:/file: news URLs.
        raw_url = (item.url or "").strip()
        is_web = raw_url.lower().startswith(("http://", "https://"))
        url = _html.escape(raw_url, quote=True) if is_web else ""
        title_html = (
            f"<a href='{url}' style='color:{t.text};text-decoration:none;"
            f"font-weight:600;'>{title}</a>" if url
            else f"<span style='color:{t.text};font-weight:600;'>{title}</span>"
        )
        title_lbl = QLabel(title_html)
        title_lbl.setWordWrap(True)
        title_lbl.setOpenExternalLinks(True)
        title_lbl.setTextInteractionFlags(Qt.TextBrowserInteraction)
        title_lbl.setToolTip("Open article in your browser")
        lay.addWidget(title_lbl)

        when = _relative_time(getattr(item, "published", None))
        publisher = item.publisher or ""
        meta = "  ·  ".join(b for b in (publisher, when) if b)
        label = getattr(item, "sentiment_label", None)
        if label:
            color = _SENTIMENT_COLORS.get(label.strip().lower(), t.text_muted)
            meta = f"{meta}   " if meta else ""
            meta_html = (
                f"<span style='color:{t.text_muted};font-size:{t.base_pt - 2}px'>{_html.escape(meta)}</span>"
                f"<span style='color:{color};font-size:{t.base_pt - 3}px;font-weight:700'>"
                f"{_html.escape(label.upper())}</span>"
            )
        else:
            meta_html = (
                f"<span style='color:{t.text_muted};font-size:{t.base_pt - 2}px'>"
                f"{_html.escape(meta)}</span>"
            )
        meta_lbl = QLabel(meta_html)
        meta_lbl.setTextFormat(Qt.RichText)
        lay.addWidget(meta_lbl)
        return card

    # ── Lifecycle ──
    def shutdown(self) -> None:
        self._pending = None
        if self._thread is not None and self._thread.isRunning():
            self._thread.quit()
            self._thread.wait(3000)

    def retheme(self) -> None:
        self._render() if self._items else self._show_message(
            f"No recent news for {self._symbol}." if self._symbol
            else "Select a symbol to see its news."
        )
