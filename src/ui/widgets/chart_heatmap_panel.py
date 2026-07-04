"""Reusable intraday-chart + heatmaps panel for Live Market Watch.

Both the Portfolio and Watchlist tabs of the Live Market Watch section show the
same right-hand cockpit, laid out as a resizable vertical :class:`QSplitter` of
three individually toggleable panels:

- a click-through **intraday chart** (``"price"``),
- a weight-sized **holdings treemap** shaded by day change (``"treemap"``), and
- an equal-tile **day-change heatmap grid** (``"daychange"``).

This widget encapsulates all three, including the background ``IntradayWorker``
thread that fetches a symbol's 1-minute frame off the UI thread. Feed it
snapshots via :meth:`set_data`; drive the chart from a table's row-click via
:meth:`load_intraday`. The host (LiveWatchView) persists the splitter geometry
via :meth:`splitter` and toggles panels via :meth:`set_panel_visible`.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, QThread
from PySide6.QtWidgets import QHBoxLayout, QLabel, QSplitter, QVBoxLayout, QWidget

from src.charts import plotly_charts as charts

from ..explanations import tooltip_html
from ..worker import IntradayWorker
from .info_label import InfoLabel
from .plotly_widget import PlotlyWidget

_HINT_IDLE = "Click a holding to see its intraday chart"

# Panel keys shared with LiveWatchView's toggles / persisted visibility.
PANEL_PRICE = "price"
PANEL_TREEMAP = "treemap"
PANEL_DAYCHANGE = "daychange"


class ChartHeatmapPanel(QWidget):
    """Intraday click-through chart + treemap + day-change grid, self-managing
    its fetch. Its three sub-panels live in a vertical splitter and can be shown
    or hidden individually."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._quotes: dict = {}
        self._order: list[str] = []
        self._weights: dict[str, float] = {}

        # Intraday click-through state (self-managed background fetch).
        self._selected: Optional[str] = None
        self._intraday_df = None
        self._intr_thread: Optional[QThread] = None
        self._intr_worker: Optional[IntradayWorker] = None
        self._intr_fetching = False
        self._intr_pending: Optional[str] = None

        cp = QVBoxLayout(self)
        cp.setContentsMargins(0, 0, 0, 0)
        cp.setSpacing(6)

        # A vertical splitter so the three stacked panels are user-resizable.
        self._splitter = QSplitter(Qt.Vertical)
        self._splitter.setChildrenCollapsible(False)

        # ── Price panel: hint + intraday chart (hidden together) ──
        self._price_box = QWidget()
        pb = QVBoxLayout(self._price_box)
        pb.setContentsMargins(0, 0, 0, 0)
        pb.setSpacing(4)
        self._chart_hint = QLabel(_HINT_IDLE)
        self._chart_hint.setObjectName("muted")
        self._intraday = PlotlyWidget()
        pb.addWidget(self._chart_hint)
        pb.addWidget(self._intraday, 1)

        # ── Treemap panel ──
        self._treemap = PlotlyWidget()

        # ── Day-change heatmap grid: titled header (+ "?") over the chart ──
        self._daychange_box = QWidget()
        db = QVBoxLayout(self._daychange_box)
        db.setContentsMargins(0, 0, 0, 0)
        db.setSpacing(4)
        head = QHBoxLayout()
        head.setSpacing(6)
        self._dc_title = QLabel("Day-Change Heatmap")
        self._dc_title.setObjectName("muted")
        head.addWidget(self._dc_title)
        head.addWidget(InfoLabel(tooltip_html("day_change_heatmap")))
        head.addStretch(1)
        self._daychange = PlotlyWidget()
        db.addLayout(head)
        db.addWidget(self._daychange, 1)

        for w in (self._price_box, self._treemap, self._daychange_box):
            self._splitter.addWidget(w)
        self._splitter.setStretchFactor(0, 3)
        self._splitter.setStretchFactor(1, 2)
        self._splitter.setStretchFactor(2, 2)
        cp.addWidget(self._splitter, 1)

    # ── Panel visibility + splitter access (driven by LiveWatchView) ──
    def splitter(self) -> QSplitter:
        return self._splitter

    def set_panel_visible(self, key: str, visible: bool) -> None:
        w = {
            PANEL_PRICE: self._price_box,
            PANEL_TREEMAP: self._treemap,
            PANEL_DAYCHANGE: self._daychange_box,
        }.get(key)
        if w is not None:
            w.setVisible(bool(visible))

    # ── Data feed ──
    def set_data(self, order, weights, quotes) -> None:
        """Refresh the treemap + day-change grid from a new snapshot and auto-load
        the first symbol's intraday chart on the first snapshot. ``weights`` may be
        empty (equal treemap tiles). Re-renders the current intraday chart so its
        prev-close line tracks the latest quote."""
        self._quotes = dict(quotes or {})
        self._order = list(order or [])
        self._weights = dict(weights or {})
        self._render_treemap()
        self._render_daychange()
        if self._selected is None and self._order:
            self.load_intraday(self._order[0])
        elif self._selected is not None and self._intraday_df is not None:
            self._render_intraday(self._selected, self._intraday_df)

    def _render_treemap(self) -> None:
        order = self._order or list(self._quotes.keys())
        changes = {t: getattr(self._quotes.get(t), "change_pct", None) for t in order}
        weights = self._weights or {t: 1.0 for t in order}  # equal tiles if none
        self._treemap.set_figure(charts.holdings_treemap(order, weights, changes))

    def _render_daychange(self) -> None:
        order = self._order or list(self._quotes.keys())
        changes = {t: getattr(self._quotes.get(t), "change_pct", None) for t in order}
        self._daychange.set_figure(charts.day_change_heatmap(order, changes))

    def selected(self) -> Optional[str]:
        """The symbol currently charted, if any."""
        return self._selected

    # ── Intraday click-through ──
    def load_intraday(self, symbol: str) -> None:
        symbol = (symbol or "").strip().upper()
        if not symbol:
            return
        self._selected = symbol
        self._chart_hint.setText(f"Intraday · {symbol}")
        if self._intr_fetching:
            self._intr_pending = symbol  # coalesce rapid clicks
            return
        self._intr_fetching = True
        self._intr_thread = QThread(self)
        self._intr_worker = IntradayWorker(symbol)
        self._intr_worker.moveToThread(self._intr_thread)
        self._intr_thread.started.connect(self._intr_worker.run)
        self._intr_worker.done.connect(self._on_intraday)
        self._intr_worker.failed.connect(self._on_intraday_failed)
        self._intr_worker.done.connect(self._intr_thread.quit)
        self._intr_worker.failed.connect(self._intr_thread.quit)
        self._intr_thread.finished.connect(self._cleanup_intraday_thread)
        self._intr_thread.start()

    def _on_intraday(self, payload) -> None:
        ticker, df = payload
        self._intraday_df = df
        self._render_intraday(ticker, df)

    def _on_intraday_failed(self, _message: str) -> None:
        pass  # best-effort; the chart keeps its last content

    def _render_intraday(self, ticker: str, df) -> None:
        prev = getattr(self._quotes.get(ticker), "prev_close", None)
        fig = charts.intraday_chart(df, ticker=ticker, prev_close=prev) if df is not None else None
        self._intraday.set_figure(fig)

    def _cleanup_intraday_thread(self) -> None:
        if self._intr_worker is not None:
            self._intr_worker.deleteLater()
        if self._intr_thread is not None:
            self._intr_thread.deleteLater()
        self._intr_worker = None
        self._intr_thread = None
        self._intr_fetching = False
        if self._intr_pending:
            nxt, self._intr_pending = self._intr_pending, None
            self.load_intraday(nxt)

    # ── Lifecycle ──
    def reset(self) -> None:
        """Forget the charted symbol so the next snapshot auto-loads the new
        universe's first entry; clear the intraday chart + hint."""
        self._selected = None
        self._intraday_df = None
        self._intr_pending = None
        self._intraday.set_figure(None)
        self._chart_hint.setText(_HINT_IDLE)

    def shutdown(self) -> None:
        self._intr_pending = None
        if self._intr_thread is not None and self._intr_thread.isRunning():
            self._intr_thread.quit()
            self._intr_thread.wait(3000)

    def retheme(self) -> None:
        """Rebuild all charts with the fresh chart palette from stored state."""
        if self._order or self._quotes:
            self._render_treemap()
            self._render_daychange()
        if self._selected is not None and self._intraday_df is not None:
            self._render_intraday(self._selected, self._intraday_df)
