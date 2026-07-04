"""The Live Market Watch right-hand cockpit: price chart + news + a heatmap.

Both the Portfolio and Watchlist tabs show the same stacked cockpit, laid out as
a resizable vertical :class:`QSplitter` of three individually toggleable panels:

- a **TradingView-style candlestick chart** (``"chart"``) with timeframe buttons
  (1D…5Y), driven by the selected symbol (:class:`LightweightChartWidget`);
- a **per-symbol news panel** (``"news"``) that follows the charted symbol; and
- a **day-change heatmap** (``"heatmap"``) — a weight-sized *treemap* on the
  Portfolio tab, or an equal-tile *grid* on the Watchlist tab (``heatmap_style``).

Selecting a symbol (:meth:`load_symbol`, wired to a table row-click) fetches its
OHLC off the UI thread and points the news panel at it. The host (LiveWatchView)
persists the splitter geometry via :meth:`splitter` and toggles panels via
:meth:`set_panel_visible`.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, QThread
from PySide6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from src.charts import plotly_charts as charts
from src.data.market_data import DEFAULT_TIMEFRAME, OHLC_TIMEFRAMES

from .. import theme
from ..explanations import tooltip_html
from ..worker import OhlcWorker
from .info_label import InfoLabel
from .lightweight_chart import LightweightChartWidget
from .news_panel import NewsPanel
from .plotly_widget import PlotlyWidget

_HINT_IDLE = "Click a holding to see its chart"

# Panel keys shared with LiveWatchView's toggles / persisted visibility.
PANEL_CHART = "chart"
PANEL_NEWS = "news"
PANEL_HEATMAP = "heatmap"


class ChartHeatmapPanel(QWidget):
    """Candlestick chart + per-symbol news + a day-change heatmap, self-managing
    its OHLC fetch. Its three sub-panels live in a vertical splitter and can be
    shown or hidden individually. ``heatmap_style`` is ``"treemap"`` (weight-sized)
    or ``"grid"`` (equal tiles)."""

    def __init__(self, heatmap_style: str = "treemap", parent=None) -> None:
        super().__init__(parent)
        self._heatmap_style = heatmap_style if heatmap_style in ("treemap", "grid") else "treemap"
        self._quotes: dict = {}
        self._order: list[str] = []
        self._weights: dict[str, float] = {}

        # Selection + OHLC fetch state (self-managed background fetch).
        self._selected: Optional[str] = None
        self._timeframe: str = DEFAULT_TIMEFRAME
        self._ohlc_df = None
        self._ohlc_thread: Optional[QThread] = None
        self._ohlc_worker: Optional[OhlcWorker] = None
        self._ohlc_fetching = False
        self._ohlc_pending = False  # a re-fetch is wanted once the current ends

        cp = QVBoxLayout(self)
        cp.setContentsMargins(0, 0, 0, 0)
        cp.setSpacing(6)

        self._splitter = QSplitter(Qt.Vertical)
        self._splitter.setChildrenCollapsible(False)

        # ── Chart panel: timeframe buttons + hint, then the candlestick chart ──
        self._chart_box = QWidget()
        cb = QVBoxLayout(self._chart_box)
        cb.setContentsMargins(0, 0, 0, 0)
        cb.setSpacing(4)
        bar = QHBoxLayout()
        bar.setSpacing(4)
        self._chart_hint = QLabel(_HINT_IDLE)
        self._chart_hint.setObjectName("muted")
        bar.addWidget(self._chart_hint)
        bar.addStretch(1)
        self._tf_group = QButtonGroup(self)
        self._tf_group.setExclusive(True)
        self._tf_buttons: dict[str, QPushButton] = {}
        for tf in OHLC_TIMEFRAMES:
            b = QPushButton(tf)
            b.setCheckable(True)
            b.setCursor(Qt.PointingHandCursor)
            b.setChecked(tf == self._timeframe)
            b.clicked.connect(lambda _=False, t=tf: self._on_timeframe(t))
            self._tf_group.addButton(b)
            self._tf_buttons[tf] = b
            bar.addWidget(b)
        cb.addLayout(bar)
        self._chart = LightweightChartWidget()
        cb.addWidget(self._chart, 1)

        # ── News panel (follows the charted symbol) ──
        self._news = NewsPanel()

        # ── Heatmap panel: titled header (+ "?") over the chart ──
        self._heatmap_box = QWidget()
        hb = QVBoxLayout(self._heatmap_box)
        hb.setContentsMargins(0, 0, 0, 0)
        hb.setSpacing(4)
        hhead = QHBoxLayout()
        hhead.setSpacing(6)
        is_grid = self._heatmap_style == "grid"
        self._hm_title = QLabel("Day-Change Heatmap" if is_grid else "Holdings Heatmap")
        self._hm_title.setObjectName("muted")
        hhead.addWidget(self._hm_title)
        hhead.addWidget(InfoLabel(tooltip_html("day_change_heatmap")))
        hhead.addStretch(1)
        self._heatmap = PlotlyWidget()
        hb.addLayout(hhead)
        hb.addWidget(self._heatmap, 1)

        for w in (self._chart_box, self._news, self._heatmap_box):
            self._splitter.addWidget(w)
        self._splitter.setStretchFactor(0, 4)
        self._splitter.setStretchFactor(1, 2)
        self._splitter.setStretchFactor(2, 2)
        cp.addWidget(self._splitter, 1)

        self._style_timeframe_buttons()

    # ── Panel visibility + splitter access (driven by LiveWatchView) ──
    def splitter(self) -> QSplitter:
        return self._splitter

    def set_panel_visible(self, key: str, visible: bool) -> None:
        w = {
            PANEL_CHART: self._chart_box,
            PANEL_NEWS: self._news,
            PANEL_HEATMAP: self._heatmap_box,
        }.get(key)
        if w is not None:
            w.setVisible(bool(visible))

    def selected(self) -> Optional[str]:
        return self._selected

    # ── Data feed ──
    def set_data(self, order, weights, quotes) -> None:
        """Refresh the heatmap from a new snapshot and auto-load the first symbol's
        chart on the first snapshot. ``weights`` may be empty (equal treemap tiles).
        Re-renders the current chart so its prev-close line tracks the latest quote."""
        self._quotes = dict(quotes or {})
        self._order = list(order or [])
        self._weights = dict(weights or {})
        self._render_heatmap()
        if self._selected is None and self._order:
            self.load_symbol(self._order[0])
        elif self._selected is not None and self._ohlc_df is not None:
            self._render_chart()

    def _render_heatmap(self) -> None:
        order = self._order or list(self._quotes.keys())
        changes = {t: getattr(self._quotes.get(t), "change_pct", None) for t in order}
        if self._heatmap_style == "grid":
            self._heatmap.set_figure(charts.day_change_heatmap(order, changes))
        else:
            weights = self._weights or {t: 1.0 for t in order}  # equal tiles if none
            self._heatmap.set_figure(charts.holdings_treemap(order, weights, changes))

    # ── Symbol selection → chart + news ──
    def load_symbol(self, symbol: str) -> None:
        symbol = (symbol or "").strip().upper()
        if not symbol:
            return
        self._selected = symbol
        self._chart_hint.setText(f"{symbol} · {self._timeframe}")
        self._news.set_symbol(symbol)
        self._fetch_ohlc()

    def _on_timeframe(self, tf: str) -> None:
        if tf not in OHLC_TIMEFRAMES:
            return
        self._timeframe = tf
        self._style_timeframe_buttons()
        if self._selected:
            self._chart_hint.setText(f"{self._selected} · {tf}")
            self._fetch_ohlc()

    def _fetch_ohlc(self) -> None:
        if not self._selected:
            return
        if self._ohlc_fetching:
            self._ohlc_pending = True  # coalesce rapid symbol/timeframe clicks
            return
        self._ohlc_fetching = True
        self._ohlc_thread = QThread(self)
        self._ohlc_worker = OhlcWorker(self._selected, self._timeframe)
        self._ohlc_worker.moveToThread(self._ohlc_thread)
        self._ohlc_thread.started.connect(self._ohlc_worker.run)
        self._ohlc_worker.done.connect(self._on_ohlc)
        self._ohlc_worker.failed.connect(self._on_ohlc_failed)
        self._ohlc_worker.done.connect(self._ohlc_thread.quit)
        self._ohlc_worker.failed.connect(self._ohlc_thread.quit)
        self._ohlc_thread.finished.connect(self._cleanup_ohlc_thread)
        self._ohlc_thread.start()

    def _on_ohlc(self, payload) -> None:
        ticker, timeframe, df = payload
        # Drop a stale reply if the user has since changed symbol or timeframe.
        if ticker != self._selected or timeframe != self._timeframe:
            return
        self._ohlc_df = df
        self._render_chart()

    def _on_ohlc_failed(self, _message: str) -> None:
        pass  # best-effort; the chart keeps its last content

    def _render_chart(self) -> None:
        prev = getattr(self._quotes.get(self._selected), "prev_close", None)
        self._chart.set_ohlc(self._selected, self._ohlc_df, self._timeframe, prev_close=prev)

    def _cleanup_ohlc_thread(self) -> None:
        if self._ohlc_worker is not None:
            self._ohlc_worker.deleteLater()
        if self._ohlc_thread is not None:
            self._ohlc_thread.deleteLater()
        self._ohlc_worker = None
        self._ohlc_thread = None
        self._ohlc_fetching = False
        if self._ohlc_pending:
            self._ohlc_pending = False
            self._fetch_ohlc()

    # ── Timeframe button styling ──
    def _style_timeframe_buttons(self) -> None:
        t = theme.ACTIVE
        qss = (
            f"QPushButton {{ background:transparent; border:1px solid {t.border_light};"
            f" color:{t.text_muted}; padding:1px 8px; border-radius:{max(3, t.radius - 6)}px;"
            f" font-size:{t.label_pt}px; font-weight:600; }}"
            f"QPushButton:hover {{ color:{t.text}; border-color:{t.accent}; }}"
            f"QPushButton:checked {{ background:{t.accent}; color:{t.accent_text};"
            f" border-color:{t.accent}; }}"
        )
        for b in self._tf_buttons.values():
            b.setStyleSheet(qss)

    # ── Lifecycle ──
    def reset(self) -> None:
        """Forget the charted symbol so the next snapshot auto-loads the new
        universe's first entry; clear the chart, news, and hint."""
        self._selected = None
        self._ohlc_df = None
        self._ohlc_pending = False
        self._chart.clear()
        self._news.clear()
        self._chart_hint.setText(_HINT_IDLE)

    def shutdown(self) -> None:
        self._ohlc_pending = False
        if self._ohlc_thread is not None and self._ohlc_thread.isRunning():
            self._ohlc_thread.quit()
            self._ohlc_thread.wait(3000)
        self._news.shutdown()

    def retheme(self) -> None:
        """Rebuild the heatmap + chart with the fresh palette from stored state."""
        if self._order or self._quotes:
            self._render_heatmap()
        self._style_timeframe_buttons()
        self._chart.retheme()
        self._news.retheme()
