"""The Live Market Watch cockpit — a drag-and-drop grid of four panels.

Both the Portfolio and Watchlist tabs show the same cockpit, laid out as a
free-form snap grid (:class:`GridDashboard`) of distinct, movable/resizable cards:

- the **quotes table** (passed in by the host — portfolio holdings or the
  watchlist), titled ``"table"``;
- a **TradingView-style candlestick chart** (``"chart"``) with timeframe buttons
  (1D…5Y) in its header, driven by the selected symbol;
- a **per-symbol news panel** (``"news"``) that follows the charted symbol; and
- a **day-change heatmap** (``"heatmap"``) — a weight-sized *treemap* on the
  Portfolio tab, or an equal-tile *grid* on the Watchlist (``heatmap_style``).

Drag a card's header to move it, drag its corner to resize; the arrangement snaps
to the grid, reflows, and persists via ``settings``/``layout_key``. Selecting a
symbol (:meth:`load_symbol`, wired to a table row-click) fetches its OHLC off the
UI thread and points the news panel at it.
"""

from __future__ import annotations

import json
from typing import Optional

from PySide6.QtCore import Qt, QThread
from PySide6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.charts import plotly_charts as charts
from src.data.market_data import DEFAULT_TIMEFRAME, OHLC_TIMEFRAMES

from .. import theme
from ..explanations import tooltip_html
from ..worker import OhlcWorker
from .grid_dashboard import GridDashboard
from .info_label import InfoLabel
from .lightweight_chart import LightweightChartWidget
from .news_panel import NewsPanel
from .plotly_widget import PlotlyWidget

# Panel keys shared with LiveWatchView's toggles / persisted layout.
PANEL_TABLE = "table"
PANEL_CHART = "chart"
PANEL_NEWS = "news"
PANEL_HEATMAP = "heatmap"

# Default grid cells (12 cols × 12 rows): table left full-height, chart top-right,
# news + heatmap along the bottom-right.
_DEFAULT_CELLS = {
    PANEL_TABLE: (0, 0, 4, 12),
    PANEL_CHART: (4, 0, 8, 7),
    PANEL_NEWS: (4, 7, 4, 5),
    PANEL_HEATMAP: (8, 7, 4, 5),
}


class ChartHeatmapPanel(QWidget):
    """Grid cockpit: quotes table + candlestick chart + per-symbol news + heatmap,
    self-managing its OHLC fetch. ``heatmap_style`` is ``"treemap"`` (weight-sized)
    or ``"grid"`` (equal tiles)."""

    def __init__(self, heatmap_style: str = "treemap", table_widget: QWidget = None,
                 table_title: str = "Portfolio", settings=None, layout_key: str = "",
                 parent=None) -> None:
        super().__init__(parent)
        self._heatmap_style = heatmap_style if heatmap_style in ("treemap", "grid") else "treemap"
        self._settings = settings
        self._layout_key = layout_key
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
        self._ohlc_pending = False

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        self._dash = GridDashboard()

        # ── Chart panel content + timeframe buttons (go in the card header) ──
        self._chart = LightweightChartWidget()
        tf_bar = QWidget()
        tb = QHBoxLayout(tf_bar)
        tb.setContentsMargins(0, 0, 0, 0)
        tb.setSpacing(4)
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
            tb.addWidget(b)

        self._news = NewsPanel(show_header=False)
        self._heatmap = PlotlyWidget()

        # ── Add the four cards ──
        c = _DEFAULT_CELLS
        self._table_title = table_title
        if table_widget is not None:
            self._dash.add_panel(PANEL_TABLE, table_title.upper(), table_widget, *c[PANEL_TABLE][2:], x=c[PANEL_TABLE][0], y=c[PANEL_TABLE][1])
        self._chart_panel = self._dash.add_panel(
            PANEL_CHART, "PRICE CHART", self._chart, c[PANEL_CHART][2], c[PANEL_CHART][3],
            x=c[PANEL_CHART][0], y=c[PANEL_CHART][1], header_extra=tf_bar)
        self._news_panel = self._dash.add_panel(
            PANEL_NEWS, "NEWS", self._news, c[PANEL_NEWS][2], c[PANEL_NEWS][3],
            x=c[PANEL_NEWS][0], y=c[PANEL_NEWS][1],
            header_extra=InfoLabel(tooltip_html("news_feed")))
        hm_title = "DAY-CHANGE HEATMAP" if self._heatmap_style == "grid" else "HOLDINGS HEATMAP"
        self._dash.add_panel(
            PANEL_HEATMAP, hm_title, self._heatmap, c[PANEL_HEATMAP][2], c[PANEL_HEATMAP][3],
            x=c[PANEL_HEATMAP][0], y=c[PANEL_HEATMAP][1],
            header_extra=InfoLabel(tooltip_html("day_change_heatmap")))
        root.addWidget(self._dash, 1)

        self._news.symbolChanged.connect(self._on_news_symbol)
        self._dash.layoutChanged.connect(self._save_layout)
        self._restore_layout()
        self._style_timeframe_buttons()

    # ── Panel visibility + layout persistence (driven by LiveWatchView) ──
    def set_panel_visible(self, key: str, visible: bool) -> None:
        self._dash.set_panel_visible(key, visible)

    def panel_visible(self, key: str) -> bool:
        return self._dash.is_visible(key)

    def _save_layout(self) -> None:
        if self._settings is not None and self._layout_key:
            self._settings.set(self._layout_key, json.dumps(self._dash.save_layout()))

    def _restore_layout(self) -> None:
        if self._settings is None or not self._layout_key:
            return
        raw = self._settings.get(self._layout_key, "")
        if raw:
            try:
                self._dash.restore_layout(json.loads(raw))
            except Exception:
                pass

    def selected(self) -> Optional[str]:
        return self._selected

    # ── Data feed ──
    def set_data(self, order, weights, quotes) -> None:
        """Refresh the heatmap from a new snapshot and auto-load the first symbol's
        chart on the first snapshot. Re-renders the current chart so its prev-close
        line tracks the latest quote."""
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
            weights = self._weights or {t: 1.0 for t in order}
            self._heatmap.set_figure(charts.holdings_treemap(order, weights, changes))

    # ── Symbol selection → chart + news ──
    def load_symbol(self, symbol: str) -> None:
        symbol = (symbol or "").strip().upper()
        if not symbol:
            return
        self._selected = symbol
        self._chart_panel.set_title(f"PRICE CHART · {symbol} · {self._timeframe}")
        self._news.set_symbol(symbol)
        self._fetch_ohlc()

    def _on_news_symbol(self, sym: str) -> None:
        self._news_panel.set_title(f"NEWS · {sym}" if sym else "NEWS")

    def _on_timeframe(self, tf: str) -> None:
        if tf not in OHLC_TIMEFRAMES:
            return
        self._timeframe = tf
        self._style_timeframe_buttons()
        if self._selected:
            self._chart_panel.set_title(f"PRICE CHART · {self._selected} · {tf}")
            self._fetch_ohlc()

    def _fetch_ohlc(self) -> None:
        if not self._selected:
            return
        if self._ohlc_fetching:
            self._ohlc_pending = True
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
        if ticker != self._selected or timeframe != self._timeframe:
            return
        self._ohlc_df = df
        self._render_chart()

    def _on_ohlc_failed(self, _message: str) -> None:
        pass

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
            f" color:{t.text_muted}; padding:1px 7px; border-radius:{max(3, t.radius - 6)}px;"
            f" font-size:{t.label_pt - 1}px; font-weight:600; }}"
            f"QPushButton:hover {{ color:{t.text}; border-color:{t.accent}; }}"
            f"QPushButton:checked {{ background:{t.accent}; color:{t.accent_text};"
            f" border-color:{t.accent}; }}"
        )
        for b in self._tf_buttons.values():
            b.setStyleSheet(qss)

    # ── Lifecycle ──
    def reset(self) -> None:
        """Forget the charted symbol so the next snapshot auto-loads the new
        universe's first entry; clear the chart, news, and title."""
        self._selected = None
        self._ohlc_df = None
        self._ohlc_pending = False
        self._chart.clear()
        self._news.clear()
        self._chart_panel.set_title("PRICE CHART")

    def shutdown(self) -> None:
        self._ohlc_pending = False
        if self._ohlc_thread is not None and self._ohlc_thread.isRunning():
            self._ohlc_thread.quit()
            self._ohlc_thread.wait(3000)
        self._news.shutdown()

    def retheme(self) -> None:
        if self._order or self._quotes:
            self._render_heatmap()
        self._style_timeframe_buttons()
        self._chart.retheme()
        self._news.retheme()
        self._dash.retheme()
