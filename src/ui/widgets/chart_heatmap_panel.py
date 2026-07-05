"""The Live Market Watch cockpit — a drag-and-drop grid of live panels.

Both the Portfolio and Watchlist tabs show the same cockpit (:class:`GridDashboard`)
of distinct, movable/resizable cards. The core cards are the quotes **table**
(passed in by the host), a TradingView-style candlestick **chart** (timeframe
buttons in its header), a per-symbol **news** panel, and a day-change **heatmap**
(treemap on Portfolio, grid on Watchlist). Additional live cards: **top movers**,
**upcoming events**, and **alerts** on both tabs, plus **Day P&L contributors**
and an **allocation** donut on the Portfolio tab (which has weights + capital).

Selecting a symbol (:meth:`load_symbol`, wired to a table row-click) fetches its
OHLC off the UI thread and points the news panel at it. Every snapshot
(:meth:`set_data`) feeds all the data-driven cards. Panels are added/removed via
the host's menu (:meth:`available_panels` / :meth:`set_panel_visible`) and the
arrangement persists via ``settings``/``layout_key``.
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
from .live_panels import (
    AlertsPanel,
    AllocationPanel,
    ContributorsPanel,
    EventsPanel,
    TopMoversPanel,
)
from .news_panel import NewsPanel
from .plotly_widget import PlotlyWidget

# Panel keys (also used by the host menu + persisted layout).
PANEL_TABLE = "table"
PANEL_CHART = "chart"
PANEL_NEWS = "news"
PANEL_HEATMAP = "heatmap"
PANEL_MOVERS = "movers"
PANEL_CONTRIB = "contrib"
PANEL_ALLOC = "alloc"
PANEL_EVENTS = "events"
PANEL_ALERTS = "alerts"

# Default card size (cols, rows) for the initial auto-flow placement (used only
# for a panel with no entry in the shipped default layout below).
_SIZES = {
    PANEL_TABLE: (4, 8),
    PANEL_CHART: (8, 8),
    PANEL_NEWS: (4, 4),
    PANEL_HEATMAP: (4, 4),
    PANEL_MOVERS: (4, 4),
    PANEL_CONTRIB: (4, 4),
    PANEL_ALLOC: (4, 4),
    PANEL_EVENTS: (6, 4),
    PANEL_ALERTS: (6, 4),
}

# The curated default arrangement shipped with the app (a fresh install with no
# saved layout gets this). Each is {panel_id: [x, y, w, h], "__hidden__": [...]}.
_DEFAULT_LAYOUT_PORTFOLIO = {
    "table": [0, 0, 4, 13], "chart": [4, 0, 8, 22], "news": [0, 22, 4, 19],
    "heatmap": [4, 22, 4, 19], "movers": [2, 13, 2, 9], "contrib": [0, 13, 2, 9],
    "alloc": [10, 22, 2, 19], "events": [8, 22, 2, 12], "alerts": [8, 34, 2, 7],
    "__hidden__": [],
}
_DEFAULT_LAYOUT_WATCHLIST = {
    "table": [0, 0, 4, 9], "chart": [4, 0, 8, 10], "news": [0, 11, 4, 5],
    "heatmap": [4, 10, 6, 6], "movers": [10, 10, 2, 6], "events": [0, 9, 2, 2],
    "alerts": [2, 9, 2, 2], "__hidden__": [],
}


class ChartHeatmapPanel(QWidget):
    """Grid cockpit of live cards. ``heatmap_style`` is ``"treemap"`` (Portfolio,
    weight-sized) or ``"grid"`` (Watchlist, equal tiles). Portfolio also gets the
    Day P&L contributors + allocation cards."""

    def __init__(self, heatmap_style: str = "treemap", table_widget: QWidget = None,
                 table_title: str = "Portfolio", settings=None, layout_key: str = "",
                 parent=None) -> None:
        super().__init__(parent)
        self._heatmap_style = heatmap_style if heatmap_style in ("treemap", "grid") else "treemap"
        self._is_portfolio = self._heatmap_style == "treemap"
        self._settings = settings
        self._layout_key = layout_key
        self._snapshot = ([], {}, {}, 0.0, 0.0, {})  # order, weights, quotes, cap, cash, cb

        # Selection + OHLC fetch state.
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

        self._available: list[tuple[str, str]] = []
        self._chart = LightweightChartWidget()
        self._news = NewsPanel(show_header=False)
        self._heatmap = PlotlyWidget()
        self._movers = TopMoversPanel()
        self._events = EventsPanel()
        self._alerts = AlertsPanel()
        self._contrib = ContributorsPanel() if self._is_portfolio else None
        self._alloc = AllocationPanel() if self._is_portfolio else None

        # ── Core cards ──
        if table_widget is not None:
            self._add(PANEL_TABLE, table_title.upper(), table_widget)
        self._chart_panel = self._add(PANEL_CHART, "PRICE CHART", self._chart,
                                      header_extra=self._build_tf_bar())
        self._news_panel = self._add(PANEL_NEWS, "NEWS", self._news,
                                     header_extra=InfoLabel(tooltip_html("news_feed")))
        hm_title = "DAY-CHANGE HEATMAP" if not self._is_portfolio else "HOLDINGS HEATMAP"
        self._add(PANEL_HEATMAP, hm_title, self._heatmap,
                  header_extra=InfoLabel(tooltip_html("day_change_heatmap")))

        # ── Extra live cards ──
        self._add(PANEL_MOVERS, "TOP MOVERS", self._movers)
        if self._is_portfolio:
            self._add(PANEL_CONTRIB, "DAY P&L CONTRIBUTORS", self._contrib)
            self._add(PANEL_ALLOC, "ALLOCATION", self._alloc)
        self._add(PANEL_EVENTS, "UPCOMING EVENTS", self._events)
        self._add(PANEL_ALERTS, "ALERTS", self._alerts)

        root.addWidget(self._dash, 1)

        self._news.symbolChanged.connect(self._on_news_symbol)
        self._dash.layoutChanged.connect(self._save_layout)
        self._restore_layout()
        self._style_timeframe_buttons()

    def _add(self, key, title, widget, header_extra=None):
        w, h = _SIZES[key]
        self._available.append((key, title.title()))
        return self._dash.add_panel(key, title, widget, w, h, header_extra=header_extra)

    def _build_tf_bar(self) -> QWidget:
        bar = QWidget()
        tb = QHBoxLayout(bar)
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
        return bar

    # ── Host menu API ──
    def available_panels(self) -> list[tuple[str, str]]:
        return list(self._available)

    def set_panel_visible(self, key: str, visible: bool) -> None:
        self._dash.set_panel_visible(key, visible)

    def panel_visible(self, key: str) -> bool:
        return self._dash.is_visible(key)

    def _save_layout(self) -> None:
        if self._settings is not None and self._layout_key:
            self._settings.set(self._layout_key, json.dumps(self._dash.save_layout()))

    def _restore_layout(self) -> None:
        # A user's saved layout wins; otherwise apply the shipped default arrangement.
        if self._settings is not None and self._layout_key:
            raw = self._settings.get(self._layout_key, "")
            if raw:
                try:
                    self._dash.restore_layout(json.loads(raw))
                    return
                except Exception:
                    pass
        default = _DEFAULT_LAYOUT_PORTFOLIO if self._is_portfolio else _DEFAULT_LAYOUT_WATCHLIST
        self._dash.restore_layout(dict(default))

    def selected(self) -> Optional[str]:
        return self._selected

    # ── Data feed ──
    def set_data(self, order, weights, quotes, capital: float = 0.0,
                 cash: float = 0.0, cost_basis=None) -> None:
        self._snapshot = (list(order or []), dict(weights or {}), dict(quotes or {}),
                          float(capital or 0.0), float(cash or 0.0), dict(cost_basis or {}))
        self._feed_panels()
        if self._selected is None and self._snapshot[0]:
            self.load_symbol(self._snapshot[0][0])
        elif self._selected is not None and self._ohlc_df is not None:
            self._render_chart()

    def _feed_panels(self) -> None:
        order, weights, quotes, capital, cash, cost_basis = self._snapshot
        self._render_heatmap()
        self._movers.update_data(order or list(quotes.keys()), quotes)
        self._events.set_universe(order or list(quotes.keys()))
        self._alerts.update_data(quotes)
        if self._contrib is not None:
            self._contrib.update_data(order, weights, quotes, max(capital - cash, 0.0))
        if self._alloc is not None:
            self._alloc.update_data(weights, quotes, capital, cash, cost_basis)

    def _render_heatmap(self) -> None:
        order, weights, quotes = self._snapshot[0], self._snapshot[1], self._snapshot[2]
        order = order or list(quotes.keys())
        changes = {t: getattr(quotes.get(t), "change_pct", None) for t in order}
        if self._heatmap_style == "grid":
            self._heatmap.set_figure(charts.day_change_heatmap(order, changes))
        else:
            w = weights or {t: 1.0 for t in order}
            self._heatmap.set_figure(charts.holdings_treemap(order, w, changes))

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
        prev = getattr(self._snapshot[2].get(self._selected), "prev_close", None)
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
        self._events.shutdown()

    def retheme(self) -> None:
        self._feed_panels()
        self._style_timeframe_buttons()
        self._chart.retheme()
        self._news.retheme()
        if self._alloc is not None:
            self._alloc.retheme()
        self._dash.retheme()
