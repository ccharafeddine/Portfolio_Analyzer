"""Live Market Watch — a distinct in-app section (a QStackedWidget page).

Two tabs, each with a quotes table plus a click-through intraday chart and a
day-change heatmap (the shared :class:`ChartHeatmapPanel`):

- **Portfolio** — a cockpit for the currently loaded portfolio, built on delayed
  quotes (``market_data.fetch_quotes``): a live header (weighted day change, day
  P&L, and — when cost basis is set — market value vs. cost and unrealized P&L),
  refresh controls, an "as of … · delayed" freshness stamp, and a sortable
  quotes table (ticker, last, change, day range, volume, weight).
- **Watchlist** — a persistent, user-curated symbol list (:class:`WatchlistPanel`)
  with its own quotes table, chart, and heatmap, fully decoupled from analysis.

The Portfolio tab is passive about data: the main window owns the poll timer +
worker and pushes snapshots in via :meth:`set_quotes`. It emits
:attr:`refreshRequested` and :attr:`refreshIntervalChanged` for the main window
to act on. The Watchlist tab self-manages its own polling.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMenu,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from . import theme
from .quote_format import fmt_pct as _fmt_pct
from .quote_format import fmt_price as _fmt_price
from .quote_format import fmt_signed as _fmt_signed
from .quote_format import fmt_volume as _fmt_volume
from .settings import AppSettings
from .widgets.chart_heatmap_panel import (
    PANEL_DAYCHANGE,
    PANEL_PRICE,
    PANEL_TREEMAP,
    ChartHeatmapPanel,
)

# (label, seconds) — 0 means auto-refresh off.
REFRESH_OPTIONS = [("Off", 0), ("15s", 15), ("30s", 30), ("60s", 60)]
DEFAULT_INTERVAL = 30

_COLUMNS = ["Ticker", "Last", "Chg", "Chg %", "Day Range", "Volume", "Weight"]

# Toggleable panels: (key, menu label, default-visible). The chart keys match
# ChartHeatmapPanel; "watchlist" toggles the Watchlist tab's visibility.
PANEL_WATCHLIST = "watchlist"
_PANELS = [
    (PANEL_PRICE, "Price chart", True),
    (PANEL_TREEMAP, "Holdings treemap", True),
    (PANEL_DAYCHANGE, "Day-change heatmap", True),
    (PANEL_WATCHLIST, "Watchlist tab", True),
]

# QSettings keys for the persisted dashboard layout.
_KEY_H_SPLIT = "market_watch/h_splitter"   # Portfolio: quotes table | charts
_KEY_V_SPLIT = "market_watch/v_splitter"   # Portfolio charts: price | treemap | daychange
_KEY_SHOW = "market_watch/show_{}"          # per-panel visibility


class _NumItem(QTableWidgetItem):
    """A table cell that sorts by a numeric key rather than its display text."""

    def __init__(self, text: str, key: float) -> None:
        super().__init__(text)
        self.setData(Qt.UserRole, float(key))
        self.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)

    def __lt__(self, other) -> bool:  # noqa: D105
        try:
            return self.data(Qt.UserRole) < other.data(Qt.UserRole)
        except Exception:
            return super().__lt__(other)


_NEG_INF = float("-inf")


class LiveWatchView(QWidget):
    backRequested = Signal()
    refreshRequested = Signal()
    refreshIntervalChanged = Signal(int)  # seconds; 0 = off
    alertsRequested = Signal()
    costBasisEditRequested = Signal()

    def __init__(
        self,
        get_current_config: Optional[Callable[[], object]] = None,
        parent=None,
        settings: Optional[AppSettings] = None,
    ) -> None:
        super().__init__(parent)
        self._get_current_config = get_current_config
        self._settings = settings if settings is not None else AppSettings()
        self._panel_visible: dict[str, bool] = {}
        self._tickers: list[str] = []
        self._weights: dict[str, float] = {}
        self._cost_basis: dict[str, float] = {}
        self._capital: float = 0.0
        self._cash: float = 0.0
        self._quotes: dict = {}

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 12, 14, 10)
        root.setSpacing(10)

        # ── Top bar: back, title, freshness stamp ──
        top = QHBoxLayout()
        self._back_btn = QPushButton("←  Back to Analysis")
        self._back_btn.setCursor(Qt.PointingHandCursor)
        self._back_btn.clicked.connect(self.backRequested.emit)
        self._title = QLabel("Live Market Watch")
        self._stamp = QLabel("")
        top.addWidget(self._back_btn)
        top.addSpacing(12)
        top.addWidget(self._title)
        top.addStretch(1)
        top.addWidget(self._stamp)
        top.addSpacing(12)
        top.addWidget(self._build_panels_button())
        root.addLayout(top)

        # ── Portfolio tab: live header + controls + (quotes table | charts) ──
        port_tab = QWidget()
        pv = QVBoxLayout(port_tab)
        pv.setContentsMargins(0, 8, 0, 0)
        pv.setSpacing(10)

        # Live portfolio header (stat blocks)
        self._header = QHBoxLayout()
        self._header.setSpacing(22)
        self._stats: dict[str, QLabel] = {}
        for name in ("Day Change", "Day P&L", "Market Value", "Unrealized P&L"):
            block, value = self._make_stat(name)
            self._stats[name] = value
            self._header.addLayout(block)
        self._header.addStretch(1)
        pv.addLayout(self._header)
        # The Unrealized P&L stat doubles as a "set cost basis" affordance.
        unreal = self._stats["Unrealized P&L"]
        unreal.setTextInteractionFlags(Qt.LinksAccessibleByMouse)
        unreal.linkActivated.connect(lambda _href: self.costBasisEditRequested.emit())

        # Refresh controls
        controls = QHBoxLayout()
        controls.setSpacing(8)
        controls.addWidget(self._muted_label("Auto-refresh"))
        self._interval = QComboBox()
        for label, secs in REFRESH_OPTIONS:
            self._interval.addItem(label, secs)
        self._interval.setCurrentIndex(
            next((i for i, (_, s) in enumerate(REFRESH_OPTIONS) if s == DEFAULT_INTERVAL), 0)
        )
        self._interval.currentIndexChanged.connect(self._on_interval_changed)
        controls.addWidget(self._interval)
        self._refresh_btn = QPushButton("Refresh now")
        self._refresh_btn.setObjectName("secondary")
        self._refresh_btn.setCursor(Qt.PointingHandCursor)
        self._refresh_btn.clicked.connect(self.refreshRequested.emit)
        controls.addWidget(self._refresh_btn)
        self._alerts_btn = QPushButton("Alerts…")
        self._alerts_btn.setObjectName("secondary")
        self._alerts_btn.setCursor(Qt.PointingHandCursor)
        self._alerts_btn.clicked.connect(self.alertsRequested.emit)
        controls.addWidget(self._alerts_btn)
        controls.addStretch(1)
        pv.addLayout(controls)

        # Quotes table
        self._table = QTableWidget(0, len(_COLUMNS))
        self._table.setHorizontalHeaderLabels(_COLUMNS)
        self._table.verticalHeader().setVisible(False)
        self._table.setSortingEnabled(True)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        hdr = self._table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        for c in range(1, len(_COLUMNS)):
            hdr.setSectionResizeMode(c, QHeaderView.Stretch)
        self._table.cellClicked.connect(self._on_cell_clicked)
        self._table.setContextMenuPolicy(Qt.CustomContextMenu)
        self._table.customContextMenuRequested.connect(self._on_table_menu)

        # Body: quotes table (left) | charts cockpit (right), user-resizable.
        self._chart = ChartHeatmapPanel()
        self._h_splitter = QSplitter(Qt.Horizontal)
        self._h_splitter.addWidget(self._table)
        self._h_splitter.addWidget(self._chart)
        self._h_splitter.setStretchFactor(0, 5)
        self._h_splitter.setStretchFactor(1, 4)
        self._h_splitter.setChildrenCollapsible(False)
        self._h_splitter.splitterMoved.connect(self._save_splitters)
        self._chart.splitter().splitterMoved.connect(self._save_splitters)
        pv.addWidget(self._h_splitter, 1)

        # ── Watchlist tab: a persistent, user-curated symbol list with its own
        # quotes table + intraday chart + heatmap. Fully self-contained. ──
        from .watchlist_panel import WatchlistPanel

        self._watchlist = WatchlistPanel()

        self._tabs = QTabWidget()
        self._tabs.addTab(port_tab, "Portfolio")
        self._watchlist_tab_index = self._tabs.addTab(self._watchlist, "Watchlist")
        root.addWidget(self._tabs, 1)

        self._restore_layout()
        self.retheme()

    # ── Small builders ──
    def _build_panels_button(self) -> QToolButton:
        """A 'Panels ▾' popup of checkable actions to show/hide each panel — the
        app's menubar convention (checkable QActions), kept in-section since these
        toggles only make sense while Live Market Watch is on screen."""
        btn = QToolButton()
        btn.setText("Panels")
        btn.setObjectName("secondary")
        btn.setCursor(Qt.PointingHandCursor)
        btn.setPopupMode(QToolButton.InstantPopup)
        menu = QMenu(btn)
        self._panel_actions: dict[str, QAction] = {}
        for key, label, _default in _PANELS:
            act = QAction(label, self, checkable=True)
            act.toggled.connect(lambda checked, k=key: self._on_panel_toggled(k, checked))
            menu.addAction(act)
            self._panel_actions[key] = act
        btn.setMenu(menu)
        return btn
    def _make_stat(self, name: str):
        box = QVBoxLayout()
        box.setSpacing(2)
        label = QLabel(name.upper())
        label.setObjectName("liveStatLabel")
        value = QLabel("—")
        value.setObjectName("liveStatValue")
        box.addWidget(label)
        box.addWidget(value)
        return box, value

    def _muted_label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName("muted")
        return lbl

    # ── Resizable / toggleable panel dashboard ──
    def _restore_layout(self) -> None:
        """Restore splitter geometry + per-panel visibility from settings. Called
        once at build time, before the view is shown."""
        # Splitter geometry (QByteArray via saveState/restoreState, like the main
        # window persists its own geometry through AppSettings).
        h_state = self._settings.get(_KEY_H_SPLIT)
        if h_state:
            self._h_splitter.restoreState(h_state)
        v_state = self._settings.get(_KEY_V_SPLIT)
        if v_state:
            self._chart.splitter().restoreState(v_state)

        # Per-panel visibility. Reflect each into its checkable action (without
        # re-emitting) and apply it, so the menu, the widgets, and settings agree.
        for key, _label, default in _PANELS:
            visible = self._get_bool(_KEY_SHOW.format(key), default)
            self._panel_visible[key] = visible
            act = self._panel_actions.get(key)
            if act is not None:
                act.blockSignals(True)
                act.setChecked(visible)
                act.blockSignals(False)
            self._apply_panel_visibility(key, visible)

    def _apply_panel_visibility(self, key: str, visible: bool) -> None:
        """Show/hide the widget(s) a panel key maps to. Chart panels apply to both
        the Portfolio cockpit and the Watchlist tab so the two stay consistent."""
        if key == PANEL_WATCHLIST:
            self._tabs.setTabVisible(self._watchlist_tab_index, visible)
        else:
            self._chart.set_panel_visible(key, visible)
            self._watchlist.chart_panel().set_panel_visible(key, visible)

    def _on_panel_toggled(self, key: str, checked: bool) -> None:
        self.set_panel_visible(key, checked)

    def set_panel_visible(self, key: str, visible: bool) -> None:
        """Show/hide a panel and persist the choice. Keeps the menu action in sync
        when called programmatically."""
        visible = bool(visible)
        self._panel_visible[key] = visible
        act = self._panel_actions.get(key)
        if act is not None and act.isChecked() != visible:
            act.blockSignals(True)
            act.setChecked(visible)
            act.blockSignals(False)
        self._apply_panel_visibility(key, visible)
        self._settings.set(_KEY_SHOW.format(key), visible)

    def panel_visible(self, key: str) -> bool:
        return bool(self._panel_visible.get(key, True))

    def _save_splitters(self, *args) -> None:
        self._settings.set(_KEY_H_SPLIT, self._h_splitter.saveState())
        self._settings.set(_KEY_V_SPLIT, self._chart.splitter().saveState())

    def _save_layout(self) -> None:
        """Persist splitter geometry + every panel's visibility at once."""
        self._save_splitters()
        for key in self._panel_visible:
            self._settings.set(_KEY_SHOW.format(key), self._panel_visible[key])

    def _get_bool(self, key: str, default: bool) -> bool:
        v = self._settings.get(key, default)
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.strip().lower() in ("1", "true", "yes", "on")
        return bool(v)

    def hideEvent(self, event) -> None:  # noqa: N802 (Qt override)
        # Leaving the section is a natural checkpoint to persist the layout.
        self._save_layout()
        super().hideEvent(event)

    # ── Public API (driven by the main window) ──
    def set_portfolio(self, config) -> None:
        """Feed the current portfolio's universe so weights + cost basis are
        available for the header and the weight column."""
        if config is None:
            return
        self._tickers = list(getattr(config, "tickers", []) or [])
        self._weights = dict(getattr(config, "weights", {}) or {})
        self._cost_basis = dict(getattr(config, "cost_basis", {}) or {})
        self._capital = float(getattr(config, "capital", 0.0) or 0.0)
        self._cash = float(getattr(config, "cash", 0.0) or 0.0)

    def tickers(self) -> list[str]:
        return list(self._tickers)

    def current_interval(self) -> int:
        return int(self._interval.currentData() or 0)

    def set_interval(self, secs: int) -> None:
        """Select the combo entry matching ``secs`` (no-op if none matches).
        Does not emit :attr:`refreshIntervalChanged`."""
        idx = next((i for i, (_, s) in enumerate(REFRESH_OPTIONS) if s == int(secs)), None)
        if idx is not None:
            self._interval.blockSignals(True)
            self._interval.setCurrentIndex(idx)
            self._interval.blockSignals(False)

    def set_quotes(self, quotes: dict) -> None:
        self._quotes = dict(quotes or {})
        self._populate_table()
        self._update_header()
        self._update_stamp()
        order = self._tickers or list(self._quotes.keys())
        self._chart.set_data(order, self._weights, self._quotes)

    # ── Rendering ──
    def _populate_table(self) -> None:
        t = theme.ACTIVE
        order = self._tickers or list(self._quotes.keys())
        self._table.setSortingEnabled(False)
        self._table.setRowCount(len(order))
        for row, sym in enumerate(order):
            q = self._quotes.get(sym)
            last = getattr(q, "last", None)
            chg = getattr(q, "change", None)
            pct = getattr(q, "change_pct", None)
            hi = getattr(q, "day_high", None)
            lo = getattr(q, "day_low", None)
            vol = getattr(q, "volume", None)
            weight = self._weights.get(sym)

            name_item = QTableWidgetItem(sym)
            name_item.setData(Qt.UserRole, sym)
            self._table.setItem(row, 0, name_item)
            self._table.setItem(row, 1, _NumItem(_fmt_price(last), last if last is not None else _NEG_INF))

            chg_item = _NumItem(_fmt_signed(chg), chg if chg is not None else _NEG_INF)
            pct_item = _NumItem(_fmt_signed(pct, pct=True), pct if pct is not None else _NEG_INF)
            if isinstance(pct, (int, float)):
                color = t.green if pct >= 0 else t.red
                from PySide6.QtGui import QColor

                chg_item.setForeground(QColor(color))
                pct_item.setForeground(QColor(color))
            self._table.setItem(row, 2, chg_item)
            self._table.setItem(row, 3, pct_item)

            rng = f"{_fmt_price(lo)} – {_fmt_price(hi)}" if (lo is not None or hi is not None) else "—"
            rng_item = QTableWidgetItem(rng)
            rng_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self._table.setItem(row, 4, rng_item)
            self._table.setItem(row, 5, _NumItem(_fmt_volume(vol), vol if vol is not None else _NEG_INF))
            self._table.setItem(
                row, 6, _NumItem(_fmt_pct(weight) if weight is not None else "—",
                                 weight if weight is not None else _NEG_INF)
            )
        self._table.setSortingEnabled(True)

    def _update_header(self) -> None:
        # Weighted day change over holdings that have both a weight and a quote.
        wsum = 0.0
        wpct = 0.0
        for sym, w in self._weights.items():
            q = self._quotes.get(sym)
            pct = getattr(q, "change_pct", None)
            if isinstance(pct, (int, float)):
                wsum += w
                wpct += w * pct
        day_pct = (wpct / wsum) if wsum > 0 else None
        self._set_stat("Day Change", _fmt_pct(day_pct), day_pct)
        day_pnl = (self._capital * day_pct) if (day_pct is not None and self._capital) else None
        self._set_stat("Day P&L", self._money(day_pnl), day_pct)

        # Market value vs. cost basis — only when cost basis is set for all holdings.
        mv, cost = self._market_value_and_cost()
        if mv is None:
            self._set_stat("Market Value", "—", None)
            self._set_cost_basis_prompt()
        else:
            self._set_stat("Market Value", self._money(mv), None)
            unreal = mv - cost
            unreal_pct = (unreal / cost) if cost else None
            txt = self._money(unreal)
            if unreal_pct is not None:
                txt += f"  ({_fmt_pct(unreal_pct)})"
            self._set_stat("Unrealized P&L", txt, unreal)

    # ── Charts: intraday (click-through) + holdings heatmap (delegated) ──
    def _on_cell_clicked(self, row: int, _col: int) -> None:
        item = self._table.item(row, 0)
        if item is not None:
            self._chart.load_intraday(item.text())

    def _on_table_menu(self, pos) -> None:
        from PySide6.QtWidgets import QMenu

        menu = QMenu(self)
        item = self._table.itemAt(pos)
        if item is not None:
            sym = self._table.item(item.row(), 0).text()
            menu.addAction(f"Intraday chart · {sym}",
                           lambda: self._chart.load_intraday(sym))
        menu.addAction("Set cost basis…", self.costBasisEditRequested.emit)
        menu.exec(self._table.viewport().mapToGlobal(pos))

    def reset_selection(self) -> None:
        """Forget the charted ticker so the next snapshot auto-loads the new
        portfolio's first holding; clear the intraday chart + hint."""
        self._chart.reset()

    def shutdown(self) -> None:
        """Persist the layout, then stop the background fetch threads cleanly
        (called on app close)."""
        self._save_layout()
        self._chart.shutdown()
        self._watchlist.shutdown()

    def _market_value_and_cost(self):
        """Return (market_value, invested_cost) when every holding has a positive
        cost basis, else (None, None). Implied shares = capital*weight / cost."""
        if not self._capital or not self._weights:
            return None, None
        mv = 0.0
        cost = 0.0
        invested_total = max(self._capital - self._cash, 0.0)  # capital is total now
        for sym, w in self._weights.items():
            cb = self._cost_basis.get(sym)
            last = getattr(self._quotes.get(sym), "last", None)
            if not isinstance(cb, (int, float)) or cb <= 0 or not isinstance(last, (int, float)):
                return None, None
            invested = invested_total * w
            shares = invested / cb
            mv += shares * last
            cost += invested
        # Cash held alongside the stocks: adds to both value and cost (no gain), so
        # the total market value is complete and the unrealized % reflects cash drag.
        if self._cash:
            mv += self._cash
            cost += self._cash
        return mv, cost

    def _money(self, v) -> str:
        if not isinstance(v, (int, float)) or v != v:
            return "—"
        sign = "-" if v < 0 else ""
        return f"{sign}${abs(v):,.0f}"

    def _update_stamp(self) -> None:
        as_of = None
        realtime = False
        source = None
        for q in self._quotes.values():
            if getattr(q, "as_of", None) and as_of is None:
                as_of = q.as_of
            if getattr(q, "realtime", False):
                realtime = True
                source = getattr(q, "source", None)
        if not as_of:
            self._stamp.setText("")
            return
        try:
            dt = datetime.fromisoformat(as_of)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            local = dt.astimezone().strftime("%I:%M:%S %p").lstrip("0")
        except Exception:
            local = as_of
        mode = f"real-time · {source}" if realtime else "delayed"
        self._stamp.setText(f"as of {local}  ·  {mode}")

    # ── Stat styling ──
    def _set_stat(self, name: str, text: str, signed) -> None:
        lbl = self._stats.get(name)
        if lbl is None:
            return
        t = theme.ACTIVE
        color = t.text
        if isinstance(signed, (int, float)) and signed == signed and signed != 0:
            color = t.green if signed > 0 else t.red
        lbl.setStyleSheet(
            f"color:{color};font-size:{t.statval_pt}px;font-weight:700;"
            f"font-family:{t.mono};background:transparent;"
        )
        lbl.setCursor(Qt.ArrowCursor)
        lbl.setText(text)

    def _set_cost_basis_prompt(self) -> None:
        """Render Unrealized P&L as a clickable 'set cost basis' link."""
        lbl = self._stats.get("Unrealized P&L")
        if lbl is None:
            return
        t = theme.ACTIVE
        lbl.setStyleSheet(
            f"font-size:{t.base_pt + 1}px;font-weight:700;background:transparent;"
        )
        lbl.setCursor(Qt.PointingHandCursor)
        lbl.setText(
            f"<a href='setcb' style='color:{t.accent};text-decoration:none;'>"
            f"set cost basis ›</a>"
        )

    def _on_interval_changed(self, _idx: int) -> None:
        self.refreshIntervalChanged.emit(self.current_interval())

    def retheme(self) -> None:
        t = theme.ACTIVE
        self._title.setStyleSheet(
            f"color:{t.text};font-size:{t.heading_pt + 2}px;font-weight:700;"
        )
        self._stamp.setStyleSheet(f"color:{t.text_muted};font-size:{t.base_pt - 1}px;")
        # Restyle the small uppercase stat labels via a scoped stylesheet.
        self.setStyleSheet(
            f"QLabel#liveStatLabel {{ color:{t.text_muted}; font-size:{t.label_pt}px;"
            f" font-weight:600; letter-spacing:0.06em; }}"
        )
        # Re-render values (colors depend on theme) if we have data.
        if self._quotes:
            self._update_header()
            self._populate_table()
        self._chart.retheme()
        self._watchlist.retheme()
