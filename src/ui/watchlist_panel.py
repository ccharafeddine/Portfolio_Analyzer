"""Watchlist panel for the Live Market Watch section.

A persistent, user-curated list of symbols with delayed live quotes, shown as a
native sortable Qt table below the auto-portfolio cockpit. Fully self-contained
and decoupled from the analysis pipeline: it owns its ``QSettings``-backed store,
a background quote-fetch thread (the shared ``WatchlistWorker`` pattern), and an
auto-refresh timer that only ticks while the section is on screen.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.data.market_data import normalize_symbol

from . import theme
from .explanations import tooltip_html
from .quote_format import fmt_price as _fmt_price
from .quote_format import fmt_signed as _fmt_signed
from .watchlist import WatchlistStore
from .widgets.chart_heatmap_panel import ChartHeatmapPanel
from .widgets.info_label import InfoLabel
from .worker import WatchlistWorker

DEFAULT_REFRESH_SECS = 60
_COLUMNS = ["Symbol", "Name", "Last", "Chg %", ""]
_REMOVE_COL = 4
_NEG_INF = float("-inf")


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


class WatchlistPanel(QWidget):
    """Self-contained watchlist widget (store + worker + timer)."""

    # Emitted whenever the persisted symbol set changes (add/remove), in case a
    # host wants to react. Not required for the panel to function on its own.
    changed = Signal()

    def __init__(self, store: WatchlistStore | None = None, parent=None,
                 settings=None, layout_key: str = "") -> None:
        super().__init__(parent)
        self._store = store if store is not None else WatchlistStore()
        self._quotes: dict = {}
        self._thread: QThread | None = None
        self._worker: WatchlistWorker | None = None
        self._in_flight = False
        self._repoll = False
        self._repoll_force = False

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 8, 0, 0)
        root.setSpacing(8)

        # ── Add controls: input + Add + Refresh, sitting directly above the table
        # (top-left, over the Symbol/Last/Chg% columns), with the help "?" at the
        # far right of the row. ──
        self._input = QLineEdit()
        self._input.setPlaceholderText("Add symbol (e.g. NVDA, BTC)")
        self._input.setMaximumWidth(240)
        self._input.setClearButtonEnabled(True)
        self._input.returnPressed.connect(self._on_add)
        self._add_btn = QPushButton("Add")
        self._add_btn.setObjectName("secondary")
        self._add_btn.setCursor(Qt.PointingHandCursor)
        self._add_btn.clicked.connect(self._on_add)
        self._refresh_btn = QPushButton("Refresh")
        self._refresh_btn.setObjectName("secondary")
        self._refresh_btn.setCursor(Qt.PointingHandCursor)
        self._refresh_btn.clicked.connect(lambda: self._poll(force=True))

        controls = QHBoxLayout()
        controls.setSpacing(8)
        controls.addWidget(self._input)
        controls.addWidget(self._add_btn)
        controls.addWidget(self._refresh_btn)
        controls.addStretch(1)
        controls.addWidget(InfoLabel(tooltip_html("watchlist")))

        # ── Inline, non-blocking notice (hidden until an add is rejected) ──
        self._notice = QLabel("")
        self._notice.setVisible(False)
        self._notice_timer = QTimer(self)
        self._notice_timer.setSingleShot(True)
        self._notice_timer.timeout.connect(lambda: self._notice.setVisible(False))

        # ── Table ──
        self._table = QTableWidget(0, len(_COLUMNS))
        self._table.setHorizontalHeaderLabels(_COLUMNS)
        self._table.verticalHeader().setVisible(False)
        self._table.setSortingEnabled(True)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        hdr = self._table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(1, QHeaderView.Stretch)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(_REMOVE_COL, QHeaderView.ResizeToContents)
        self._table.cellClicked.connect(self._on_cell_clicked)
        self._table.setContextMenuPolicy(Qt.CustomContextMenu)
        self._table.customContextMenuRequested.connect(self._on_menu)

        # ── Left column: add controls + notice + table, stacked ──
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(8)
        lv.addLayout(controls)
        lv.addWidget(self._notice)
        lv.addWidget(self._table, 1)

        # ── Grid cockpit, mirroring the Portfolio tab: the watchlist (add controls
        # + table) is one card alongside the chart, news, and heatmap. Watchlist
        # symbols aren't weighted, so its heatmap is the equal-tile grid. ──
        self._chart = ChartHeatmapPanel(
            heatmap_style="grid", table_widget=left, table_title="Watchlist",
            settings=settings, layout_key=layout_key,
        )
        root.addWidget(self._chart, 1)

        # ── Auto-refresh timer (started/stopped by show/hide events) ──
        self._timer = QTimer(self)
        self._timer.setInterval(DEFAULT_REFRESH_SECS * 1000)
        self._timer.timeout.connect(self._poll)

        self._render()
        self.retheme()

    def chart_panel(self) -> ChartHeatmapPanel:
        """The shared intraday/treemap/day-change panel, so the host can apply the
        same panel-visibility toggles it applies to the Portfolio cockpit."""
        return self._chart

    # ── Add / remove ──
    def _on_add(self) -> None:
        raw = self._input.text()
        sym = self._store.add(raw)
        if sym is None:
            norm = normalize_symbol(raw)
            if not norm:
                self._show_notice(f"“{raw.strip()}” isn’t a recognizable symbol.")
            else:
                self._show_notice(f"{norm} is already in your watchlist.")
            return
        self._input.clear()
        self._notice.setVisible(False)
        self._render()
        self.changed.emit()
        self._poll(force=True)

    def _remove(self, sym: str) -> None:
        if self._store.remove(sym):
            self._quotes.pop(sym, None)
            if self._chart.selected() == normalize_symbol(sym):
                self._chart.reset()
            self._render()
            self._sync_chart()
            self.changed.emit()

    def _on_cell_clicked(self, row: int, col: int) -> None:
        item = self._table.item(row, 0)
        if item is None:
            return
        if col == _REMOVE_COL:
            self._remove(item.text())
        else:  # any other cell charts that symbol + loads its news
            self._chart.load_symbol(item.text())

    def _on_menu(self, pos) -> None:
        item = self._table.itemAt(pos)
        if item is None:
            return
        sym = self._table.item(item.row(), 0).text()
        menu = QMenu(self)
        menu.addAction(f"Remove {sym}", lambda: self._remove(sym))
        menu.exec(self._table.viewport().mapToGlobal(pos))

    def _show_notice(self, text: str) -> None:
        self._notice.setText(text)
        self._notice.setVisible(True)
        self._notice_timer.start(4000)

    # ── Rendering ──
    def _render(self) -> None:
        """Full rebuild — used when the symbol set changes (add/remove)."""
        syms = self._store.symbols()
        self._table.setSortingEnabled(False)
        self._table.setRowCount(len(syms))
        for row, sym in enumerate(syms):
            sym_item = QTableWidgetItem(sym)
            sym_item.setData(Qt.UserRole, sym)
            self._table.setItem(row, 0, sym_item)
            self._table.setItem(row, 1, QTableWidgetItem(""))
            self._table.setItem(row, 2, _NumItem("—", _NEG_INF))
            self._table.setItem(row, 3, _NumItem("—", _NEG_INF))
            x = QTableWidgetItem("✕")
            x.setTextAlignment(Qt.AlignCenter)
            x.setToolTip(f"Remove {sym}")
            self._table.setItem(row, _REMOVE_COL, x)
        self._table.setSortingEnabled(True)
        self._apply_quotes()

    def _apply_quotes(self) -> None:
        """Fill the price/name/change cells in place, preserving the user's sort."""
        t = theme.ACTIVE
        for row in range(self._table.rowCount()):
            item0 = self._table.item(row, 0)
            if item0 is None:
                continue
            q = self._quotes.get(item0.text())
            last = getattr(q, "last", None)
            pct = getattr(q, "change_pct", None)
            name = getattr(q, "name", None)
            if name:
                self._table.item(row, 1).setText(str(name))
            last_item = self._table.item(row, 2)
            last_item.setText(_fmt_price(last))
            last_item.setData(Qt.UserRole, last if last is not None else _NEG_INF)
            pct_item = self._table.item(row, 3)
            pct_item.setText(_fmt_signed(pct, pct=True))
            pct_item.setData(Qt.UserRole, pct if pct is not None else _NEG_INF)
            if isinstance(pct, (int, float)) and pct == pct:
                pct_item.setForeground(QColor(t.green if pct >= 0 else t.red))
            else:
                pct_item.setForeground(QColor(t.text_muted))

    # ── Quote polling (worker thread) ──
    def _poll(self, force: bool = False) -> None:
        syms = self._store.symbols()
        if not syms:
            self._quotes = {}
            self._render()
            return
        if self._in_flight:
            self._repoll = True
            self._repoll_force = self._repoll_force or force
            return
        self._in_flight = True
        self._thread = QThread(self)
        self._worker = WatchlistWorker(syms, use_cache=not force)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.done.connect(self._on_quotes)
        self._worker.failed.connect(self._on_failed)
        self._worker.done.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup)
        self._thread.start()

    def _on_quotes(self, quotes: dict) -> None:
        self._quotes = dict(quotes or {})
        self._apply_quotes()
        self._sync_chart()

    def _sync_chart(self) -> None:
        """Feed the current symbols + quotes to the chart/heatmap (equal tiles)."""
        self._chart.set_data(self._store.symbols(), {}, self._quotes)

    def _on_failed(self, _message: str) -> None:
        pass  # delayed data is best-effort; the next tick retries

    def _cleanup(self) -> None:
        if self._worker is not None:
            self._worker.deleteLater()
        if self._thread is not None:
            self._thread.deleteLater()
        self._worker = None
        self._thread = None
        self._in_flight = False
        if self._repoll:
            self._repoll = False
            force, self._repoll_force = self._repoll_force, False
            QTimer.singleShot(0, lambda: self._poll(force=force))

    # ── Visibility-gated auto-refresh ──
    def showEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().showEvent(event)
        if not self._timer.isActive():
            self._timer.start()
        self._poll()  # refresh on entering the section

    def hideEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().hideEvent(event)
        self._timer.stop()

    def shutdown(self) -> None:
        """Stop the timer + fetch threads cleanly (called on app close)."""
        self._repoll = False
        self._timer.stop()
        if self._thread is not None and self._thread.isRunning():
            self._thread.quit()
            self._thread.wait(3000)
        self._chart.shutdown()

    # ── Theming ──
    def retheme(self) -> None:
        t = theme.ACTIVE
        self._notice.setStyleSheet(
            f"color:{t.red};font-size:{t.base_pt - 1}px;background:transparent;"
        )
        if self._quotes:
            self._apply_quotes()
        self._chart.retheme()
