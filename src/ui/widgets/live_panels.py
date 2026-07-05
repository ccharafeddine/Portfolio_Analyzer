"""Small data panels for the Live Market Watch grid cockpit.

Each is a self-contained content widget the cockpit drops into a grid card and
feeds on every quotes snapshot:

- :class:`TopMoversPanel`   — holdings/symbols sorted by today's move.
- :class:`ContributorsPanel` — each holding's dollar contribution to today's P&L.
- :class:`AllocationPanel`   — a live allocation donut (value weights when cost
  basis is known, else target weights), with a Cash slice.
- :class:`EventsPanel`       — upcoming earnings / ex-dividend dates (fetched off
  the UI thread).
- :class:`AlertsPanel`       — active price alerts and how close each is to firing.

All are Qt widgets but their data reduction is simple and defensive.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from PySide6.QtCore import Qt, QThread
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .. import theme
from ..worker import CalendarWorker
from .plotly_widget import PlotlyWidget


def _num(v) -> Optional[float]:
    return float(v) if isinstance(v, (int, float)) and v == v else None


def _table(columns) -> QTableWidget:
    t = QTableWidget(0, len(columns))
    t.setHorizontalHeaderLabels(columns)
    t.verticalHeader().setVisible(False)
    t.setEditTriggers(QTableWidget.NoEditTriggers)
    t.setSelectionMode(QTableWidget.NoSelection)
    t.setShowGrid(False)
    hdr = t.horizontalHeader()
    hdr.setSectionResizeMode(0, QHeaderView.Stretch)
    for c in range(1, len(columns)):
        hdr.setSectionResizeMode(c, QHeaderView.ResizeToContents)
    return t


def _cell(text: str, color: Optional[str] = None, right: bool = False) -> QTableWidgetItem:
    it = QTableWidgetItem(text)
    if right:
        it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
    if color:
        it.setForeground(QColor(color))
    return it


def _placeholder(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setObjectName("muted")
    lbl.setAlignment(Qt.AlignCenter)
    lbl.setWordWrap(True)
    return lbl


class _StackPanel(QWidget):
    """A panel that swaps between a table and a centered placeholder message."""

    def __init__(self, columns) -> None:
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self._table = _table(columns)
        self._msg = _placeholder("")
        lay.addWidget(self._table)
        lay.addWidget(self._msg)
        self._show_table(False)

    def _show_table(self, on: bool) -> None:
        self._table.setVisible(on)
        self._msg.setVisible(not on)

    def _empty(self, text: str) -> None:
        self._msg.setText(text)
        self._show_table(False)


class TopMoversPanel(_StackPanel):
    def __init__(self) -> None:
        super().__init__(["Symbol", "Last", "Day"])

    def update_data(self, order, quotes) -> None:
        rows = []
        for sym in order or []:
            q = (quotes or {}).get(sym)
            pct = _num(getattr(q, "change_pct", None))
            if pct is not None:
                rows.append((sym, _num(getattr(q, "last", None)), pct))
        if not rows:
            self._empty("No quotes yet.")
            return
        rows.sort(key=lambda r: r[2], reverse=True)
        t = theme.ACTIVE
        self._table.setRowCount(len(rows))
        for i, (sym, last, pct) in enumerate(rows):
            color = t.green if pct >= 0 else t.red
            self._table.setItem(i, 0, _cell(sym))
            self._table.setItem(i, 1, _cell(f"{last:,.2f}" if last is not None else "—", right=True))
            self._table.setItem(i, 2, _cell(f"{pct * 100:+.2f}%", color, right=True))
        self._show_table(True)


class ContributorsPanel(_StackPanel):
    def __init__(self) -> None:
        super().__init__(["Symbol", "Day P&L"])

    def update_data(self, order, weights, quotes, invested) -> None:
        rows = []
        for sym, w in (weights or {}).items():
            pct = _num(getattr((quotes or {}).get(sym), "change_pct", None))
            if pct is not None and invested:
                rows.append((sym, invested * w * pct))
        if not rows:
            self._empty("Set weights + capital to see contributions.")
            return
        rows.sort(key=lambda r: abs(r[1]), reverse=True)
        t = theme.ACTIVE
        self._table.setRowCount(len(rows))
        for i, (sym, dollars) in enumerate(rows):
            color = t.green if dollars >= 0 else t.red
            sign = "-" if dollars < 0 else ""
            self._table.setItem(i, 0, _cell(sym))
            self._table.setItem(i, 1, _cell(f"{sign}${abs(dollars):,.0f}", color, right=True))
        self._show_table(True)


class AlertsPanel(_StackPanel):
    def __init__(self) -> None:
        super().__init__(["Alert", "Last", "To go"])
        from ..alerts import AlertStore

        self._store = AlertStore()

    def update_data(self, quotes) -> None:
        self._store.load()  # pick up edits made in the Alerts dialog
        rows = []
        for a in self._store.all():
            if not a.enabled:
                continue
            last = _num(getattr((quotes or {}).get(a.ticker), "last", None))
            dist = (abs(last - a.price) / a.price) if (last is not None and a.price) else None
            rows.append((a, last, dist))
        if not rows:
            self._empty("No active price alerts.\nAdd them in Settings → Price Alerts.")
            return
        rows.sort(key=lambda r: (r[2] is None, r[2] if r[2] is not None else 1e9))
        t = theme.ACTIVE
        self._table.setRowCount(len(rows))
        for i, (a, last, dist) in enumerate(rows):
            self._table.setItem(i, 0, _cell(a.describe()))
            self._table.setItem(i, 1, _cell(f"{last:,.2f}" if last is not None else "—", right=True))
            near = dist is not None and dist <= 0.02
            self._table.setItem(
                i, 2, _cell(f"{dist * 100:.1f}%" if dist is not None else "—",
                            t.accent if near else t.text_slate, right=True))
        self._show_table(True)


class EventsPanel(_StackPanel):
    def __init__(self) -> None:
        super().__init__(["Ticker", "Event", "When"])
        self._universe: list[str] = []
        self._fetching: list[str] = []   # the universe the in-flight fetch was started for
        self._thread: Optional[QThread] = None
        self._worker: Optional[CalendarWorker] = None
        self._empty("No holdings yet.")

    def set_universe(self, tickers) -> None:
        tickers = list(tickers or [])
        if tickers == self._universe:
            return
        self._universe = tickers
        if not tickers:
            self._empty("No holdings yet.")
            return
        if self._thread is not None:
            return  # a fetch is running; _cleanup re-fires if the universe drifted
        self._empty("Loading upcoming events…")
        self._fetching = list(tickers)
        self._thread = QThread(self)
        self._worker = CalendarWorker(tickers)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.done.connect(self._on_events)
        self._worker.failed.connect(self._on_failed)
        self._worker.done.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup)
        self._thread.start()

    def _on_events(self, events) -> None:
        events = events or []
        if not events:
            self._empty("No upcoming earnings or ex-dividend dates.")
            return
        today = datetime.now().date()
        self._table.setRowCount(len(events))
        for i, e in enumerate(events):
            self._table.setItem(i, 0, _cell(e.ticker))
            self._table.setItem(i, 1, _cell(e.kind))
            self._table.setItem(i, 2, _cell(_when(e.date, today), right=True))
        self._show_table(True)

    def _on_failed(self, _msg: str) -> None:
        self._empty("Couldn't load events.")

    def _cleanup(self) -> None:
        if self._worker is not None:
            self._worker.deleteLater()
        if self._thread is not None:
            self._thread.deleteLater()
        self._worker = None
        self._thread = None
        # If the universe changed while that fetch was in flight, fetch again now.
        if self._universe and self._universe != self._fetching:
            pending, self._universe = self._universe, []
            self.set_universe(pending)

    def shutdown(self) -> None:
        if self._thread is not None and self._thread.isRunning():
            self._thread.quit()
            self._thread.wait(3000)


class AllocationPanel(QWidget):
    def __init__(self) -> None:
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self._chart = PlotlyWidget()
        lay.addWidget(self._chart)
        self._last = None  # (weights, quotes, capital, cash, cost_basis)

    def update_data(self, weights, quotes, capital, cash, cost_basis) -> None:
        self._last = (weights, quotes, capital, cash, cost_basis)
        self._render()

    def _render(self) -> None:
        from src.charts.plotly_charts import allocation_donut

        weights, quotes, capital, cash, cost_basis = self._last or ({}, {}, 0, 0, {})
        weights = weights or {}
        if not weights:
            self._chart.set_figure(None)
            return
        invested = max(float(capital or 0.0) - float(cash or 0.0), 0.0)
        alloc = self._live_weights(weights, quotes or {}, invested, cost_basis or {})
        title = "Live allocation" if alloc is not None else "Target allocation"
        if alloc is None:
            # Fallback to target weights, renormalized so the slices (+ Cash) sum to 1.
            total = sum(v for v in weights.values() if isinstance(v, (int, float)))
            alloc = {k: v / total for k, v in weights.items()} if total > 0 else dict(weights)
        if cash and capital:
            y = invested / float(capital)
            alloc = {k: v * y for k, v in alloc.items()}
            alloc["Cash"] = float(cash) / float(capital)
        self._chart.set_figure(allocation_donut(alloc, title))

    @staticmethod
    def _live_weights(weights, quotes, invested, cost_basis):
        """Value-weighted allocation from live prices when a positive cost basis is
        known for every holding (shows drift vs target); else None."""
        if not invested:
            return None
        values = {}
        for sym, w in weights.items():
            cb = cost_basis.get(sym)
            last = _num(getattr(quotes.get(sym), "last", None))
            if not (isinstance(cb, (int, float)) and cb > 0 and last is not None):
                return None
            values[sym] = (invested * w / cb) * last
        total = sum(values.values())
        if total <= 0:
            return None
        return {k: v / total for k, v in values.items()}

    def retheme(self) -> None:
        if self._last is not None:
            self._render()


def _when(iso: str, today) -> str:
    try:
        d = datetime.fromisoformat(iso).date()
    except Exception:
        return iso
    days = (d - today).days
    if days == 0:
        return "today"
    if days == 1:
        return "tomorrow"
    return f"{days}d"
