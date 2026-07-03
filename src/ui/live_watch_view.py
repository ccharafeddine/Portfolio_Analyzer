"""Live Market Watch — a distinct in-app section (a QStackedWidget page).

A cockpit for the currently loaded portfolio, built on delayed quotes
(``market_data.fetch_quotes``, typically 15–20 min behind the exchange):

- a live portfolio header (weighted day change, day P&L, and — when cost basis
  is set — market value vs. cost and unrealized P&L),
- refresh controls (auto-refresh interval + a manual refresh) and an
  "as of … · delayed" freshness stamp,
- a sortable quotes table (ticker, last, change, day range, volume, weight).

The view is passive about data: the main window owns the poll timer + worker and
pushes snapshots in via :meth:`set_quotes`. It emits :attr:`refreshRequested`
and :attr:`refreshIntervalChanged` for the main window to act on.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from . import theme

# (label, seconds) — 0 means auto-refresh off.
REFRESH_OPTIONS = [("Off", 0), ("15s", 15), ("30s", 30), ("60s", 60)]
DEFAULT_INTERVAL = 30

_COLUMNS = ["Ticker", "Last", "Chg", "Chg %", "Day Range", "Volume", "Weight"]


def _fmt_price(v) -> str:
    return f"{v:,.2f}" if isinstance(v, (int, float)) else "—"


def _fmt_signed(v, pct=False) -> str:
    if not isinstance(v, (int, float)) or v != v:
        return "—"
    body = f"{abs(v) * 100:.2f}%" if pct else f"{abs(v):,.2f}"
    return f"{'+' if v >= 0 else '-'}{body}"


def _fmt_pct(v, d=2) -> str:
    return f"{v * 100:.{d}f}%" if isinstance(v, (int, float)) else "—"


def _fmt_volume(v) -> str:
    if not isinstance(v, (int, float)) or v != v:
        return "—"
    for unit, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(v) >= div:
            return f"{v / div:.2f}{unit}"
    return f"{v:.0f}"


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

    def __init__(
        self,
        get_current_config: Optional[Callable[[], object]] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._get_current_config = get_current_config
        self._tickers: list[str] = []
        self._weights: dict[str, float] = {}
        self._cost_basis: dict[str, float] = {}
        self._capital: float = 0.0
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
        root.addLayout(top)

        # ── Live portfolio header (stat blocks) ──
        self._header = QHBoxLayout()
        self._header.setSpacing(22)
        self._stats: dict[str, QLabel] = {}
        for name in ("Day Change", "Day P&L", "Market Value", "Unrealized P&L"):
            block, value = self._make_stat(name)
            self._stats[name] = value
            self._header.addLayout(block)
        self._header.addStretch(1)
        root.addLayout(self._header)

        # ── Refresh controls ──
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
        controls.addStretch(1)
        root.addLayout(controls)

        # ── Quotes table ──
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
        root.addWidget(self._table, 1)

        self.retheme()

    # ── Small builders ──
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
            self._set_stat("Unrealized P&L", "set cost basis", None)
        else:
            self._set_stat("Market Value", self._money(mv), None)
            unreal = mv - cost
            unreal_pct = (unreal / cost) if cost else None
            txt = self._money(unreal)
            if unreal_pct is not None:
                txt += f"  ({_fmt_pct(unreal_pct)})"
            self._set_stat("Unrealized P&L", txt, unreal)

    def _market_value_and_cost(self):
        """Return (market_value, invested_cost) when every holding has a positive
        cost basis, else (None, None). Implied shares = capital*weight / cost."""
        if not self._capital or not self._weights:
            return None, None
        mv = 0.0
        cost = 0.0
        for sym, w in self._weights.items():
            cb = self._cost_basis.get(sym)
            last = getattr(self._quotes.get(sym), "last", None)
            if not isinstance(cb, (int, float)) or cb <= 0 or not isinstance(last, (int, float)):
                return None, None
            invested = self._capital * w
            shares = invested / cb
            mv += shares * last
            cost += invested
        return mv, cost

    def _money(self, v) -> str:
        if not isinstance(v, (int, float)) or v != v:
            return "—"
        sign = "-" if v < 0 else ""
        return f"{sign}${abs(v):,.0f}"

    def _update_stamp(self) -> None:
        as_of = None
        for q in self._quotes.values():
            as_of = getattr(q, "as_of", None)
            if as_of:
                break
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
        self._stamp.setText(f"as of {local}  ·  delayed")

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
        lbl.setText(text)

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
