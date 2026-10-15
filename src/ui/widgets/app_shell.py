"""App-shell widgets for the redesigned layout.

- ``TopBar``     pinned context bar: brand, portfolio / benchmark / period / Rf
                 chips (click to edit the configuration) and the main actions.
- ``NavPanel``   grouped left navigation that replaces the old row of tabs.
- ``KpiStrip``   hairline-divided headline numbers with a benchmark sub-line.
- ``DataStatus`` status-bar read-out: data source, as-of, days, gaps, Rf source.

All styling comes from ``theme.stylesheet()`` via object names; nothing here
hard-codes colours except the P&L tone, which is read from the active theme.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QPushButton,
    QToolButton, QVBoxLayout, QWidget,
)

from .. import theme
from ..explanations import tooltip_html

# ── Top bar ────────────────────────────────────────────────────────────────


class TopBar(QFrame):
    runClicked = Signal()
    configureClicked = Signal()
    exportClicked = Signal()

    def __init__(self, app_name: str = "Portfolio Analyzer", parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("topBar")
        row = QHBoxLayout(self)
        row.setContentsMargins(14, 6, 10, 6)
        row.setSpacing(8)

        brand = QLabel(app_name.upper())
        brand.setObjectName("brand")
        row.addWidget(brand)
        row.addSpacing(14)

        self.chip_portfolio = self._chip("Untitled portfolio", "Edit holdings and settings")
        self.chip_benchmark = self._chip("SPY", "Benchmark (edit in configuration)")
        self.chip_period = self._chip("—", "Analysis period (edit in configuration)")
        self.lbl_rf = QLabel("")
        self.lbl_rf.setObjectName("ctxLabel")
        vs = QLabel("vs")
        vs.setObjectName("ctxLabel")
        for w in (self.chip_portfolio, vs, self.chip_benchmark, self.chip_period):
            row.addWidget(w)
        row.addSpacing(6)
        row.addWidget(self.lbl_rf)
        row.addStretch(1)

        self.btn_configure = self._button("Configure", "ghost", "Show or hide the configuration panel (Ctrl+B)")
        self.btn_export = self._button("Export", "ghost", "Reports and CSV data pack")
        self.btn_run = self._button("Run analysis", "primary", "Run the analysis (Ctrl+R)")
        for b in (self.btn_configure, self.btn_export, self.btn_run):
            row.addWidget(b)
        self.btn_configure.clicked.connect(self.configureClicked)
        self.btn_export.clicked.connect(self.exportClicked)
        self.btn_run.clicked.connect(self.runClicked)

    def _chip(self, text: str, tip: str) -> QToolButton:
        b = QToolButton()
        b.setObjectName("ctxChip")
        b.setText(text)
        b.setToolTip(tip)
        b.setCursor(Qt.PointingHandCursor)
        b.clicked.connect(self.configureClicked)
        return b

    @staticmethod
    def _button(text: str, name: str, tip: str) -> QPushButton:
        b = QPushButton(text)
        b.setObjectName(name)
        b.setToolTip(tip)
        b.setCursor(Qt.PointingHandCursor)
        return b

    def set_context(self, portfolio: str, benchmark: str, start, end,
                    rf_text: str = "") -> None:
        self.chip_portfolio.setText(portfolio or "Untitled portfolio")
        self.chip_benchmark.setText(benchmark or "—")
        fmt = lambda d: d.strftime("%b %Y") if hasattr(d, "strftime") else str(d)  # noqa: E731
        self.chip_period.setText(f"{fmt(start)} – {fmt(end)}" if start and end else "—")
        self.lbl_rf.setText(rf_text)

    def set_running(self, running: bool) -> None:
        self.btn_run.setEnabled(not running)
        self.btn_run.setText("Running…" if running else "Run analysis")


# ── Navigation ─────────────────────────────────────────────────────────────

NAV_GROUPS: list[tuple[str, list[tuple[str, str]]]] = [
    # (group, [(display label, page key = ResultsView tab label)])
    ("Analysis", [("Overview", "Overview"), ("Performance", "Performance"),
                  ("Risk", "Risk"), ("Attribution", "Attribution"), ("Income", "Income")]),
    ("Construction", [("Optimization", "Optimization"), ("Forecast", "Forecast")]),
    ("Research", [("Fundamentals", "Fundamentals"), ("News", "News"), ("Macro", "Macro")]),
    ("Output", [("Data & exports", "Data")]),
    ("Tools", [("Live Market Watch", "@live"), ("Compare portfolios", "@compare")]),
]

_ROLE_KEY = Qt.UserRole + 1


class NavPanel(QFrame):
    """Grouped navigation list. Emits ``pageSelected(key)`` for result pages and
    ``actionTriggered(key)`` for tool entries (keys starting with '@')."""

    pageSelected = Signal(str)
    actionTriggered = Signal(str)
    WIDTH = 188

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("navPanel")
        self.setFixedWidth(self.WIDTH)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 8, 0, 8)
        self.list = QListWidget()
        self.list.setObjectName("navList")
        self.list.setFocusPolicy(Qt.StrongFocus)
        self.list.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        lay.addWidget(self.list)
        self._items: dict[str, QListWidgetItem] = {}
        for group, entries in NAV_GROUPS:
            head = QListWidgetItem(group.upper())
            head.setFlags(Qt.NoItemFlags)
            f = head.font()
            f.setPointSizeF(max(6.0, f.pointSizeF() * 0.78))
            f.setBold(True)
            head.setFont(f)
            head.setForeground(_qcolor(theme.ACTIVE.text_muted))
            head.setSizeHint(head.sizeHint().expandedTo(_qsize(0, 26)))
            self.list.addItem(head)
            for label, key in entries:
                it = QListWidgetItem(label)
                it.setData(_ROLE_KEY, key)
                self.list.addItem(it)
                self._items[key] = it
        self._items["Macro"].setHidden(True)  # shown once FRED data is available
        self._current = "Overview"
        self.list.itemClicked.connect(self._on_clicked)
        self.list.currentItemChanged.connect(lambda cur, _prev: self._on_clicked(cur))
        self.select("Overview")

    def keys(self) -> list[str]:
        return list(self._items)

    def _on_clicked(self, item: Optional[QListWidgetItem]) -> None:
        if item is None:
            return
        key = item.data(_ROLE_KEY)
        if not key:
            return
        if key.startswith("@"):
            self.actionTriggered.emit(key[1:])
            self.select(self._current)  # keep the page highlight where it was
            return
        self._current = key
        self.pageSelected.emit(key)

    def select(self, key: str) -> None:
        it = self._items.get(key)
        if it is None or it.isHidden():
            return
        self._current = key
        self.list.blockSignals(True)
        self.list.setCurrentItem(it)
        self.list.blockSignals(False)

    def current(self) -> str:
        return self._current

    def set_available(self, key: str, available: bool) -> None:
        it = self._items.get(key)
        if it is not None:
            it.setHidden(not available)

    def retheme(self) -> None:
        for i in range(self.list.count()):
            it = self.list.item(i)
            if not it.data(_ROLE_KEY):
                it.setForeground(_qcolor(theme.ACTIVE.text_muted))


def _qcolor(c: str):
    from PySide6.QtGui import QColor
    return QColor(c)


def _qsize(w: int, h: int):
    from PySide6.QtCore import QSize
    return QSize(w, h)


# ── KPI strip ──────────────────────────────────────────────────────────────


class _KpiCell(QFrame):
    def __init__(self, label: str, info_key: Optional[str]):
        super().__init__()
        self.setObjectName("kpiCell")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 7, 12, 7)
        lay.setSpacing(1)
        self.label = QLabel(label.upper())
        self.label.setObjectName("kpiLabel")
        if info_key:
            self.setToolTip(tooltip_html(info_key))
        self.value = QLabel("—")
        self.value.setObjectName("kpiValue")
        self.sub = QLabel(" ")
        self.sub.setObjectName("kpiSub")
        for w in (self.label, self.value, self.sub):
            lay.addWidget(w)

    def set(self, value: str, sub: str = "", tone: Optional[str] = None) -> None:
        self.value.setText(value)
        self.sub.setText(sub or " ")
        t = theme.ACTIVE
        color = t.green if tone == "pos" else t.red if tone == "neg" else None
        self.value.setStyleSheet(f"color: {color};" if color else "")


class KpiStrip(QFrame):
    """Eight headline numbers. ``set_metric(name, value, sub, tone)``."""

    NAMES = ["NAV", "CAGR", "Excess / yr", "Volatility", "Sharpe", "Max drawdown",
             "Beta", "Tracking error"]
    INFO_KEYS = {"CAGR": "total_return", "Sharpe": "sharpe", "Max drawdown": "max_drawdown",
                 "Volatility": "volatility", "Excess / yr": "alpha", "Beta": "beta"}

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("kpiStrip")
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)
        self._cells: dict[str, _KpiCell] = {}
        self._build()

    def _build(self) -> None:
        for name in self.NAMES:
            cell = _KpiCell(name, self.INFO_KEYS.get(name))
            self._cells[name] = cell
            self._layout.addWidget(cell, 1)

    def cell(self, name: str) -> _KpiCell:
        return self._cells[name]

    def set_metric(self, name: str, value: str, sub: str = "", tone: Optional[str] = None) -> None:
        if name in self._cells:
            self._cells[name].set(value, sub, tone)

    def clear(self) -> None:
        for c in self._cells.values():
            c.set("—")

    def retheme(self) -> None:
        while self._layout.count():
            w = self._layout.takeAt(0).widget()
            if w is not None:
                w.deleteLater()
        self._cells.clear()
        self._build()


# ── Data status ────────────────────────────────────────────────────────────


def data_status_text(results) -> str:
    """'Data: Yahoo Finance · as of 31 Dec 2025 · 1,508 trading days · 0 gaps · Rf …'"""
    from src.analytics.summary import data_quality

    if results is None:
        return "Data: Yahoo Finance · no analysis run yet"
    q = data_quality(getattr(results, "prices", None))
    parts = ["Data: Yahoo Finance"]
    if q["as_of"] is not None:
        parts.append(f"as of {q['as_of']:%d %b %Y}")
    parts.append(f"{q['days']:,} trading days")
    parts.append(f"{q['gaps']:,} gap{'s' if q['gaps'] != 1 else ''}")
    rf = getattr(results, "risk_free", None)
    if rf is not None:
        parts.append(f"Rf {rf.rate:.2%} ({rf.source})")
    n_warn = len(getattr(results, "warnings", []) or [])
    if n_warn:
        parts.append(f"{n_warn} warning{'s' if n_warn != 1 else ''}")
    return " · ".join(parts)


class DataStatus(QLabel):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(data_status_text(None), parent)
        self.setObjectName("dataStatus")

    def update_from(self, results) -> None:
        self.setText(data_status_text(results))
