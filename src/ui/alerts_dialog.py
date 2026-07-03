"""Price Alerts dialog — add / remove / enable per-ticker price crossings.

Alerts fire when a holding's (delayed or real-time) price crosses a threshold,
delivered as a desktop notification. This dialog only edits the ``AlertStore``;
the main window evaluates alerts on each quotes snapshot and does the notifying.
"""

from __future__ import annotations

from typing import Callable, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
)

from . import theme
from .alerts import AlertStore


class AlertsDialog(QDialog):
    def __init__(
        self,
        store: AlertStore,
        tickers: Optional[list] = None,
        get_price: Optional[Callable[[str], Optional[float]]] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Price Alerts")
        self.setMinimumWidth(460)
        self._store = store
        self._get_price = get_price
        self._loading = False

        root = QVBoxLayout(self)
        intro = QLabel(
            "Get a desktop notification when a holding's price crosses a level. "
            "Alerts use the same delayed/real-time quotes as Live Market Watch and "
            "fire once per crossing."
        )
        intro.setWordWrap(True)
        root.addWidget(intro)

        # Existing alerts — checkbox = enabled.
        self._list = QListWidget()
        self._list.itemChanged.connect(self._on_item_changed)
        root.addWidget(self._list, 1)

        remove_row = QHBoxLayout()
        remove_row.addStretch(1)
        self._remove_btn = QPushButton("Remove selected")
        self._remove_btn.setObjectName("secondary")
        self._remove_btn.clicked.connect(self._on_remove)
        remove_row.addWidget(self._remove_btn)
        root.addLayout(remove_row)

        # Add row.
        add = QHBoxLayout()
        self._ticker = QComboBox()
        self._ticker.setEditable(True)
        for t in (tickers or []):
            self._ticker.addItem(str(t).upper())
        self._ticker.setMinimumWidth(90)
        self._ticker.currentTextChanged.connect(self._prefill_price)
        self._dir = QComboBox()
        self._dir.addItems(["Above", "Below"])
        self._price = QDoubleSpinBox()
        self._price.setRange(0.0, 1_000_000_000.0)
        self._price.setDecimals(2)
        self._price.setMaximumWidth(140)
        self._add_btn = QPushButton("Add alert")
        self._add_btn.clicked.connect(self._on_add)
        add.addWidget(QLabel("Ticker"))
        add.addWidget(self._ticker)
        add.addWidget(self._dir)
        add.addWidget(QLabel("at"))
        add.addWidget(self._price)
        add.addWidget(self._add_btn)
        root.addLayout(add)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        buttons.clicked.connect(lambda _b: self.accept())
        root.addWidget(buttons)

        self._refresh()
        self._prefill_price(self._ticker.currentText())

    # ── List ──
    def _refresh(self) -> None:
        self._loading = True
        self._list.clear()
        alerts = self._store.all()
        if not alerts:
            placeholder = QListWidgetItem("No alerts yet — add one below.")
            placeholder.setFlags(Qt.NoItemFlags)
            placeholder.setForeground(_color(theme.ACTIVE.text_muted))
            self._list.addItem(placeholder)
        for a in alerts:
            item = QListWidgetItem(a.describe())
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if a.enabled else Qt.Unchecked)
            if not a.enabled:
                item.setForeground(_color(theme.ACTIVE.text_muted))
            self._list.addItem(item)
        self._loading = False

    def _on_item_changed(self, item: QListWidgetItem) -> None:
        if self._loading:
            return
        idx = self._list.row(item)
        if 0 <= idx < len(self._store.all()):
            self._store.set_enabled(idx, item.checkState() == Qt.Checked)
            self._refresh()

    def _on_remove(self) -> None:
        idx = self._list.currentRow()
        if 0 <= idx < len(self._store.all()):
            self._store.remove(idx)
            self._refresh()

    def _on_add(self) -> None:
        ticker = self._ticker.currentText().strip().upper()
        if not ticker or self._price.value() <= 0:
            return
        direction = self._dir.currentText()
        self._store.add(ticker, direction, self._price.value())
        self._refresh()

    def _prefill_price(self, ticker: str) -> None:
        if self._get_price and ticker:
            price = self._get_price(ticker.strip().upper())
            if isinstance(price, (int, float)) and price > 0:
                self._price.setValue(float(price))


def _color(value: str):
    from PySide6.QtGui import QColor

    return QColor(value)
