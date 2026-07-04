"""Cost-basis editor — set average cost per share for each holding.

Opened from Live Market Watch (the "set cost basis" link or a row's right-click
menu). Returns ``{ticker: cost}`` for the holdings the user filled in; the main
window applies it to the config, refreshes the live P&L, offers to save the
portfolio, and re-runs the analysis (cost basis feeds the tax/return figures).
"""

from __future__ import annotations

from typing import Callable, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)


class CostBasisDialog(QDialog):
    def __init__(
        self,
        tickers: list,
        existing: Optional[dict] = None,
        get_price: Optional[Callable[[str], Optional[float]]] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Set Cost Basis")
        self.setMinimumWidth(360)
        existing = existing or {}

        root = QVBoxLayout(self)
        intro = QLabel(
            "Average cost per share for each holding. Used for unrealized P&L in "
            "Live Market Watch and for tax figures in the analysis. Leave a field at "
            "0 to skip it."
        )
        intro.setWordWrap(True)
        root.addWidget(intro)

        # Scroll in case of a large universe.
        area = QScrollArea()
        area.setWidgetResizable(True)
        inner = QWidget()
        form = QFormLayout(inner)
        form.setSpacing(8)
        self._spins: dict[str, QDoubleSpinBox] = {}
        for t in tickers:
            t = str(t).upper()
            spin = QDoubleSpinBox()
            spin.setRange(0.0, 1_000_000_000.0)
            spin.setDecimals(2)
            spin.setPrefix("$ ")
            if t in existing:
                spin.setValue(float(existing[t]))
            elif get_price:
                price = get_price(t)
                if isinstance(price, (int, float)) and price > 0:
                    spin.setValue(float(price))
            self._spins[t] = spin
            form.addRow(t, spin)
        area.setWidget(inner)
        root.addWidget(area, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.Save).setDefault(True)
        root.addWidget(buttons)

    def values(self) -> dict:
        """Return ``{ticker: cost}`` for the fields with a positive value."""
        return {t: s.value() for t, s in self._spins.items() if s.value() > 0}
