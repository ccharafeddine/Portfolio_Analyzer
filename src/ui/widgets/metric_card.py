"""Headline metric cards (Total Return, Sharpe, etc.) for the top strip."""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from .. import theme
from ..explanations import tooltip_html
from .info_label import InfoLabel


class MetricCard(QFrame):
    """A single metric: uppercase label, big mono value, optional delta.

    An optional ``info_key`` adds a small (i) with a plain-English explanation.
    """

    def __init__(
        self, label: str, info_key: Optional[str] = None, parent: Optional[QWidget] = None
    ) -> None:
        super().__init__(parent)
        t = theme.ACTIVE
        self._mono = t.mono
        self.setStyleSheet(
            f"MetricCard {{ background-color: {t.card}; border: 1px solid {t.border_light};"
            f" border-radius: {t.radius}px; }}"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(t.card_pad_h, t.card_pad_v, t.card_pad_h, t.card_pad_v)
        layout.setSpacing(4)

        label_row = QHBoxLayout()
        label_row.setSpacing(5)
        self._label = QLabel(label.upper())
        self._label.setStyleSheet(
            f"color: {t.text_muted}; font-size: {t.label_pt}px; font-weight: 600;"
            " letter-spacing: 0.06em; background: transparent;"
        )
        label_row.addWidget(self._label)
        if info_key:
            label_row.addWidget(InfoLabel(tooltip_html(info_key)))
        label_row.addStretch(1)

        row = QHBoxLayout()
        row.setSpacing(8)
        self._value = QLabel("—")
        self._value.setStyleSheet(
            f"color: {t.text}; font-size: {t.metric_pt}px; font-weight: 700;"
            f" font-family: {t.mono}; background: transparent;"
        )
        self._delta = QLabel("")
        self._delta.setStyleSheet(
            f"font-size: {t.base_pt}px; font-family: {t.mono}; background: transparent;"
        )
        self._delta.setAlignment(Qt.AlignBottom)
        row.addWidget(self._value)
        row.addWidget(self._delta)
        row.addStretch(1)

        layout.addLayout(label_row)
        layout.addLayout(row)

    def set_value(self, value: str, delta: Optional[str] = None) -> None:
        self._value.setText(value)
        if delta:
            t = theme.ACTIVE
            color = t.green if delta.startswith("+") else t.red
            self._delta.setStyleSheet(
                f"color: {color}; font-size: {t.base_pt}px; font-family: {self._mono};"
                " background: transparent;"
            )
            self._delta.setText(delta)
        else:
            self._delta.setText("")


class MetricStrip(QWidget):
    """A row of headline metric cards keyed by name."""

    NAMES = ["Total Return", "Sharpe Ratio", "Max Drawdown", "Volatility", "Alpha", "Beta"]
    INFO_KEYS = {
        "Total Return": "total_return",
        "Sharpe Ratio": "sharpe",
        "Max Drawdown": "max_drawdown",
        "Volatility": "volatility",
        "Alpha": "alpha",
        "Beta": "beta",
    }

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(12)
        self._cards: dict[str, MetricCard] = {}
        self._build_cards()

    def _build_cards(self) -> None:
        for name in self.NAMES:
            card = MetricCard(name, info_key=self.INFO_KEYS.get(name))
            self._cards[name] = card
            self._layout.addWidget(card)

    def set_metric(self, name: str, value: str, delta: Optional[str] = None) -> None:
        if name in self._cards:
            self._cards[name].set_value(value, delta)

    def clear(self) -> None:
        for card in self._cards.values():
            card.set_value("—")

    def retheme(self) -> None:
        """Rebuild the cards with the active theme (values are re-applied by the
        caller)."""
        while self._layout.count():
            item = self._layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._cards.clear()
        self._build_cards()
