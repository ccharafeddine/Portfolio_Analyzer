"""A clickable circular '?' help badge that shows a plain-English explanation.

Hover shows it as a tooltip; clicking shows it immediately (reliable even when
hover is finicky). Used for the help badges on metric cards and config settings.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QToolTip, QWidget

from .. import theme


class InfoLabel(QLabel):
    def __init__(self, html: str, parent: Optional[QWidget] = None) -> None:
        super().__init__("?", parent)
        self._html = html
        self.setToolTip(html)
        self.setCursor(Qt.WhatsThisCursor)
        self.setAlignment(Qt.AlignCenter)
        self._apply_style()

    def _apply_style(self) -> None:
        t = theme.ACTIVE
        size = max(15, round(t.base_pt * 1.35))
        self.setFixedSize(size, size)
        self.setStyleSheet(
            f"color: {t.text_muted}; border: 1px solid {t.text_muted};"
            f" border-radius: {size // 2}px; font-size: {max(9, t.base_pt - 3)}px;"
            " font-weight: 700; background: transparent;"
        )

    def enterEvent(self, event) -> None:  # noqa: N802 (Qt override)
        # Show the explanation immediately on hover. Relying on Qt's built-in
        # tooltip timer proved unreliable here, so we drive QToolTip directly.
        if self._html:
            pos = self.mapToGlobal(self.rect().bottomLeft())
            QToolTip.showText(pos, self._html, self)
        super().enterEvent(event)

    def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt override)
        if self._html:
            QToolTip.showText(event.globalPosition().toPoint(), self._html, self)
        super().mousePressEvent(event)
