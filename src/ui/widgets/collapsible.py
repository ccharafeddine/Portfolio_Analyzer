"""A collapsible disclosure section: a clickable header that shows/hides its body.

Used to tuck advanced configuration away from everyday users.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QToolButton, QVBoxLayout, QWidget


class CollapsibleSection(QWidget):
    def __init__(self, title: str, expanded: bool = False, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(6)

        self._header = QToolButton()
        self._header.setObjectName("collapsibleHeader")
        self._header.setText(title.upper())
        self._header.setCheckable(True)
        self._header.setChecked(expanded)
        self._header.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._header.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self._header.setCursor(Qt.PointingHandCursor)
        self._header.clicked.connect(self._on_toggle)
        outer.addWidget(self._header)

        self._body = QWidget()
        self.body_layout = QVBoxLayout(self._body)
        self.body_layout.setContentsMargins(2, 2, 2, 4)
        self.body_layout.setSpacing(8)
        self._body.setVisible(expanded)
        outer.addWidget(self._body)

    def _on_toggle(self, checked: bool) -> None:
        self._header.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
        self._body.setVisible(checked)

    def add_widget(self, widget: QWidget) -> None:
        self.body_layout.addWidget(widget)

    def add_layout(self, layout) -> None:
        self.body_layout.addLayout(layout)
