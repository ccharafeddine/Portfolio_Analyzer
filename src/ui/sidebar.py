"""A collapsible left sidebar (like Claude's chat sidebar).

Wraps a content widget (the config panel). A toggle button collapses the sidebar
to a narrow rail or expands it back, with a short width animation. Unlike a
QDockWidget it never detaches into its own window.
"""

from __future__ import annotations

from PySide6.QtCore import (
    QEasingCurve,
    QParallelAnimationGroup,
    QPropertyAnimation,
    QSize,
    Qt,
)
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .assets import asset


class Sidebar(QFrame):
    EXPANDED_W = 400
    COLLAPSED_W = 46

    def __init__(self, content: QWidget, title: str = "Configuration") -> None:
        super().__init__()
        self.setObjectName("sidebar")
        # Pin the expanded width (both min and max) so the layout can't shrink it
        # below what the config panel needs.
        self.setFixedWidth(self.EXPANDED_W)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        header = QWidget()
        header.setObjectName("sidebarHeader")
        hb = QHBoxLayout(header)
        hb.setContentsMargins(12, 8, 6, 8)
        self._title = QLabel(title.upper())
        self._title.setObjectName("sidebarTitle")
        self._icon_left = QIcon(asset("chevron_left.svg"))
        self._icon_right = QIcon(asset("chevron_right.svg"))
        self._toggle = QToolButton()
        self._toggle.setObjectName("sidebarToggle")
        self._toggle.setIcon(self._icon_left)
        self._toggle.setIconSize(QSize(18, 18))
        self._toggle.setCursor(Qt.PointingHandCursor)
        self._toggle.setToolTip("Collapse / expand sidebar")
        hb.addWidget(self._title)
        hb.addStretch(1)
        hb.addWidget(self._toggle)
        outer.addWidget(header)

        self._content = content
        outer.addWidget(content, 1)

        self._collapsed = False
        # Animate both min and max width together so the widget stays pinned to a
        # definite width at every frame (and at rest).
        self._group = QParallelAnimationGroup(self)
        self._anim_min = QPropertyAnimation(self, b"minimumWidth", self)
        self._anim_max = QPropertyAnimation(self, b"maximumWidth", self)
        for a in (self._anim_min, self._anim_max):
            a.setDuration(160)
            a.setEasingCurve(QEasingCurve.InOutCubic)
            self._group.addAnimation(a)
        self._group.finished.connect(self._on_anim_done)

        self._toggle.clicked.connect(self.toggle)

    def toggle(self) -> None:
        self.set_collapsed(not self._collapsed)

    def set_collapsed(self, collapsed: bool) -> None:
        if collapsed == self._collapsed:
            return
        self._collapsed = collapsed
        start = self.width()
        end = self.COLLAPSED_W if collapsed else self.EXPANDED_W
        if collapsed:
            # Hide content immediately so it doesn't squish during the animation.
            self._content.setVisible(False)
            self._title.setVisible(False)
            self._toggle.setIcon(self._icon_right)
        else:
            self._title.setVisible(True)
            self._toggle.setIcon(self._icon_left)
        self._group.stop()
        for a in (self._anim_min, self._anim_max):
            a.setStartValue(start)
            a.setEndValue(end)
        self._group.start()

    def _on_anim_done(self) -> None:
        # Re-pin to a fixed width at rest.
        end = self.COLLAPSED_W if self._collapsed else self.EXPANDED_W
        self.setFixedWidth(end)
        if not self._collapsed:
            self._content.setVisible(True)
