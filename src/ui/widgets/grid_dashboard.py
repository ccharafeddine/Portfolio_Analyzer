"""A free-form, snap-to-grid dashboard of draggable/resizable card panels.

Each panel is a distinct card (:class:`GridPanel`) with a header you drag to move
it and a bottom-right grip you drag to resize it. Movement and sizing snap to a
fixed-column grid; other panels reflow out of the way (see :mod:`grid_layout`).
The grid is fluid — columns divide the width and rows divide the height, so the
whole dashboard always fills its viewport (panels shrink rather than scroll when
many rows are stacked).

During a drag/resize the panel's content is hidden and a dashed placeholder shows
the snapped target, so dragging stays smooth even for web-view panels (charts).
Layout (positions, sizes, hidden panels) round-trips via :meth:`save_layout` /
:meth:`restore_layout`.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from .. import theme
from .grid_layout import DEFAULT_COLS, GridItem, bottom_row, clamp, find_free, resolve

_MIN_W = 2
_MIN_H = 2
_BASE_ROWS = 12  # the layout height the default grid is designed to fill
_GAP = 8


class _DragZone(QWidget):
    """A bare widget that reports left-button press/drag/release in global coords.
    Child widgets (buttons, badges) still get their own clicks — only presses on
    the bare zone start a drag."""

    pressed = Signal(QPoint)
    moved = Signal(QPoint)
    released = Signal()

    def __init__(self, cursor: Qt.CursorShape, parent=None) -> None:
        super().__init__(parent)
        self.setCursor(cursor)
        self._active = False

    def mousePressEvent(self, e) -> None:  # noqa: N802
        if e.button() == Qt.LeftButton:
            self._active = True
            self.pressed.emit(e.globalPosition().toPoint())
            e.accept()

    def mouseMoveEvent(self, e) -> None:  # noqa: N802
        if self._active:
            self.moved.emit(e.globalPosition().toPoint())
            e.accept()

    def mouseReleaseEvent(self, e) -> None:  # noqa: N802
        if self._active and e.button() == Qt.LeftButton:
            self._active = False
            self.released.emit()
            e.accept()


class GridPanel(QFrame):
    """One card: a draggable header (title + optional extra widget) over content,
    with a resize grip in the bottom-right corner."""

    dragStart = Signal(str, QPoint)
    dragMove = Signal(str, QPoint)
    dragEnd = Signal(str)
    resizeStart = Signal(str, QPoint)
    resizeMove = Signal(str, QPoint)
    resizeEnd = Signal(str)

    def __init__(self, panel_id: str, title: str, content: QWidget,
                 header_extra: Optional[QWidget] = None, parent=None) -> None:
        super().__init__(parent)
        self.panel_id = panel_id
        self.setObjectName("gridPanel")
        self._content = content

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Header = drag zone with a title and an optional right-aligned widget.
        self._header = _DragZone(Qt.OpenHandCursor)
        self._header.setObjectName("gridPanelHeader")
        hb = QHBoxLayout(self._header)
        hb.setContentsMargins(10, 5, 8, 5)
        hb.setSpacing(6)
        self._title = QLabel(title)
        self._title.setObjectName("gridPanelTitle")
        hb.addWidget(self._title)
        hb.addStretch(1)
        if header_extra is not None:
            hb.addWidget(header_extra)
        root.addWidget(self._header)

        body = QWidget()
        bl = QVBoxLayout(body)
        bl.setContentsMargins(6, 6, 6, 6)
        bl.setSpacing(0)
        bl.addWidget(content)
        root.addWidget(body, 1)
        self._body = body

        # Resize grip, positioned manually in resizeEvent (a floating child).
        self._grip = _DragZone(Qt.SizeFDiagCursor, self)
        self._grip.setObjectName("gridPanelGrip")
        self._grip.setFixedSize(16, 16)

        self._header.pressed.connect(lambda p: self.dragStart.emit(self.panel_id, p))
        self._header.moved.connect(lambda p: self.dragMove.emit(self.panel_id, p))
        self._header.released.connect(lambda: self.dragEnd.emit(self.panel_id))
        self._grip.pressed.connect(lambda p: self.resizeStart.emit(self.panel_id, p))
        self._grip.moved.connect(lambda p: self.resizeMove.emit(self.panel_id, p))
        self._grip.released.connect(lambda: self.resizeEnd.emit(self.panel_id))
        self.retheme()

    def resizeEvent(self, e) -> None:  # noqa: N802
        self._grip.move(self.width() - self._grip.width() - 2,
                        self.height() - self._grip.height() - 2)
        self._grip.raise_()
        super().resizeEvent(e)

    def set_dragging(self, on: bool) -> None:
        """Hide the content while moving/resizing so the drag stays smooth."""
        self._body.setVisible(not on)

    def set_title(self, text: str) -> None:
        self._title.setText(text)

    def retheme(self) -> None:
        t = theme.ACTIVE
        self.setStyleSheet(
            f"QFrame#gridPanel {{ background:{t.card}; border:1px solid {t.border_light};"
            f" border-radius:{max(6, t.radius - 2)}px; }}"
            f"QWidget#gridPanelHeader {{ background:{t.panel};"
            f" border-top-left-radius:{max(6, t.radius - 2)}px;"
            f" border-top-right-radius:{max(6, t.radius - 2)}px;"
            f" border-bottom:1px solid {t.border}; }}"
            f"QLabel#gridPanelTitle {{ color:{t.text_muted}; font-size:{t.label_pt}px;"
            f" font-weight:700; letter-spacing:0.05em; background:transparent; }}"
            f"QWidget#gridPanelGrip {{ background:transparent;"
            f" border-right:2px solid {t.text_muted}; border-bottom:2px solid {t.text_muted};"
            f" border-bottom-right-radius:{max(6, t.radius - 2)}px; }}"
        )


class GridDashboard(QWidget):
    """Container that places :class:`GridPanel` cards on a fluid snap grid."""

    layoutChanged = Signal()

    def __init__(self, cols: int = DEFAULT_COLS, parent=None) -> None:
        super().__init__(parent)
        self._cols = cols
        self._items: dict[str, GridItem] = {}       # visible panels only
        self._panels: dict[str, GridPanel] = {}
        self._hidden: dict[str, GridItem] = {}       # remembered cells while hidden
        self._order: list[str] = []
        self._drag: Optional[dict] = None

        self._placeholder = QFrame(self)
        self._placeholder.setObjectName("gridPlaceholder")
        self._placeholder.hide()
        self._retheme_placeholder()

    # ── Building ──
    def add_panel(self, panel_id: str, title: str, content: QWidget,
                  w: int, h: int, x: Optional[int] = None, y: Optional[int] = None,
                  header_extra: Optional[QWidget] = None) -> GridPanel:
        if x is None or y is None:
            x, y = find_free(list(self._items.values()), w, h, self._cols)
        item = clamp(GridItem(panel_id, x, y, w, h), self._cols, _MIN_W, _MIN_H)
        self._items[panel_id] = item
        panel = GridPanel(panel_id, title, content, header_extra=header_extra, parent=self)
        panel.dragStart.connect(self._on_drag_start)
        panel.dragMove.connect(self._on_drag_move)
        panel.dragEnd.connect(self._on_drag_end)
        panel.resizeStart.connect(self._on_resize_start)
        panel.resizeMove.connect(self._on_resize_move)
        panel.resizeEnd.connect(self._on_resize_end)
        self._panels[panel_id] = panel
        self._order.append(panel_id)
        panel.show()
        self._relayout()
        return panel

    def panel(self, panel_id: str) -> Optional[GridPanel]:
        return self._panels.get(panel_id)

    # ── Visibility ──
    def is_visible(self, panel_id: str) -> bool:
        return panel_id in self._items

    def set_panel_visible(self, panel_id: str, visible: bool) -> None:
        panel = self._panels.get(panel_id)
        if panel is None:
            return
        if visible and panel_id not in self._items:
            item = self._hidden.pop(panel_id, None) or GridItem(panel_id, 0, 0, 6, 6)
            self._items[panel_id] = item
            self._items = {k: v for k, v in self._items.items()}
            resolve(list(self._items.values()), panel_id)
            panel.show()
            self._relayout()
            self.layoutChanged.emit()
        elif not visible and panel_id in self._items:
            self._hidden[panel_id] = self._items.pop(panel_id)
            panel.hide()
            self._relayout()
            self.layoutChanged.emit()

    # ── Geometry ──
    def _row_h(self) -> float:
        rows = max(_BASE_ROWS, bottom_row(list(self._items.values())))
        return self.height() / max(1, rows)

    def _col_w(self) -> float:
        return self.width() / max(1, self._cols)

    def _cell_rect(self, item: GridItem) -> QRect:
        cw, rh = self._col_w(), self._row_h()
        g = _GAP // 2
        left = round(item.x * cw) + g
        top = round(item.y * rh) + g
        right = round((item.x + item.w) * cw) - g
        bottom = round((item.y + item.h) * rh) - g
        return QRect(left, top, max(40, right - left), max(30, bottom - top))

    def _relayout(self) -> None:
        for pid, item in self._items.items():
            self._panels[pid].setGeometry(self._cell_rect(item))

    def resizeEvent(self, e) -> None:  # noqa: N802
        self._relayout()
        super().resizeEvent(e)

    # ── Drag (move) ──
    def _on_drag_start(self, pid: str, gpos: QPoint) -> None:
        panel = self._panels.get(pid)
        if panel is None or pid not in self._items:
            return
        top_left = panel.pos()
        cursor = self.mapFromGlobal(gpos)
        self._drag = {"mode": "move", "id": pid,
                      "offset": cursor - top_left}
        panel.set_dragging(True)
        panel.raise_()
        self._show_placeholder(self._items[pid])

    def _on_drag_move(self, pid: str, gpos: QPoint) -> None:
        if not self._drag or self._drag["id"] != pid:
            return
        panel = self._panels[pid]
        cursor = self.mapFromGlobal(gpos)
        tl = cursor - self._drag["offset"]
        panel.move(tl)
        item = self._items[pid]
        sx = max(0, min(round(tl.x() / self._col_w()), self._cols - item.w))
        sy = max(0, round(tl.y() / self._row_h()))
        self._show_placeholder(GridItem(pid, sx, sy, item.w, item.h))
        self._drag["target"] = (sx, sy)

    def _on_drag_end(self, pid: str) -> None:
        if not self._drag or self._drag["id"] != pid:
            return
        sx, sy = self._drag.get("target", (self._items[pid].x, self._items[pid].y))
        item = self._items[pid]
        item.x, item.y = sx, sy
        resolve(list(self._items.values()), pid)
        self._end_drag(pid)

    # ── Resize ──
    def _on_resize_start(self, pid: str, gpos: QPoint) -> None:
        panel = self._panels.get(pid)
        if panel is None or pid not in self._items:
            return
        self._drag = {"mode": "resize", "id": pid,
                      "start": self.mapFromGlobal(gpos),
                      "size0": panel.size()}
        panel.set_dragging(True)
        panel.raise_()
        self._show_placeholder(self._items[pid])

    def _on_resize_move(self, pid: str, gpos: QPoint) -> None:
        if not self._drag or self._drag["id"] != pid:
            return
        panel = self._panels[pid]
        delta = self.mapFromGlobal(gpos) - self._drag["start"]
        new_w = max(40, self._drag["size0"].width() + delta.x())
        new_h = max(30, self._drag["size0"].height() + delta.y())
        panel.resize(new_w, new_h)
        item = self._items[pid]
        sw = max(_MIN_W, min(round(new_w / self._col_w()), self._cols - item.x))
        sh = max(_MIN_H, round(new_h / self._row_h()))
        self._show_placeholder(GridItem(pid, item.x, item.y, sw, sh))
        self._drag["target"] = (sw, sh)

    def _on_resize_end(self, pid: str) -> None:
        if not self._drag or self._drag["id"] != pid:
            return
        item = self._items[pid]
        sw, sh = self._drag.get("target", (item.w, item.h))
        item.w, item.h = sw, sh
        clamp(item, self._cols, _MIN_W, _MIN_H)
        resolve(list(self._items.values()), pid)
        self._end_drag(pid)

    def _end_drag(self, pid: str) -> None:
        self._panels[pid].set_dragging(False)
        self._placeholder.hide()
        self._drag = None
        self._relayout()
        self.layoutChanged.emit()

    # ── Placeholder ──
    def _show_placeholder(self, item: GridItem) -> None:
        self._placeholder.setGeometry(self._cell_rect(item))
        self._placeholder.show()
        self._placeholder.raise_()
        self._panels[item.id].raise_()

    def _retheme_placeholder(self) -> None:
        t = theme.ACTIVE
        self._placeholder.setStyleSheet(
            f"QFrame#gridPlaceholder {{ background:{t.accent}22;"
            f" border:2px dashed {t.accent}; border-radius:{max(6, t.radius - 2)}px; }}"
        )

    # ── Persistence ──
    def save_layout(self) -> dict:
        data = {pid: list(it.as_tuple()) for pid, it in self._items.items()}
        data["__hidden__"] = list(self._hidden.keys())
        return data

    def restore_layout(self, data: dict) -> None:
        if not isinstance(data, dict):
            return
        hidden = set(data.get("__hidden__", []) or [])
        new_items: dict[str, GridItem] = {}
        for pid, panel in self._panels.items():
            cell = data.get(pid)
            if pid in hidden:
                panel.hide()
                if isinstance(cell, (list, tuple)) and len(cell) == 4:
                    self._hidden[pid] = GridItem(pid, *(int(v) for v in cell))
                continue
            if isinstance(cell, (list, tuple)) and len(cell) == 4:
                new_items[pid] = clamp(GridItem(pid, *(int(v) for v in cell)),
                                       self._cols, _MIN_W, _MIN_H)
                panel.show()
            elif pid in self._items:
                new_items[pid] = self._items[pid]
        if new_items:
            self._items = new_items
        self._relayout()

    def retheme(self) -> None:
        self._retheme_placeholder()
        for panel in self._panels.values():
            panel.retheme()
