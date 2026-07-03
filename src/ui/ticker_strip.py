"""Always-on ticker strip for the bottom status bar.

A slow horizontal marquee of the analyzed portfolio's holdings (symbol, last
price, day change %), colored by the active theme's green/red. It lives in the
status bar and is shown whenever the bar isn't hosting the analysis progress
bar. Data is delayed (see ``market_data.fetch_quotes``); the strip only renders
whatever quotes it's handed via :meth:`set_quotes`.

Rendering is a custom ``paintEvent`` so the scroll is continuous and the colors
follow the theme exactly. Motion pauses while the pointer hovers, and clicking a
symbol emits :attr:`symbolClicked` (the main window opens it in Live Market
Watch).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QFont, QFontMetrics, QPainter
from PySide6.QtWidgets import QWidget

from . import theme


@dataclass
class _Segment:
    symbol: str
    sym_text: str
    detail_text: str
    detail_color: str
    width: int          # full advance width incl. trailing gap
    sym_width: int      # width of the symbol portion (for detail placement)


_GAP = 46          # px between one holding and the next
_SPEED = 1.0       # px per frame (~30 fps) — deliberately unhurried
_FRAME_MS = 33


class TickerStrip(QWidget):
    symbolClicked = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._quotes: dict = {}
        self._order: list[str] = []
        self._segments: list[_Segment] = []
        self._cum: list[int] = []      # cumulative start x of each segment
        self._total = 0
        self._offset = 0.0
        self._hover = False
        self.setMouseTracking(True)
        self.setCursor(Qt.PointingHandCursor)

        self._apply_font()
        self._timer = QTimer(self)
        self._timer.setInterval(_FRAME_MS)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    # ── Public API ──
    def set_quotes(self, quotes: dict, order: Optional[list[str]] = None) -> None:
        """Update the displayed quotes. ``order`` fixes the display order (falls
        back to the quotes' own key order / previous order)."""
        self._quotes = dict(quotes or {})
        if order:
            self._order = [t for t in order if t in self._quotes]
        else:
            self._order = list(self._quotes.keys())
        self._rebuild_segments()
        self.update()

    def clear(self) -> None:
        self._quotes = {}
        self._order = []
        self._segments = []
        self._cum = []
        self._total = 0
        self._offset = 0.0
        self.update()

    def retheme(self) -> None:
        self._apply_font()
        self._rebuild_segments()
        self.update()

    # ── Motion ──
    def _tick(self) -> None:
        if self._hover or self._total <= 0 or not self.isVisible():
            return
        self._offset += _SPEED
        if self._offset >= self._total:
            self._offset -= self._total
        self.update()

    def enterEvent(self, event) -> None:  # noqa: N802 (Qt override)
        self._hover = True
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802
        self._hover = False
        super().leaveEvent(event)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton and self._total > 0:
            sym = self._symbol_at(event.position().x())
            if sym:
                self.symbolClicked.emit(sym)
        super().mousePressEvent(event)

    # ── Layout / geometry ──
    def _apply_font(self) -> None:
        t = theme.ACTIVE
        self._font = QFont()
        self._font.setPixelSize(max(11, int(t.base_pt)))
        self._fm = QFontMetrics(self._font)
        self.setFixedHeight(self._fm.height() + 8)

    def _rebuild_segments(self) -> None:
        t = theme.ACTIVE
        fm = self._fm
        segs: list[_Segment] = []
        for sym in self._order:
            q = self._quotes.get(sym)
            sym_text = f"{sym} "
            if q is None or not getattr(q, "ok", False):
                detail = "—"
                color = t.text_muted
            else:
                last = q.last
                pct = q.change_pct
                price = f"{last:,.2f}" if last is not None else "—"
                if pct is None:
                    detail = price
                    color = t.text_slate
                else:
                    arrow = "▲" if pct >= 0 else "▼"
                    sign = "+" if pct >= 0 else ""
                    detail = f"{price} {arrow}{sign}{pct * 100:.2f}%"
                    color = t.green if pct >= 0 else t.red
            sym_w = fm.horizontalAdvance(sym_text)
            det_w = fm.horizontalAdvance(detail)
            segs.append(
                _Segment(
                    symbol=sym,
                    sym_text=sym_text,
                    detail_text=detail,
                    detail_color=color,
                    width=sym_w + det_w + _GAP,
                    sym_width=sym_w,
                )
            )
        self._segments = segs
        self._cum = []
        run = 0
        for s in segs:
            self._cum.append(run)
            run += s.width
        self._total = run
        if self._total > 0:
            self._offset %= self._total

    def _symbol_at(self, x: float) -> Optional[str]:
        if self._total <= 0:
            return None
        pos = (self._offset + x) % self._total
        for i, start in enumerate(self._cum):
            if start <= pos < start + self._segments[i].width:
                return self._segments[i].symbol
        return None

    # ── Paint ──
    def paintEvent(self, event) -> None:  # noqa: N802
        t = theme.ACTIVE
        painter = QPainter(self)
        painter.fillRect(self.rect(), _qcolor(t.panel))
        painter.setFont(self._font)
        base_y = (self.height() + self._fm.ascent() - self._fm.descent()) // 2

        if self._total <= 0:
            painter.setPen(_qcolor(t.text_muted))
            painter.drawText(10, base_y, "Waiting for quotes…")
            painter.end()
            return

        width = self.width()
        x = -(self._offset % self._total)
        # Repeat the segment set until the visible width is filled.
        guard = 0
        while x < width and guard < 10000:
            for seg in self._segments:
                if x + seg.width > 0 and x < width:
                    self._draw_segment(painter, seg, int(x), base_y, t)
                x += seg.width
            guard += 1
        painter.end()

    def _draw_segment(self, painter, seg, x, y, t) -> None:
        painter.setPen(_qcolor(t.text))
        painter.drawText(x, y, seg.sym_text)
        painter.setPen(_qcolor(seg.detail_color))
        painter.drawText(x + seg.sym_width, y, seg.detail_text)
        # A muted separator dot before the next holding.
        painter.setPen(_qcolor(t.text_muted))
        painter.drawText(x + seg.width - int(_GAP * 0.62), y, "•")


def _qcolor(value: str):
    from PySide6.QtGui import QColor

    return QColor(value)
