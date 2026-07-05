"""Price-flash for the live quote tables.

When a row's price changes on a refresh, its background pulses green (up) or red
(down) and fades back over ~half a second — the classic trading-terminal cue that
draws the eye to exactly what just moved. Flashes are keyed by *symbol* (not row
index) so they land correctly even if the table re-sorts, and a single shared
QTimer drives the fade for all active rows.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QTimer
from PySide6.QtGui import QColor

from .. import theme


def _rgb(hex_color: str):
    h = (hex_color or "").lstrip("#")
    if len(h) != 6:
        return (128, 128, 128)
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


class RowFlasher:
    """Fades a green/red highlight across a table row on a value change."""

    def __init__(self, table, sym_col: int = 0, columns: Optional[list] = None,
                 steps: int = 12, interval_ms: int = 40, peak_alpha: int = 90) -> None:
        self._table = table
        self._sym_col = sym_col
        self._columns = columns          # None = whole row
        self._steps = max(1, steps)
        self._peak = peak_alpha
        self._active: dict[str, list] = {}   # symbol -> [remaining_steps, up]
        self._timer = QTimer(table)
        self._timer.setInterval(interval_ms)
        self._timer.timeout.connect(self._tick)

    def flash(self, symbol: str, up: bool) -> None:
        self._active[symbol] = [self._steps, bool(up)]
        self._paint(symbol, self._steps, bool(up))
        if not self._timer.isActive():
            self._timer.start()

    def clear(self) -> None:
        for sym in list(self._active):
            self._paint(sym, 0, self._active[sym][1])
        self._active.clear()
        self._timer.stop()

    # ── internals ──
    def _tick(self) -> None:
        finished = []
        for sym, st in self._active.items():
            st[0] -= 1
            self._paint(sym, max(0, st[0]), st[1])
            if st[0] <= 0:
                finished.append(sym)
        for sym in finished:
            self._active.pop(sym, None)
        if not self._active:
            self._timer.stop()

    def _row_of(self, symbol: str) -> int:
        for r in range(self._table.rowCount()):
            it = self._table.item(r, self._sym_col)
            if it is not None and it.text() == symbol:
                return r
        return -1

    def _paint(self, symbol: str, remaining: int, up: bool) -> None:
        row = self._row_of(symbol)
        if row < 0:
            return
        t = theme.ACTIVE
        r, g, b = _rgb(t.green if up else t.red)
        color = QColor(r, g, b, int(self._peak * remaining / self._steps))
        cols = self._columns if self._columns is not None else range(self._table.columnCount())
        for c in cols:
            it = self._table.item(row, c)
            if it is not None:
                it.setBackground(color)
