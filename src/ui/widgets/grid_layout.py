"""Pure grid-placement engine for the drag-and-drop dashboard (Qt-free).

Panels live on a fixed-column grid (``x, y`` = column/row origin; ``w, h`` =
column/row span). This module has no Qt dependency so the placement logic — the
error-prone part — can be unit-tested exhaustively. The Qt widget layer
(:mod:`grid_dashboard`) maps these integer cells to pixel geometry.

Collision model: when one item is moved/resized, every item it overlaps is pushed
straight down just far enough to clear it, cascading. The moved item stays put.
No auto-compaction — panels stay where the user drops them (free-form), so gaps
are allowed. ``y`` only ever increases during resolution, so it terminates.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_COLS = 12
# Hard bound on a card's row span/origin, so a corrupt or hostile saved layout
# can't carry an absurd height that degrades the whole grid to sub-pixel rows.
_MAX_ROWS = 200


@dataclass
class GridItem:
    id: str
    x: int
    y: int
    w: int
    h: int

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)


def overlaps(a: GridItem, b: GridItem) -> bool:
    return not (
        a.x + a.w <= b.x or b.x + b.w <= a.x
        or a.y + a.h <= b.y or b.y + b.h <= a.y
    )


def clamp(item: GridItem, cols: int = DEFAULT_COLS,
          min_w: int = 1, min_h: int = 1) -> GridItem:
    """Clamp an item to valid bounds: span within [min, cols], origin on-grid."""
    item.w = max(min_w, min(int(item.w), cols))
    # Bound height too (a hostile/corrupt saved layout could carry an absurd h).
    item.h = max(min_h, min(int(item.h), _MAX_ROWS))
    item.x = max(0, min(int(item.x), cols - item.w))
    item.y = max(0, min(int(item.y), _MAX_ROWS))
    return item


def resolve(items: list[GridItem], moved_id: str) -> list[GridItem]:
    """Push every item overlapping ``moved_id`` downward until nothing overlaps.
    The moved item never moves. Mutates and returns ``items``."""
    by_id = {it.id: it for it in items}
    if moved_id not in by_id:
        return items
    queue = [moved_id]
    # Safety cap: each item can only be pushed a bounded number of times since y
    # is monotonically increasing, but guard against pathological input anyway.
    guard = 0
    guard_max = 1000 + len(items) * len(items) * (sum(it.h for it in items) + 1)
    while queue and guard < guard_max:
        guard += 1
        cur = by_id[queue.pop(0)]
        for it in items:
            if it.id == cur.id or it.id == moved_id:
                continue
            if overlaps(cur, it):
                it.y = cur.y + cur.h  # drop it just below cur
                queue.append(it.id)
    return items


def find_free(items: list[GridItem], w: int, h: int,
              cols: int = DEFAULT_COLS) -> tuple[int, int]:
    """Return the top-most, left-most ``(x, y)`` where a ``w×h`` item fits without
    overlapping any existing item."""
    w = max(1, min(w, cols))
    y = 0
    while True:
        for x in range(0, cols - w + 1):
            probe = GridItem("__probe__", x, y, w, h)
            if not any(overlaps(probe, it) for it in items):
                return x, y
        y += 1


def bottom_row(items: list[GridItem]) -> int:
    """The first empty row below all items (i.e. total rows used)."""
    return max((it.y + it.h for it in items), default=0)
