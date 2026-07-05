"""Grid-placement engine (Qt-free): clamp, collision resolve, free-slot search."""

from src.ui.widgets.grid_layout import (
    GridItem,
    bottom_row,
    clamp,
    find_free,
    overlaps,
    resolve,
)


def _items(*specs):
    return [GridItem(*s) for s in specs]


def test_overlaps():
    a = GridItem("a", 0, 0, 4, 4)
    assert overlaps(a, GridItem("b", 2, 2, 4, 4)) is True
    assert overlaps(a, GridItem("c", 4, 0, 4, 4)) is False   # edge-adjacent, no overlap
    assert overlaps(a, GridItem("d", 0, 4, 4, 4)) is False


def test_clamp_bounds_and_min():
    it = clamp(GridItem("x", -3, -1, 99, 0), cols=12, min_w=2, min_h=2)
    assert it.w == 12 and it.h == 2          # w capped to cols, h floored to min
    assert it.x == 0 and it.y == 0           # origin clamped non-negative
    it2 = clamp(GridItem("y", 11, 0, 3, 1), cols=12)
    assert it2.x == 9 and it2.w == 3         # x pulled in so x+w <= cols


def test_resolve_pushes_overlap_down():
    items = _items(("a", 0, 0, 6, 4), ("b", 0, 0, 6, 4))  # b was under a, now overlapping
    resolve(items, "a")                                   # a is the anchor
    a, b = items
    assert a.as_tuple() == (0, 0, 6, 4)                   # anchor unmoved
    assert b.y == 4 and not overlaps(a, b)                # b pushed just below a


def test_resolve_cascades():
    # a moved onto b, b must move onto c, c must move — all clear afterwards.
    items = _items(("a", 0, 0, 12, 3), ("b", 0, 0, 6, 3), ("c", 0, 3, 6, 3))
    resolve(items, "a")
    a, b, c = items
    assert a.y == 0
    # No pair overlaps after resolution.
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            assert not overlaps(items[i], items[j])


def test_resolve_leaves_non_overlapping_alone():
    items = _items(("a", 0, 0, 6, 4), ("b", 6, 0, 6, 4))  # side by side, no overlap
    resolve(items, "a")
    assert items[1].as_tuple() == (6, 0, 6, 4)            # untouched


def test_find_free_and_bottom_row():
    items = _items(("a", 0, 0, 6, 4), ("b", 6, 0, 6, 4))
    assert find_free(items, 6, 2) == (0, 4)               # first gap below the top row
    assert bottom_row(items) == 4
    assert find_free([], 4, 2) == (0, 0)
