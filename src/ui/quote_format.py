"""Pure formatting helpers for quotes / Live Market Watch.

Qt-free on purpose so they can be unit-tested in CI (the headless runner has no
GUI libs, so importing anything under ``PySide6.QtWidgets`` fails). The Live
Market Watch view imports these; keep this module free of any PySide6 import.
"""

from __future__ import annotations


def fmt_price(v) -> str:
    return f"{v:,.2f}" if isinstance(v, (int, float)) and v == v else "—"


def fmt_signed(v, pct: bool = False) -> str:
    if not isinstance(v, (int, float)) or v != v:
        return "—"
    body = f"{abs(v) * 100:.2f}%" if pct else f"{abs(v):,.2f}"
    return f"{'+' if v >= 0 else '-'}{body}"


def fmt_pct(v, d: int = 2) -> str:
    return f"{v * 100:.{d}f}%" if isinstance(v, (int, float)) and v == v else "—"


def fmt_volume(v) -> str:
    if not isinstance(v, (int, float)) or v != v:
        return "—"
    for unit, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(v) >= div:
            return f"{v / div:.2f}{unit}"
    return f"{v:.0f}"
