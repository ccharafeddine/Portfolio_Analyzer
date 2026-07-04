"""Value formatters shared across the desktop UI.

Ported verbatim from the Streamlit app's ``_fmt_pct`` / ``_fmt_dollar`` /
``_delta_str`` so the native metric cards and tables format numbers identically.
"""

from __future__ import annotations

import math
from typing import Optional


def _is_missing(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def fmt_pct(v, decimals: int = 2) -> str:
    if _is_missing(v):
        return "—"  # em dash
    return f"{v * 100:.{decimals}f}%"


def fmt_dollar(v) -> str:
    if _is_missing(v):
        return "—"
    if abs(v) >= 1e6:
        return f"${v / 1e6:,.2f}M"
    return f"${v:,.0f}"


def fmt_ratio(v, decimals: int = 2) -> str:
    if _is_missing(v):
        return "—"
    return f"{v:.{decimals}f}"


def delta_str(v) -> Optional[str]:
    """Signed percentage delta, or None when not meaningful."""
    if _is_missing(v):
        return None
    return f"{v * 100:+.2f}%"


def js_embed(s: str) -> str:
    """Escape a serialized JSON/JS string for safe inclusion *inside* an inline
    ``<script>`` element.

    JSON does not escape ``<``/``>``/``&``, so an untrusted value containing
    ``</script>`` (e.g. a crafted ticker in a chart label) would close the script
    element early and let following markup execute. Replacing those three chars
    with their ``\\uXXXX`` escapes is transparent to JS/JSON string parsing but
    neutralizes the breakout. Use wherever ``fig.to_json()`` / ``json.dumps(...)``
    is interpolated into an inline ``<script>``.
    """
    return (
        s.replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )
