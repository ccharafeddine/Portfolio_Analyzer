"""Dense, sortable HTML tables for the web-rendered tabs.

``rich_table_html`` renders a DataFrame with typed columns: numbers are
right-aligned with tabular numerals, signed columns are coloured green/red (P&L
only), a column can carry an inline weight bar, an optional totals row is pinned
to the bottom (it does not sort), headers sort on click, and a Copy button puts
the table on the clipboard as tab-separated text that pastes straight into Excel.
"""

from __future__ import annotations

import html as _html
import math
import re
from dataclasses import dataclass
from typing import Optional

import pandas as pd

_NUM_RE = re.compile(r"^[-+−]?\$?[-+−]?[\d,]*\.?\d+(%|x|[KMB])?$")


def numeric_class(values) -> bool:
    """True when every non-blank formatted value in a column looks numeric."""
    seen = False
    for v in values:
        sv = str(v).strip()
        if sv in ("", "—", "-", "nan", "None"):
            continue
        if not _NUM_RE.match(sv.replace(" ", "")):
            return False
        seen = True
    return seen


def _isnan(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


@dataclass
class Col:
    key: str
    label: str
    kind: str = "text"   # text | pct | spct (signed pct) | num | money | ratio | int | sym
    decimals: int = 2
    bar: bool = False     # draw an inline bar proportional to the value (weights)
    sub: Optional[str] = None  # for kind="sym": a secondary muted column (e.g. name)


def fmt_value(v, kind: str, decimals: int = 2) -> str:
    if _isnan(v):
        return "—"
    if kind == "pct":
        return f"{v * 100:.{decimals}f}%"
    if kind == "spct":
        return f"{v * 100:+.{decimals}f}%".replace("-", "−")
    if kind == "money":
        return f"−${-v:,.0f}" if v < 0 else f"${v:,.0f}"
    if kind == "num":
        return f"{v:,.{decimals}f}"
    if kind == "int":
        return f"{v:,.0f}"
    if kind == "ratio":
        return f"{v:.{decimals}f}".replace("-", "−")
    return str(v)


def _cell(v, col: Col, row: dict, bar_max: float, is_total: bool = False) -> str:
    text = fmt_value(v, col.kind, col.decimals)
    sort = "" if _isnan(v) or not isinstance(v, (int, float)) else f" data-v='{float(v):.10g}'"
    if col.kind == "label":
        return f"<td data-v='{_html.escape(str(v))}'>{_html.escape(text)}</td>"
    if col.kind in ("text", "sym"):
        sub = row.get(col.sub) if col.sub else None
        inner = f"<span class='sym'>{_html.escape(text)}</span>"
        if sub:
            inner += f" <span class='nm'>{_html.escape(str(sub))}</span>"
        return f"<td data-v='{_html.escape(str(v))}'>{inner}</td>"
    cls = ["n"]
    if col.kind == "spct" and not _isnan(v):
        cls.append("pos" if v > 0 else "neg" if v < 0 else "")
    out = f"<td class='{' '.join(c for c in cls if c)}'{sort}>{_html.escape(text)}</td>"
    if col.bar:
        width = 0 if (is_total or _isnan(v) or bar_max <= 0) else max(0.0, float(v)) / bar_max * 90
        out += f"<td class='barcell'><span class='wbar' style='width:{width:.1f}px'></span></td>"
    return out


def rich_table_html(
    df: pd.DataFrame,
    columns: list,
    total: Optional[dict] = None,
    sortable: bool = True,
    copy: bool = True,
    table_id: str = "tbl",
) -> str:
    cols = [c if isinstance(c, Col) else Col(*c) for c in columns]
    heads = []
    for c in cols:
        cls = "n" if c.kind not in ("text", "sym", "label") else ""
        heads.append(f"<th class='{cls}'>{_html.escape(c.label)}</th>")
        if c.bar:
            heads.append("<th class='nosort' style='width:100px'></th>")
    bar_max = {c.key: float(pd.to_numeric(df[c.key], errors="coerce").max() or 0)
               for c in cols if c.bar}
    body = []
    for _, r in df.iterrows():
        row = r.to_dict()
        body.append("<tr>" + "".join(_cell(row.get(c.key), c, row, bar_max.get(c.key, 0))
                                     for c in cols) + "</tr>")
    foot = ""
    if total:
        foot = "<tfoot><tr class='tot'>" + "".join(
            _cell(total.get(c.key), c, total, bar_max.get(c.key, 0), True) for c in cols
        ) + "</tr></tfoot>"
    tools = ""
    if copy:
        tools = (f"<div class='tbl-tools'><button class='btn' "
                 f"onclick=\"copyTable('{table_id}', this)\">Copy</button></div>")
    cls = "tbl sortable" if sortable else "tbl"
    return (f"{tools}<table id='{table_id}' class='{cls}'><thead><tr>{''.join(heads)}"
            f"</tr></thead><tbody>{''.join(body)}</tbody>{foot}</table>")


def table_to_tsv(df: pd.DataFrame, columns: list, total: Optional[dict] = None) -> str:
    """The text the Copy button produces (Python twin, used by tests/exports)."""
    cols = [c if isinstance(c, Col) else Col(*c) for c in columns]
    lines = ["\t".join(c.label for c in cols)]
    rows = [r.to_dict() for _, r in df.iterrows()] + ([total] if total else [])
    for row in rows:
        lines.append("\t".join(fmt_value(row.get(c.key), c.kind, c.decimals) for c in cols))
    return "\n".join(lines)


# Shared JS: click-to-sort (numeric via data-v, totals stay in <tfoot>) and Copy.
TABLE_JS = r"""
document.querySelectorAll('table.sortable').forEach(function(tbl){
  var ths = tbl.tHead.rows[0].cells;
  for (var i = 0; i < ths.length; i++) (function(th, idx){
    if (th.classList.contains('nosort')) return;
    th.addEventListener('click', function(){
      var asc = !th.classList.contains('asc');
      for (var j = 0; j < ths.length; j++) ths[j].classList.remove('asc','desc');
      th.classList.add(asc ? 'asc' : 'desc');
      var rows = Array.prototype.slice.call(tbl.tBodies[0].rows);
      rows.sort(function(a, b){
        var x = a.cells[idx].getAttribute('data-v'), y = b.cells[idx].getAttribute('data-v');
        var nx = parseFloat(x), ny = parseFloat(y);
        var r = (!isNaN(nx) && !isNaN(ny)) ? nx - ny : String(x).localeCompare(String(y));
        return asc ? r : -r;
      });
      rows.forEach(function(r){ tbl.tBodies[0].appendChild(r); });
    });
  })(ths[i], i);
});
function copyTable(id, btn){
  var tbl = document.getElementById(id), out = [];
  for (var i = 0; i < tbl.rows.length; i++) {
    var cells = [], row = tbl.rows[i].cells;
    for (var j = 0; j < row.length; j++) {
      if (row[j].classList.contains('barcell') || row[j].classList.contains('nosort')) continue;
      cells.push(row[j].innerText.trim());
    }
    out.push(cells.join('\t'));
  }
  var ta = document.createElement('textarea');
  ta.value = out.join('\n'); document.body.appendChild(ta); ta.select();
  try { document.execCommand('copy'); btn.textContent = 'Copied'; } catch (e) {}
  document.body.removeChild(ta);
  setTimeout(function(){ btn.textContent = 'Copy'; }, 1200);
}
"""


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha:.3f})"


def heatmap_table_html(cal: pd.DataFrame, green: str, red: str, scale: float = 0.12,
                       table_id: str = "cal") -> str:
    """Year x month return calendar. Cell tint scales with |return| (capped at
    ``scale``); the trailing summary columns (Year total / Benchmark) are untinted."""
    summary = [c for c in cal.columns if c in ("Year total", "Benchmark")]
    months = [c for c in cal.columns if c not in summary]
    head = "<th>Year</th>" + "".join(f"<th class='n'>{_html.escape(str(c))}</th>"
                                     for c in cal.columns)
    rows = []
    for year, r in cal.iterrows():
        cells = [f"<td>{year}</td>"]
        for c in months:
            v = r[c]
            if _isnan(v):
                cells.append("<td class='n' style='opacity:.35'>—</td>")
                continue
            a = min(abs(v) / scale, 1.0) * 0.45
            bg = _rgba(green if v > 0 else red, a)
            cells.append(f"<td class='n' style='background:{bg}'>"
                         f"{fmt_value(v, 'spct', 1)}</td>")
        for c in summary:
            v = r[c]
            style = "font-weight:600" if c == "Year total" else "opacity:.7"
            cells.append(f"<td class='n' style='{style}'>{fmt_value(v, 'spct', 1)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return (f"<div class='tbl-tools'><button class='btn' onclick=\"copyTable('{table_id}', this)\">"
            f"Copy</button></div><table id='{table_id}' class='tbl hm'><thead><tr>{head}</tr>"
            f"</thead><tbody>{''.join(rows)}</tbody></table>")
