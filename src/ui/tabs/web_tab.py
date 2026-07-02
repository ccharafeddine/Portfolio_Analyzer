"""Single-web-view tab base.

The previous design created one ``QWebEngineView`` per chart. Each of those is a
native OS window, so a tab with six charts spawned six native sub-windows that
visibly "popped in" and settled — the flicker the user saw on every run and tab
switch.

``WebTab`` instead renders an entire tab as ONE HTML page inside a single web
view: headings, explanations, tables, stat grids, and all charts (as Plotly
divs). One native window per tab, created once. Subclasses keep the same
``_populate`` + ``add_*`` API as the old Qt base, so tab code is unchanged; the
helpers now emit HTML instead of Qt widgets.

Explanations use CSS hover tooltips (reliable, unlike the Qt tooltip), and expand
to inline blurbs in Beginner mode.
"""

from __future__ import annotations

import html as _html
import json
import re
from typing import Callable, Optional

import math
import pandas as pd
from PySide6.QtCore import QUrl
from PySide6.QtGui import QColor
from PySide6.QtWebEngineCore import QWebEngineSettings
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QVBoxLayout, QWidget

from .. import explanations
from .. import theme
from ..widgets.plotly_widget import _ensure_plotly_js, _web_resources_dir


def _fmt_default(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        if math.isnan(value):
            return "—"
        return f"{value:,.4f}" if abs(value) < 1 else f"{value:,.2f}"
    return str(value)


class WebTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        _ensure_plotly_js()
        self._base_url = QUrl.fromLocalFile(str(_web_resources_dir()) + "/")

        self._view = QWebEngineView(self)
        self._view.page().setBackgroundColor(QColor(theme.ACTIVE.chart_bg))
        self._view.settings().setAttribute(
            QWebEngineSettings.WebAttribute.ShowScrollBars, True
        )
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._view)

        self._results = None
        self._dirty = True
        self._html: list[str] = []
        self._figs: dict[str, str] = {}
        self._chart_i = 0
        self._loaded_blank = False
        self._post_script = ""  # optional JS appended to the next rendered page
        self._render_blank()

    # ── Interface used by ResultsView ──
    def set_results(self, results) -> None:
        self._results = results
        self._dirty = True

    def mark_dirty(self) -> None:
        self._dirty = True

    def prewarm(self, n: int = 0) -> None:
        # The single view is created in __init__, so the native window and the
        # QtWebEngine subsystem are already warmed. Nothing per-chart to do.
        return

    def ensure_populated(self) -> None:
        if not self._dirty:
            return
        self._html = []
        self._figs = {}
        self._chart_i = 0
        if self._results is None:
            self._render_blank()
        else:
            self._populate(self._results)
            self._view.setHtml(self._build_page(), self._base_url)
        self._post_script = ""  # one-shot; don't repeat on later renders
        self._dirty = False

    def _populate(self, results) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    # ── Page assembly ──
    def _render_blank(self) -> None:
        t = theme.ACTIVE
        self._view.setHtml(
            f"<html><head>{self._css()}</head><body>"
            f"<div class='empty'>Run an analysis to populate this tab.</div>"
            f"</body></html>",
            self._base_url,
        )

    def _group_sections(self, blocks: list[str]) -> str:
        """Wrap each ``<h3>`` heading and the blocks that follow it (up to the next
        ``<h3>``) into a collapsible section. Blocks before the first heading are
        emitted as-is (the tab intro / interpretation)."""
        out: list[str] = []
        cur: Optional[dict] = None

        def flush():
            if cur is None:
                return
            body = "\n".join(cur["body"])
            out.append(
                "<div class='section'>"
                "<div class='sec-head' onclick='toggleSec(this)'>"
                f"<span class='chev'></span><span class='sec-title'>{cur['inner']}</span>"
                "</div>"
                f"<div class='sec-body'><div class='sec-inner'>{body}</div></div></div>"
            )

        for b in blocks:
            if b.startswith("<div class='zone'"):
                flush()          # zones are standalone dividers between sections
                cur = None
                out.append(b)
            elif b.startswith("<h3>") and b.endswith("</h3>"):
                flush()
                cur = {"inner": b[4:-5], "body": []}
            elif cur is None:
                out.append(b)
            else:
                cur["body"].append(b)
        flush()
        return "\n".join(out)

    def _build_page(self) -> str:
        figs_js = ",\n".join(f"'{cid}': {js}" for cid, js in self._figs.items())
        body = self._group_sections(self._html)
        return f"""<!DOCTYPE html><html><head>
<meta charset='utf-8'/>
<script src='plotly.min.js'></script>
{self._css()}
</head><body>
<div class='wrap'>
{body}
</div>
<script>
  var FIGS = {{ {figs_js} }};
  var CFG = {{responsive:true, displaylogo:false}};
  function draw() {{
    for (var id in FIGS) {{
      Plotly.newPlot(id, FIGS[id].data, FIGS[id].layout, CFG);
    }}
  }}
  draw();
  function toggleSec(head) {{
    var sec = head.parentElement;
    var b = sec.querySelector('.sec-body');
    var collapsing = !sec.classList.contains('collapsed');
    if (collapsing) {{
      b.style.height = b.scrollHeight + 'px';
      b.offsetHeight;  // force reflow so the next change animates
      sec.classList.add('collapsed');
      b.style.height = '0px';
      b.style.opacity = '0';
    }} else {{
      sec.classList.remove('collapsed');
      b.style.opacity = '1';
      b.style.height = b.scrollHeight + 'px';
      var done = function(e) {{
        if (e.propertyName !== 'height') return;
        b.style.height = 'auto';
        b.removeEventListener('transitionend', done);
        if (window.Plotly) {{
          b.querySelectorAll('.js-plotly-plot').forEach(function(gd) {{
            Plotly.Plots.resize(gd);
          }});
        }}
      }};
      b.addEventListener('transitionend', done);
    }}
  }}
  {self._post_script}
</script>
</body></html>"""

    def _css(self) -> str:
        t = theme.ACTIVE
        pad = t.content_spacing
        return f"""<style>
  html,body{{margin:0;background:{t.chart_bg};color:{t.text};
    font-family:{t.font};font-size:{t.base_pt}px}}
  .wrap{{display:flex;flex-direction:column;gap:{pad}px;
    padding:{pad + 4}px {pad + 4}px {pad + 12}px}}
  .empty{{color:{t.text_muted};text-align:center;padding:60px 20px}}
  h3{{font-size:{t.heading_pt}px;font-weight:700;margin:6px 0 0}}
  h4{{font-size:{t.base_pt + 1}px;font-weight:600;margin:0 0 6px;color:{t.text}}}
  .section{{border-top:1px solid {t.border};padding-top:{max(6, pad - 4)}px}}
  .sec-head{{display:flex;align-items:center;gap:10px;cursor:pointer;user-select:none}}
  .sec-title{{font-size:{t.heading_pt}px;font-weight:700}}
  .sec-head:hover .sec-title{{color:{t.accent}}}
  .sec-body{{overflow:hidden;transition:height .24s ease,opacity .2s ease}}
  .sec-inner{{display:flex;flex-direction:column;gap:{pad}px;padding-top:{pad}px}}
  .chev{{flex:none;width:0;height:0;border-left:6px solid {t.text_muted};
    border-top:5px solid transparent;border-bottom:5px solid transparent;
    transition:transform .15s;transform:rotate(90deg)}}
  .section.collapsed .chev{{transform:rotate(0deg)}}
  .sec-head:hover .chev{{border-left-color:{t.accent}}}
  .interp{{background:{t.card};border-left:3px solid {t.accent};
    padding:14px 18px;border-radius:0 8px 8px 0;color:{t.text_slate};line-height:1.5}}
  .blurb{{background:{t.card};color:{t.text_slate};border-radius:{max(4, t.radius - 4)}px;
    padding:8px 12px;font-size:{t.base_pt - 1}px}}
  .chart{{width:100%}}
  .chart-row{{display:flex;gap:16px}}
  .chart-row>.chart{{flex:1;min-width:0}}
  .chart-col{{display:flex;flex-direction:column;flex:1;min-width:0}}
  .chart-grid{{display:grid;gap:12px}}
  table.tbl{{width:100%;border-collapse:collapse;background:{t.card};
    border:1px solid {t.border};border-radius:{max(4, t.radius - 4)}px;overflow:hidden}}
  .tbl th{{background:{t.panel};color:{t.text_muted};text-transform:uppercase;
    font-size:{t.label_pt}px;font-weight:600;padding:8px 10px;text-align:left;
    border-bottom:1px solid {t.border}}}
  .tbl td{{padding:7px 10px;border-bottom:1px solid {t.border};font-variant-numeric:tabular-nums}}
  .tbl tr:nth-child(even) td{{background:{t.card_alt}}}
  table.tbl-fixed{{table-layout:fixed}}
  .tbl-fixed th,.tbl-fixed td{{white-space:normal;word-break:break-word}}
  .zone{{display:flex;align-items:baseline;gap:12px;margin-top:14px;
    padding:10px 0 2px;border-top:2px solid {t.accent}}}
  .zone-title{{font-size:{t.heading_pt + 5}px;font-weight:800;color:{t.text};
    letter-spacing:.01em}}
  .zone-sub{{color:{t.text_muted};font-size:{t.base_pt}px}}
  .chips{{display:flex;flex-wrap:wrap;gap:8px;margin:10px 0 2px}}
  .chip{{padding:4px 14px;border:1px solid {t.border_light};border-radius:999px;
    color:{t.text_muted};text-decoration:none;font-size:{t.base_pt - 1}px;
    font-weight:600;transition:all .12s}}
  .chip:hover{{border-color:{t.accent};color:{t.text}}}
  .chip.active{{background:{t.accent};border-color:{t.accent};color:#fff}}
  ::-webkit-scrollbar{{width:12px;height:12px}}
  ::-webkit-scrollbar-track{{background:transparent}}
  ::-webkit-scrollbar-thumb{{background:{t.border_light};border-radius:7px;
    border:3px solid {t.chart_bg}}}
  ::-webkit-scrollbar-thumb:hover{{background:{t.text_muted}}}
  ::-webkit-scrollbar-thumb:active{{background:{t.accent}}}
  .statgrid{{display:grid;gap:10px}}
  .card{{background:{t.card};border:1px solid {t.border_light};
    border-radius:{max(6, t.radius - 2)}px;padding:12px 14px}}
  .card .k{{color:{t.text_muted};font-size:{t.label_pt - 1}px;font-weight:700;
    letter-spacing:.05em;text-transform:uppercase}}
  .card .v{{color:{t.text};font-size:{t.statval_pt}px;font-weight:700;
    font-family:{t.mono};margin-top:2px}}
  .info{{color:{t.text_muted};cursor:help;position:relative;display:inline-block;
    vertical-align:middle;margin-left:8px}}
  .info:hover{{color:{t.accent}}}
  .qmark{{display:inline-block;width:1.35em;height:1.35em;line-height:1.3em;
    text-align:center;border:1px solid currentColor;border-radius:50%;
    font-weight:700;font-size:.6em}}
  .sec-title .info,h3 .info,h4 .info{{font-weight:400}}
  .info .tip{{visibility:hidden;opacity:0;transition:opacity .12s;position:absolute;
    left:0;top:130%;z-index:20;width:320px;background:{t.panel};color:{t.text};
    border:1px solid {t.accent};border-radius:6px;padding:10px 12px;
    font-size:{t.base_pt}px;font-weight:400;line-height:1.5;
    box-shadow:0 8px 24px rgba(0,0,0,.4)}}
  .info:hover .tip{{visibility:visible;opacity:1}}
</style>"""

    # ── Building blocks (same API as the old Qt base) ──
    def _qmark(self, key: Optional[str]) -> str:
        """A circled '?' badge with a hover explanation, to sit next to a title."""
        e = explanations.get(key) if key else None
        if not e:
            return ""
        tip = (
            f"<b>{_html.escape(e['title'])}</b><br><br>{_html.escape(e['what'])}<br><br>"
            f"<b>How to read it:</b> {_html.escape(e['how'])}<br><br>"
            f"<b>Why it matters:</b> {_html.escape(e['why'])}"
        )
        # stopPropagation so clicking the badge doesn't toggle its collapsible section.
        return (
            f"<span class='info' onclick='event.stopPropagation()'>"
            f"<span class='qmark'>?</span><span class='tip'>{tip}</span></span>"
        )

    def _heading_html(self, text: str, key: Optional[str] = None, tag: str = "h3",
                      allow_blurb: bool = True) -> str:
        """A heading tag with an inline circled '?'. Kept as a single, clean
        ``<tag>…</tag>`` element (no trailing blurb) so ``h3`` blocks group cleanly
        into collapsible sections. In Beginner mode the badge is dropped from
        section headings (the blurb is emitted separately by ``add_heading``)."""
        e = explanations.get(key) if key else None
        suppress = allow_blurb and e is not None and explanations.is_beginner_mode()
        badge = "" if suppress else self._qmark(key)
        return f"<{tag}>{_html.escape(text)}{badge}</{tag}>"

    def add_heading(self, text: str, explain: Optional[str] = None) -> None:
        """A section/chart heading, with an optional circled '?' inline. In Beginner
        mode a plain-English blurb is emitted just below it (inside the section)."""
        self._html.append(self._heading_html(text, explain))
        e = explanations.get(explain) if explain else None
        if e is not None and explanations.is_beginner_mode():
            self._html.append(
                f"<div class='blurb'><b>{_html.escape(e['title'])}.</b> "
                f"{_html.escape(explanations.inline_text(explain))}</div>"
            )

    def add_zone(self, title: str, subtitle: Optional[str] = None,
                 anchor: Optional[str] = None) -> None:
        """A prominent banner separating major zones of a tab (e.g. the
        portfolio-wide comparison vs. a single-company drill-down)."""
        aid = f" id='{anchor}'" if anchor else ""
        sub = (f"<span class='zone-sub'>{_html.escape(subtitle)}</span>"
               if subtitle else "")
        self._html.append(
            f"<div class='zone'{aid}>"
            f"<span class='zone-title'>{_html.escape(title)}</span>{sub}</div>"
        )

    def add_interpretation(self, text: Optional[str]) -> None:
        if not text:
            return
        self._html.append(f"<div class='interp'>{_html.escape(text)}</div>")

    def add_html(self, raw: str) -> None:
        """Append a raw HTML block (caller is responsible for escaping)."""
        if raw:
            self._html.append(raw)

    def add_explainer(self, key: Optional[str]) -> None:
        """Backwards-compatible: renders the Beginner-mode blurb (the circled '?'
        now lives next to headings/titles, so nothing is shown otherwise)."""
        if key and explanations.get(key) and explanations.is_beginner_mode():
            e = explanations.get(key)
            self._html.append(
                f"<div class='blurb'><b>{_html.escape(e['title'])}.</b> "
                f"{_html.escape(explanations.inline_text(key))}</div>"
            )

    def _fig_title(self, fig) -> Optional[str]:
        try:
            return fig.layout.title.text or None
        except Exception:
            return None

    def _blank_fig_title(self, fig) -> None:
        # Title moves into the HTML heading, so remove it from the chart and
        # reclaim the top margin it used.
        try:
            fig.update_layout(title_text="", margin_t=28)
        except Exception:
            pass

    def _chart_div(self, fig, height: int, style_extra: str = "") -> str:
        cid = f"c{self._chart_i}"
        self._chart_i += 1
        fig.update_layout(autosize=True, height=None)
        self._figs[cid] = fig.to_json()
        return f"<div id='{cid}' class='chart' style='height:{height}px;{style_extra}'></div>"

    def _scaled(self, height: int) -> int:
        return int(height * theme.ACTIVE.chart_scale)

    def add_chart(self, fig, height: int = 420, explain: Optional[str] = None) -> None:
        if fig is None:
            return
        # Put the chart's title (with the '?' badge) in an HTML heading above it,
        # which also becomes the collapsible section header for the chart.
        if explain and explanations.get(explain):
            title = self._fig_title(fig) or explanations.get(explain)["title"]
            self.add_heading(title, explain)
            self._blank_fig_title(fig)
        self._html.append(self._chart_div(fig, self._scaled(height)))

    def add_subchart(
        self, fig, height: int = 340, explain: Optional[str] = None,
        title: Optional[str] = None,
    ) -> None:
        """A chart nested inside the *current* collapsible section. Uses an ``h4``
        sub-heading (with an optional '?') so it does NOT start a new section — use
        this to group several charts under one collapsible ``add_heading``."""
        if fig is None:
            return
        label = title or (self._fig_title(fig) if fig is not None else None)
        if explain and explanations.get(explain):
            label = label or explanations.get(explain)["title"]
            self._html.append(self._heading_html(label, explain, tag="h4",
                                                 allow_blurb=False))
            self._blank_fig_title(fig)
        elif label:
            self._html.append(self._heading_html(label, None, tag="h4"))
            self._blank_fig_title(fig)
        self._html.append(self._chart_div(fig, self._scaled(height)))

    def add_chart_row(
        self, figs: list, height: int = 380, explains: Optional[list] = None
    ) -> None:
        """Side-by-side charts. ``explains`` is a per-chart list of explanation
        keys — each chart gets its own title + '?' above it."""
        figs = [f for f in figs if f is not None]
        if not figs:
            return
        keys = list(explains or [])
        keys += [None] * (len(figs) - len(keys))
        if len(figs) == 1:
            self.add_chart(figs[0], height=height, explain=keys[0])
            return
        cols = []
        for fig, key in zip(figs, keys):
            head = ""
            if key and explanations.get(key):
                title = self._fig_title(fig) or explanations.get(key)["title"]
                head = self._heading_html(title, key, tag="h4", allow_blurb=False)
                self._blank_fig_title(fig)
            cols.append(
                f"<div class='chart-col'>{head}{self._chart_div(fig, self._scaled(height))}</div>"
            )
        self._html.append(f"<div class='chart-row'>{''.join(cols)}</div>")

    def add_chart_grid(
        self, figs: list, columns: int = 3, height: int = 300, explain: Optional[str] = None
    ) -> None:
        figs = [f for f in figs if f is not None]
        if not figs:
            return
        if explain and explanations.get(explain):
            self.add_heading(explanations.get(explain)["title"], explain=explain)
        divs = "".join(self._chart_div(f, self._scaled(height)) for f in figs)
        self._html.append(
            f"<div class='chart-grid' style='grid-template-columns:repeat({columns},1fr)'>"
            f"{divs}</div>"
        )

    def add_table(
        self,
        df: pd.DataFrame,
        formatters: Optional[dict[str, Callable]] = None,
        show_index: bool = False,
        slots: Optional[int] = None,
    ) -> None:
        if df is None or len(df) == 0:
            return
        formatters = formatters or {}
        frame = df if show_index else df.reset_index(drop=True)
        disp = pd.DataFrame(index=frame.index)
        for col in frame.columns:
            fmt = formatters.get(col)
            disp[col] = [fmt(v) if fmt else _fmt_default(v) for v in frame[col]]

        colgroup, classes = "", "tbl"
        if slots and slots > 1:
            # Uniform column pitch = 100% / slots, applied to every table in a group
            # so their columns line up while each still fills the full width. Tables
            # with fewer columns are padded with blank trailing cells.
            pad = 0
            while disp.shape[1] < slots:
                pad += 1
                disp[" " * pad] = ""  # unique, blank-rendering column name
            ncols = disp.shape[1] + (1 if show_index else 0)
            pitch = 100.0 / slots
            colgroup = ("<colgroup>"
                        + "".join(f"<col style='width:{pitch:.4f}%'>" for _ in range(ncols))
                        + "</colgroup>")
            classes = ["tbl", "tbl-fixed"]

        html_table = disp.to_html(
            index=show_index, escape=True, border=0, classes=classes, justify="left"
        )
        if colgroup:
            html_table = re.sub(r"(<table\b[^>]*>)", lambda m: m.group(1) + colgroup,
                                html_table, count=1)
        self._html.append(html_table)

    def add_stat_grid(self, items: list[tuple[str, str]], columns: int = 4) -> None:
        if not items:
            return
        cards = "".join(
            f"<div class='card'><div class='k'>{_html.escape(k)}</div>"
            f"<div class='v'>{_html.escape(str(v))}</div></div>"
            for k, v in items
        )
        self._html.append(
            f"<div class='statgrid' style='grid-template-columns:repeat({columns},1fr)'>"
            f"{cards}</div>"
        )
