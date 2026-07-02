"""HTML + JS for the interactive what-if scenario builder (rendered inside the Risk tab).

All interactivity is client-side: the pipeline precomputes each driver's beta, and this
page does ``sum(beta * shock)`` live in the browser as the user edits inputs — so shocks
update the estimated impact and chart with no round-trip to Python.
"""

from __future__ import annotations

import html as _html
import json

from .. import theme


def scenario_html(model: dict) -> str:
    t = theme.ACTIVE
    drivers = model.get("drivers", [])
    value = float(model.get("value", 0.0))
    presets = model.get("presets", {})

    macro = [d for d in drivers if d.get("group") == "Macro"]
    holds = [d for d in drivers if d.get("group") == "Holding"]

    def row(d: dict) -> str:
        name = _html.escape(str(d["name"]))
        return (
            "<div class='scen-row'>"
            f"<span class='scen-lbl'>{name}</span>"
            "<input class='scen-in' type='number' step='1' value='0' "
            f"data-name=\"{name}\" data-beta='{d['beta']}' oninput='scenEdit()'>"
            "<span class='scen-unit'>%</span></div>"
        )

    def group(title: str, rows: list[dict]) -> str:
        if not rows:
            return ""
        return (f"<div class='scen-col'><div class='scen-grp'>{title}</div>"
                + "".join(row(d) for d in rows) + "</div>")

    presets_btns = "".join(
        f"<button class='scen-btn' onclick='scenPreset(this)' "
        f"data-preset=\"{_html.escape(n)}\">{_html.escape(n)}</button>"
        for n in presets
    )

    css = f"""
<style>
.scen{{margin-top:6px}}
.scen-cols{{display:flex;gap:26px;flex-wrap:wrap;align-items:flex-start}}
.scen-col{{min-width:230px}}
.scen-grp{{font-size:{t.base_pt - 1}px;font-weight:700;color:{t.text_muted};
  text-transform:uppercase;letter-spacing:.04em;margin:2px 0 8px}}
.scen-row{{display:flex;align-items:center;gap:8px;margin:5px 0}}
.scen-lbl{{flex:1;color:{t.text};font-size:{t.base_pt}px}}
.scen-in{{width:64px;background:{t.card_alt};color:{t.text};
  border:1px solid {t.border_light};border-radius:6px;padding:3px 6px;
  font-size:{t.base_pt}px;text-align:right}}
.scen-in:focus{{outline:none;border-color:{t.accent}}}
.scen-unit{{color:{t.text_muted};width:12px}}
.scen-actions{{margin:12px 0 6px;display:flex;gap:8px;flex-wrap:wrap;align-items:center}}
.scen-btn{{background:{t.card_alt};color:{t.text};border:1px solid {t.border_light};
  border-radius:14px;padding:4px 12px;font-size:{t.base_pt - 1}px;cursor:pointer}}
.scen-btn:hover{{border-color:{t.accent};color:{t.accent}}}
.scen-btn.active{{background:{t.accent};border-color:{t.accent};color:{t.accent_text}}}
.scen-btn.active:hover{{color:{t.accent_text}}}
.scen-reset{{background:transparent;color:{t.text_muted}}}
.scen-result{{margin:14px 0 6px;font-size:{t.heading_pt}px;color:{t.text}}}
.scen-result b{{font-size:{t.heading_pt + 4}px}}
</style>"""

    data = json.dumps({"value": value, "presets": presets})
    grid = t.chart_grid
    js = """
<script>
(function(){
  var SCEN = __DATA__;
  window.__scenData = SCEN;
  function fmtMoney(v){
    var sign = v < 0 ? "\\u2212" : "+";
    return sign + "$" + Math.abs(Math.round(v)).toLocaleString();
  }
  function layout(){
    return {margin:{t:10,r:12,b:40,l:54}, height:320,
      paper_bgcolor:"rgba(0,0,0,0)", plot_bgcolor:"rgba(0,0,0,0)",
      font:{color:"__TEXT__"}, showlegend:false,
      yaxis:{title:"Contribution to return (%)", gridcolor:"__GRID__", zeroline:false},
      xaxis:{gridcolor:"rgba(0,0,0,0)"}};
  }
  window.scenRecalc = function(){
    var total = 0, labels = [], vals = [];
    var ins = document.querySelectorAll(".scen-in");
    for (var i = 0; i < ins.length; i++){
      var b = parseFloat(ins[i].getAttribute("data-beta")) || 0;
      var s = (parseFloat(ins[i].value) || 0) / 100;
      var c = b * s; total += c;
      if (s !== 0){ labels.push(ins[i].getAttribute("data-name")); vals.push(+(c*100).toFixed(3)); }
    }
    var rl = document.getElementById("scenRet"); if (rl) rl.textContent = (total*100).toFixed(2) + "%";
    var pl = document.getElementById("scenPnl"); if (pl) pl.textContent = fmtMoney(SCEN.value * total);
    var colors = vals.map(function(v){ return v >= 0 ? "#3fb950" : "#f85149"; });
    if (window.Plotly) Plotly.react("scenChart",
      [{type:"bar", x:labels, y:vals, marker:{color:colors},
        hovertemplate:"%{x}: %{y:.2f}%<extra></extra>"}],
      layout(), {displaylogo:false, responsive:true});
  };
  function clearActive(){
    var b = document.querySelectorAll(".scen-btn");
    for (var i = 0; i < b.length; i++) b[i].classList.remove("active");
  }
  window.scenEdit = function(){ clearActive(); window.scenRecalc(); };
  window.scenPreset = function(btn){
    var p = SCEN.presets[btn.getAttribute("data-preset")] || {};
    var ins = document.querySelectorAll(".scen-in");
    for (var i = 0; i < ins.length; i++){
      var n = ins[i].getAttribute("data-name");
      ins[i].value = (p[n] != null) ? p[n] : 0;
    }
    clearActive(); btn.classList.add("active");
    window.scenRecalc();
  };
  window.scenReset = function(){
    clearActive();
    var ins = document.querySelectorAll(".scen-in");
    for (var i = 0; i < ins.length; i++) ins[i].value = 0;
    window.scenRecalc();
  };
  window.scenRecalc();
})();
</script>"""
    js = js.replace("__DATA__", data).replace("__TEXT__", t.text).replace("__GRID__", grid)

    body = (
        css
        + "<div class='scen'><div class='scen-cols'>"
        + group("Macro factors", macro)
        + group("Your holdings", holds)
        + "</div>"
        + "<div class='scen-actions'>"
        + "<span class='scen-lbl' style='flex:0'>Presets:</span>"
        + presets_btns
        + "<button class='scen-btn scen-reset' onclick='scenReset()'>Reset</button>"
        + "</div>"
        + "<div class='scen-result'>Estimated impact: "
        + "<b id='scenRet'>0.00%</b> (<span id='scenPnl'>+$0</span>)</div>"
        + "<div id='scenChart' class='chart' style='height:320px'></div>"
        + "</div>"
        + js
    )
    return body
