/* Portfolio Analyzer web demo front end. No framework, no secrets. The analytics run
   in a Web Worker (engine.worker.js) under Pyodide; prices come through the public
   price proxy. All result strings are inserted with textContent. */
(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const C = { bg: "#12161A", grid: "rgba(173,186,199,0.07)", text: "#E6EDF3", muted: "#768390",
    line: "#22272E", port: "#4C8DFF", bench: "#8B949E", green: "#3FB950", red: "#F47067", amber: "#D4A72C" };
  const MAX_T = 20, MAX_Y = 15;
  const PRESETS = {
    tech_hedged: { label: "Tech + hedges", holdings: { AAPL: 20, MSFT: 20, NVDA: 20, GLD: 20, TLT: 20 } },
    sixty_forty: { label: "60/40", holdings: { VTI: 60, BND: 40 } },
    all_weather: { label: "All Weather", holdings: { VTI: 30, TLT: 40, IEF: 15, GLD: 7.5, DBC: 7.5 } },
    dividend: { label: "Dividend income", holdings: { SCHD: 30, VIG: 25, JNJ: 15, PG: 15, KO: 15 } },
    global: { label: "Global equity", holdings: { VTI: 55, VXUS: 35, VNQ: 10 } },
  };
  let presets = PRESETS, last = null, activeTab = "overview";

  // ── engine (Pyodide in a Web Worker) ─────────────────────────
  const engine = { worker: null, ready: false, pending: new Map(), seq: 0, readyWaiters: [], loadMs: null, error: null };
  window.PA_TIMINGS = { navStart: performance.timeOrigin };
  function setEngine(pct, msg, done) {
    $("eprog-bar").style.width = pct + "%"; $("emsg").textContent = msg;
    $("engine").classList.toggle("ready", !!done);
  }
  function startEngine() {
    const E = window.PA_ENGINE;
    if (!E || !E.worker) throw new Error((E && E.error) || "Web Workers are unavailable");
    engine.worker = E.worker;
    const handle = (ev) => {
      const m = ev.data;
      if (m.type === "progress") { if (!engine.ready) setEngine(m.pct, m.msg); else if (m.forRun) $("loadmsg").textContent = m.msg; }
      else if (m.type === "ready") {
        engine.ready = true; engine.loadMs = m.ms;
        Object.assign(window.PA_TIMINGS, { engineReady: performance.now(), engineInside: m.timings, engineMs: m.ms, prefetch: m.stats });
        setEngine(100, `Engine ready (${(m.ms / 1000).toFixed(1)}s) · runs in your browser`, true);
        engine.readyWaiters.splice(0).forEach((f) => f());
      } else if (m.type === "result" || m.type === "error") {
        if (m.id !== undefined && engine.pending.has(m.id)) {
          const { resolve, reject } = engine.pending.get(m.id); engine.pending.delete(m.id);
          m.type === "result" ? resolve(m) : reject(new Error(m.msg));
        } else if (m.type === "error") {
          engine.error = m.msg; setEngine(0, m.msg); $("err").textContent = m.msg;
        }
      }
    };
    E.onmessage = handle;
    E.queue.splice(0).forEach(handle);
  }
  function whenReady() { return engine.ready ? Promise.resolve() : new Promise((r) => engine.readyWaiters.push(r)); }
  function analyzeInEngine(payload) {
    const id = ++engine.seq;
    return new Promise((resolve, reject) => { engine.pending.set(id, { resolve, reject }); engine.worker.postMessage({ type: "analyze", id, payload: JSON.stringify(payload) }); });
  }

  // ── formatting ───────────────────────────────────────────────
  const ok = (v) => v !== null && v !== undefined && !Number.isNaN(v);
  const pct = (v, d = 1) => ok(v) ? (v * 100).toFixed(d) + "%" : "—";
  const spct = (v, d = 1) => ok(v) ? (v >= 0 ? "+" : "−") + Math.abs(v * 100).toFixed(d) + "%" : "—";
  const num = (v, d = 2) => ok(v) ? v.toFixed(d) : "—";
  const usd = (v) => ok(v) ? "$" + Math.round(v).toLocaleString("en-US") : "—";
  const tone = (v) => !ok(v) ? "" : v >= 0 ? "pos" : "neg";
  const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined) e.textContent = text; return e; };
  const iso = (d) => d.toISOString().slice(0, 10);

  // ── holdings editor ──────────────────────────────────────────
  function addRow(t = "", w = "") {
    const rows = $("rows");
    if (rows.children.length >= MAX_T) return;
    const tr = el("tr");
    const ti = el("input", "t"); ti.value = t; ti.placeholder = "Ticker"; ti.maxLength = 12; ti.setAttribute("aria-label", "Ticker");
    const wi = el("input", "w num"); wi.value = w; wi.placeholder = "Weight %"; wi.inputMode = "decimal"; wi.setAttribute("aria-label", "Weight percent");
    const x = el("button", "x", "×"); x.type = "button"; x.title = "Remove"; x.setAttribute("aria-label", "Remove " + (t || "row"));
    x.onclick = () => { if (rows.children.length > 1) tr.remove(); updateSum(); };
    wi.oninput = updateSum;
    [ti, wi, x].forEach((n) => { const td = el("td"); td.appendChild(n); tr.appendChild(td); });
    rows.appendChild(tr); updateSum();
  }
  function readHoldings() {
    return [...$("rows").children].map((tr) => {
      const [t, w] = tr.querySelectorAll("input");
      return { ticker: t.value.trim().toUpperCase(), weight: parseFloat(w.value) };
    }).filter((h) => h.ticker || ok(h.weight));
  }
  function updateSum() {
    const hs = readHoldings(), s = hs.reduce((a, h) => a + (ok(h.weight) ? h.weight : 0), 0);
    $("wsum").textContent = `${hs.length}/${MAX_T} · Σ ${s.toFixed(1)}%`;
    $("caphint").textContent = Math.abs(s - 100) > 0.05 && s > 0 ? "Weights will be scaled to 100%." : "";
    $("add").disabled = $("rows").children.length >= MAX_T;
  }
  function loadPreset(key) {
    const p = presets[key]; if (!p) return;
    $("rows").innerHTML = "";
    Object.entries(p.holdings).forEach(([t, w]) => addRow(t, w));
  }
  function setYears(y) {
    const end = new Date(); const start = new Date(end); start.setFullYear(end.getFullYear() - y);
    $("end").value = iso(end); $("start").value = iso(start);
  }

  // ── KPI strip ────────────────────────────────────────────────
  const KPI = [
    ["NAV", (k) => [usd(k.nav), ok(k.nav_day) ? `${k.nav_day >= 0 ? "+" : "−"}${usd(Math.abs(k.nav_day))} last day` : " ", "", tone(k.nav_day)]],
    ["CAGR", (k) => [pct(k.cagr), "Bench " + pct(k.bench_cagr)]],
    ["Excess / yr", (k) => [spct(k.excess), "Alpha " + pct(k.alpha) + (ok(k.alpha_t) ? ` (t ${k.alpha_t.toFixed(1)})` : ""), tone(k.excess)]],
    ["Volatility", (k) => [pct(k.vol), "Bench " + pct(k.bench_vol)]],
    ["Sharpe", (k) => [num(k.sharpe), "Sortino " + num(k.sortino)]],
    ["Max drawdown", (k) => [pct(k.max_dd), "Bench " + pct(k.bench_max_dd), "neg"]],
    ["Beta", (k) => [num(k.beta), "SE " + num(k.beta_se)]],
    ["Tracking error", (k) => [pct(k.tracking_error), "IR " + num(k.info_ratio)]],
  ];
  function renderKpis(k, skeleton) {
    const box = $("kpis"); box.innerHTML = ""; box.classList.toggle("skeleton", !!skeleton);
    KPI.forEach(([name, f]) => {
      const [v, s, vt, st] = k ? f(k) : ["—", " "];
      const c = el("div", "kpi"); c.appendChild(el("div", "k", name));
      c.appendChild(el("div", "v " + (vt || ""), v)); c.appendChild(el("div", "s " + (st || ""), s));
      box.appendChild(c);
    });
  }

  // ── charts ───────────────────────────────────────────────────
  function layout(extra = {}) {
    const ax = { gridcolor: C.grid, zeroline: false, showline: false, tickfont: { size: 11, color: C.muted } };
    return Object.assign({
      paper_bgcolor: C.bg, plot_bgcolor: C.bg, font: { family: "Inter, Segoe UI, sans-serif", color: C.text, size: 12 },
      margin: { l: 56, r: 16, t: 28, b: 32 }, xaxis: { ...ax }, yaxis: { ...ax },
      legend: { orientation: "h", x: 0, y: 1.0, xanchor: "left", yanchor: "bottom", font: { size: 11, color: C.muted }, bgcolor: "rgba(0,0,0,0)" },
      hovermode: "x unified", hoverlabel: { bgcolor: "#151A1F", bordercolor: "#2D333B", font: { family: "Inter, sans-serif", size: 12, color: C.text } },
    }, extra);
  }
  const cfg = { displayModeBar: false, responsive: true };
  const line = (s, name, color, extra = {}) => ({ x: s.x, y: s.y, name, type: "scatter", mode: "lines", line: { color, width: 1.6 }, ...extra });
  function plot(id, data, lay) { if (window.Plotly) Plotly.react($(id), data, lay, cfg); }

  function renderCharts(r) {
    const b = r.meta.benchmark;
    plot("ch-growth", [line(r.growth.portfolio, "Portfolio", C.port, { hovertemplate: "%{y:$,.0f}" }),
      line(r.growth.benchmark, b, C.bench, { hovertemplate: "%{y:$,.0f}" })],
      layout({ yaxis: { ...layout().yaxis, tickprefix: "$", tickformat: ",.0s" } }));
    plot("ch-dd", [line(r.drawdown.portfolio, "Portfolio", C.port, { fill: "tozeroy", fillcolor: "rgba(76,141,255,0.10)", hovertemplate: "%{y:.1%}" }),
      line(r.drawdown.benchmark, b, C.bench, { hovertemplate: "%{y:.1%}" })],
      layout({ yaxis: { ...layout().yaxis, tickformat: ".0%" } }));
  }
  function renderRiskCharts(r) {
    plot("ch-rs", [line(r.rolling_sharpe, "Rolling Sharpe (252d)", C.port, { hovertemplate: "%{y:.2f}" })],
      layout({ showlegend: false, shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: 0, y1: 0, line: { color: C.line, width: 1 } }] }));
  }
  function renderAttrChart(r) {
    const rows = r.attribution.sectors.filter((s) => Math.abs(s.Total || 0) > 1e-6)
      .sort((a, b) => Math.abs(b.Total) - Math.abs(a.Total)).slice(0, 12).sort((a, b) => b.Total - a.Total);
    const names = rows.map((s) => s.Sector);
    const bar = (key, color) => ({ type: "bar", orientation: "h", name: key, y: names, x: rows.map((s) => s[key]), marker: { color }, hovertemplate: "%{x:.1%}<extra>" + key + "</extra>" });
    plot("ch-attr", [bar("Allocation", C.port), bar("Selection", C.bench), bar("Interaction", C.amber)],
      layout({ barmode: "relative", hovermode: "closest", margin: { l: 16, r: 16, t: 28, b: 32 },
        xaxis: { ...layout().xaxis, tickformat: ".0%" }, yaxis: { ...layout().yaxis, autorange: "reversed", automargin: true } }));
  }

  // ── tables ───────────────────────────────────────────────────
  function table(id, cols, rows, opts = {}) {
    const t = $(id); t.innerHTML = "";
    const thead = el("thead"), hr = el("tr");
    cols.forEach((c, i) => {
      const th = el("th", opts.sortable ? "sortable" : "", c.label); if (c.title) th.title = c.title;
      if (opts.sortable) th.onclick = () => { const dir = opts._sortCol === i ? -opts._dir : -1; opts._sortCol = i; opts._dir = dir;
        const body = rows.slice(); body.sort((a, b) => { const x = c.key(a), y = c.key(b); return (x > y ? 1 : x < y ? -1 : 0) * dir; }); fill(body); };
      hr.appendChild(th);
    });
    thead.appendChild(hr); t.appendChild(thead);
    const tbody = el("tbody"); t.appendChild(tbody);
    function fill(body) {
      tbody.innerHTML = "";
      body.forEach((row) => {
        const tr = el("tr");
        cols.forEach((c) => { const v = c.fmt(row); const td = el("td", c.tone ? c.tone(row) : "");
          if (v instanceof Node) td.appendChild(v); else td.textContent = v; tr.appendChild(td); });
        tbody.appendChild(tr);
      });
      if (opts.total) { const tr = el("tr", "total"); cols.forEach((c) => { const td = el("td", c.tone ? c.tone(opts.total) : "", c.fmt(opts.total)); tr.appendChild(td); }); tbody.appendChild(tr); }
    }
    fill(rows);
  }
  const col = (label, key, fmt = (v) => v ?? "—", toneFn, title) => ({ label, title, key: (r) => r[key] ?? -Infinity, fmt: (r) => fmt(r[key]), tone: toneFn ? (r) => toneFn(r[key]) : null });
  const badge = (text, good) => { const s = el("span", "badge " + (good ? "ok" : "bad"), text); return s; };

  function renderTables(r) {
    const hold = r.holdings.filter((h) => h.Ticker !== "Portfolio"), tot = r.holdings.find((h) => h.Ticker === "Portfolio");
    table("t-hold", [col("Ticker", "Ticker"), col("Weight", "Weight", pct), col("Value", "Value", usd),
      col("Return", "Return", (v) => spct(v), tone), col("Contrib.", "Contribution", (v) => spct(v), tone, "Initial weight × return"),
      col("Risk %", "RiskContribution", pct, null, "Share of portfolio variance, last year"), col("Beta", "Beta", num)],
      hold, { sortable: true, total: tot });
    table("t-periods", [col("Period", "Period"), col("Portfolio", "Portfolio", spct, tone), col(r.meta.benchmark, "Benchmark", spct, tone), col("Excess", "Excess", spct, tone)], r.periods);
    table("t-var", [col("Method", "Method"), col("VaR 95", "VaR 95%", (v) => pct(v, 2)), col("ES 95", "ES 95%", (v) => pct(v, 2)), col("VaR 99", "VaR 99%", (v) => pct(v, 2)), col("ES 99", "ES 99%", (v) => pct(v, 2))], r.risk.var);
    table("t-varbt", [col("Method", "Method"), col("Breaches", "Breaches", (v) => num(v, 0)), col("Expected", "Expected", (v) => num(v, 1)),
      col("Kupiec p", "Kupiec p", (v) => num(v, 3)), { label: "Verdict", key: (x) => x.Verdict, fmt: (x) => badge(x.Verdict || "—", /pass|ok|adequate/i.test(x.Verdict || "")) }], r.risk.var_backtest);
    table("t-stress", [col("Scenario", "Scenario"), col("Period", "Period"), col("Portfolio", "Portfolio", (v) => v ?? "—", (v) => String(v || "").startsWith("-") ? "neg" : "pos"),
      col(r.meta.benchmark, "Benchmark", (v) => v ?? "—", (v) => String(v || "").startsWith("-") ? "neg" : "pos"), col("Max DD", "Max DD"), col("Basis", "Basis", (v) => v ?? "—", null)], r.risk.stress);
    table("t-dds", [col("Peak", "Peak"), col("Trough", "Trough"), col("Recovery", "Recovery", (v) => v || "Ongoing"), col("Depth", "Depth", pct, () => "neg"), col("Days", "Length", (v) => num(v, 0))], r.risk.drawdowns);
    const a = r.attribution;
    $("attr-head").textContent = `Sector attribution vs ${r.meta.benchmark} · active ${spct(a.active_return)} (portfolio ${spct(a.port_return)}, benchmark ${spct(a.bench_return)})`;
    table("t-attr", [col("Sector", "Sector"), col("Port wt", "Port weight", pct), col("Bench wt", "Bench weight", pct), col("Allocation", "Allocation", spct, tone),
      col("Selection", "Selection", spct, tone), col("Interaction", "Interaction", spct, tone), col("Total", "Total", spct, tone)], a.sectors, { sortable: true });
  }

  function renderAll(r) {
    last = r;
    $("empty").hidden = true;
    renderKpis(r.kpis);
    renderTables(r);
    showTab(activeTab, true);
    const m = r.meta;
    $("ctx").textContent = `${m.tickers.join(" · ")}  vs ${m.benchmark}  ·  ${m.start} → ${m.end}`;
    $("status").textContent = `Data: ${m.data_source} · as of ${m.end} · ${m.trading_days.toLocaleString()} trading days` +
      (m.risk_free ? ` · Rf ${pct(m.risk_free.rate, 2)} (${m.risk_free.source})` : "") + (r.cached ? " · cached" : "") + (ok(window.PA_TIMINGS.lastRunMs) ? ` · computed in browser in ${(window.PA_TIMINGS.lastRunMs / 1000).toFixed(1)}s` : "") + (m.offline ? " · offline (cached prices)" : "");
    const w = $("warns"); w.hidden = !r.warnings.length; w.textContent = r.warnings.length ? "Warnings: " + r.warnings.join(" · ") : "";
  }
  function showTab(name, force) {
    activeTab = name;
    document.querySelectorAll("#tabs button").forEach((b) => b.classList.toggle("on", b.dataset.tab === name));
    if (!last) return;
    document.querySelectorAll(".tab").forEach((t) => { t.hidden = t.dataset.tab !== name; });
    if (name === "overview") renderCharts(last);
    if (name === "risk") renderRiskCharts(last);
    if (name === "attribution") renderAttrChart(last);
  }

  // ── run ──────────────────────────────────────────────────────
  const STEPS = ["Fetching prices…", "Building portfolio…", "Running regressions…", "Computing VaR and stress tests…", "Running attribution…"];
  async function run(ev) {
    ev && ev.preventDefault();
    $("err").textContent = "";
    const holdings = readHoldings();
    if (!holdings.length) return ($("err").textContent = "Add at least one ticker.");
    if (holdings.some((h) => !h.ticker || !ok(h.weight) || h.weight <= 0)) return ($("err").textContent = "Each row needs a ticker and a positive weight.");
    const body = { holdings, benchmark: $("bench").value.trim().toUpperCase() || "SPY",
      start_date: $("start").value, end_date: $("end").value, capital: parseFloat(String($("capital").value).replace(/[,$\s]/g, "")) || 100000 };
    $("run").disabled = true; $("loading").hidden = false; if (!last) renderKpis(null, true);
    let i = 0; $("loadmsg").textContent = STEPS[0];
    const tick = setInterval(() => { if (!engine.ready) return; i = Math.min(i + 1, STEPS.length - 1); $("loadmsg").textContent = STEPS[i]; }, 900);
    try {
      if (!engine.ready) { $("loadmsg").textContent = "Loading the analytics engine (first visit only)…"; await whenReady(); }
      const t0 = performance.now();
      const m = await analyzeInEngine(body);
      const data = JSON.parse(m.text);
      if (data.error) throw new Error(data.error);
      window.PA_TIMINGS.lastRunMs = Math.round(performance.now() - t0);
      renderAll(data);
      if (window.innerWidth <= 860) $("kpis").scrollIntoView({ behavior: "smooth" });
    } catch (e) {
      $("err").textContent = e.message;
      if (!last) renderKpis(null);
    } finally { clearInterval(tick); $("run").disabled = false; $("loading").hidden = true; }
  }

  async function init() {
    renderKpis(null);
    setYears(5);
    const sel = $("preset");
    Object.entries(presets).forEach(([k, p]) => { const o = el("option", "", p.label); o.value = k; sel.appendChild(o); });
    const o = el("option", "", "Custom"); o.value = ""; sel.appendChild(o);
    sel.onchange = () => sel.value && loadPreset(sel.value);
    loadPreset(Object.keys(presets)[0]);
    $("add").onclick = () => { addRow(); sel.value = ""; };
    $("rows").addEventListener("input", () => { sel.value = ""; });
    $("ranges").onclick = (e) => { const y = e.target.dataset.y; if (y) setYears(Math.min(+y, MAX_Y)); };
    $("tabs").onclick = (e) => { const t = e.target.dataset.tab; if (t) showTab(t); };
    $("form").onsubmit = run;
    try { startEngine(); } catch (e) { setEngine(0, "Could not start the engine: " + e.message); }
    if (window.PA_ENGINE) { window.PA_ENGINE.appReady = true; if (window.PA_ENGINE.pendingRun) run(); }
    if ("serviceWorker" in navigator && location.protocol === "https:") navigator.serviceWorker.register("sw.js").catch(() => {});
    window.addEventListener("resize", () => last && showTab(activeTab));
  }
  document.readyState === "loading" ? document.addEventListener("DOMContentLoaded", init) : init();
})();
