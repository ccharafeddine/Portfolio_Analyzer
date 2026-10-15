/* Web Worker that hosts Pyodide and the Portfolio Analyzer engine, so the page stays
   responsive while Python loads and runs. Protocol:
     in : {type:"init", manifest, proxyBase}       {type:"analyze", id, payload}
     out: {type:"progress", pct, msg}  {type:"ready", ms, timings}
          {type:"result", id, text, ms}  {type:"error", id?, msg}            */
"use strict";
let pyodide = null, engine = null, ready = null;
const post = (m) => self.postMessage(m);
const progress = (pct, msg) => post({ type: "progress", pct, msg });

async function init(manifest, proxyBase) {
  const t0 = performance.now(), timings = {};
  const mark = (k, t) => { timings[k] = Math.round(performance.now() - t); return performance.now(); };
  const indexURL = `https://cdn.jsdelivr.net/pyodide/v${manifest.pyodide}/full/`;
  progress(3, "Loading Python runtime…");
  let t = performance.now();
  importScripts(indexURL + "pyodide.js");
  pyodide = await loadPyodide({ indexURL });
  t = mark("runtime", t);
  progress(30, "Loading numpy, pandas, scipy…");
  let n = 0;
  await pyodide.loadPackage(manifest.packages, {
    messageCallback: (m) => { if (/^Loaded /.test(m)) progress(Math.min(80, 30 + (++n) * 5), m.replace(/^Loaded /, "Loaded ")); },
    errorCallback: (m) => console.warn(m),
  });
  t = mark("packages", t);
  progress(84, "Loading analytics engine…");
  const res = await fetch(manifest.bundle);
  if (!res.ok) throw new Error(`Engine bundle HTTP ${res.status}`);
  pyodide.unpackArchive(await res.arrayBuffer(), "zip", { extractDir: "/home/pyodide/app" });
  pyodide.runPython("import sys; sys.path.insert(0, '/home/pyodide/app')");
  t = mark("bundle", t);
  progress(92, "Importing pipeline…");
  engine = pyodide.pyimport("pa_web.engine");
  engine.install(proxyBase);
  pyodide.pyimport("src.pipeline");
  pyodide.pyimport("pa_web.service");
  mark("import", t);
  progress(100, "Ready");
  post({ type: "ready", ms: Math.round(performance.now() - t0), timings });
}

self.onmessage = async (ev) => {
  const m = ev.data || {};
  if (m.type === "init") {
    if (!ready) ready = init(m.manifest, m.proxyBase || "").catch((e) => { ready = null; post({ type: "error", msg: "Engine failed to load: " + (e.message || e) }); throw e; });
    return;
  }
  if (m.type === "analyze") {
    try {
      await ready;
      const t0 = performance.now();
      const text = engine.analyze(m.payload);
      post({ type: "result", id: m.id, text, ms: Math.round(performance.now() - t0) });
    } catch (e) {
      const msg = String(e && e.message || e);
      post({ type: "error", id: m.id, msg: msg.includes("Traceback") ? "The analysis failed: " + msg.trim().split("\n").pop() : msg });
    }
  }
};
