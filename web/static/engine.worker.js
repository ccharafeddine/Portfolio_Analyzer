/* Web Worker that hosts Pyodide and the Portfolio Analyzer engine, so the page stays
   responsive while Python loads and runs. Started from boot.js in <head>, before the
   rest of the page loads. Protocol:
     in : {type:"init", manifest, proxyBase}       {type:"analyze", id, payload}
     out: {type:"progress", pct, msg}  {type:"ready", ms, timings, stats}
          {type:"result", id, text, ms}  {type:"error", id?, msg}

   Load plan (critical path first):
     1. In parallel: pyodide.js, the wasm runtime, stdlib, lock file, our bundle, and
        (once the lock is in) every wheel we need. Downloads overlap the runtime boot.
     2. numpy + pandas + pydantic -> import the pipeline -> "ready".
   scipy is never loaded: the shared modules import it lazily, and the few
   scipy.stats calls the demo steps make are served by pa_web/stats_lite.py
   (saves a 13 MB download and ~2 s of import time).                              */
"use strict";
let pyodide = null, engine = null, ready = null;
const post = (m) => self.postMessage(m);
const progress = (pct, msg) => post({ type: "progress", pct, msg });

// Prefetch with in-flight de-duplication: Pyodide's own fetch() of a URL we already
// requested gets our pending response instead of starting a second download.
const origFetch = self.fetch.bind(self);
const pending = new Map();
const stats = { prefetched: 0, reused: 0, other: 0 };
self.fetch = (input, init) => {
  const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
  const p = pending.get(url);
  if (p) { pending.delete(url); stats.reused++; return p.then((r) => r.clone()); }
  stats.other++;
  return origFetch(input, init);
};
function prefetch(url) {
  if (!pending.has(url)) {
    stats.prefetched++;
    const p = origFetch(url, { credentials: "omit", mode: "cors" });
    p.catch(() => pending.delete(url));
    pending.set(url, p);
  }
  return pending.get(url);
}
function deps(lock, names) {
  const pk = {}; for (const [k, v] of Object.entries(lock.packages)) pk[k.toLowerCase().replace(/_/g, "-")] = v;
  const seen = new Set(), out = [];
  const walk = (n) => { n = n.toLowerCase().replace(/_/g, "-"); if (seen.has(n) || !pk[n]) return; seen.add(n); pk[n].depends.forEach(walk); out.push(pk[n].file_name); };
  names.forEach(walk);
  return out;
}

async function init(manifest, proxyBase) {
  const CORE = manifest.packages;  // numpy, pandas, pydantic (see web/build.py)
  const t0 = performance.now(), timings = {};
  const mark = (k) => { timings[k] = Math.round(performance.now() - t0); };
  const indexURL = `https://cdn.jsdelivr.net/pyodide/v${manifest.pyodide}/full/`;
  progress(3, "Downloading Python runtime…");
  ["pyodide.asm.wasm", "python_stdlib.zip"].forEach((f) => prefetch(indexURL + f));
  const bundleP = origFetch(manifest.bundle).then((r) => { if (!r.ok) throw new Error(`Engine bundle HTTP ${r.status}`); return r.arrayBuffer(); });
  const lockP = prefetch(indexURL + "pyodide-lock.json").then((r) => r.clone().json()).then((lock) => {
    deps(lock, CORE).forEach((f) => prefetch(indexURL + f));
  }).catch(() => {});
  importScripts(indexURL + "pyodide.js");
  pyodide = await loadPyodide({ indexURL });
  mark("runtime");
  await lockP;
  progress(35, "Loading numpy, pandas…");
  let n = 0;
  await pyodide.loadPackage(CORE, {
    messageCallback: (m) => { if (/^Loaded /.test(m)) progress(Math.min(75, 35 + (++n) * 5), m); },
    errorCallback: (m) => console.warn(m),
  });
  mark("packages");
  progress(80, "Loading analytics engine…");
  pyodide.unpackArchive(await bundleP, "zip", { extractDir: "/home/pyodide/app" });
  pyodide.runPython("import sys; sys.path.insert(0, '/home/pyodide/app')");
  progress(88, "Importing pipeline…");
  engine = pyodide.pyimport("pa_web.engine");
  engine.install(proxyBase);
  pyodide.pyimport("pa_web.service");
  mark("import");
  progress(100, "Ready");
  post({ type: "ready", ms: Math.round(performance.now() - t0), timings, stats });
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
