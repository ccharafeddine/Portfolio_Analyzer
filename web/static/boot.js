/* Starts the analytics engine as early as possible: this tiny script runs from <head>,
   before the stylesheet, Plotly and app.js, and kicks off the Pyodide Web Worker.
   app.js picks up window.PA_ENGINE (and any messages that arrived before it). */
(function () {
  "use strict";
  var M = window.PA_MANIFEST = __MANIFEST__;
  var E = window.PA_ENGINE = { manifest: M, queue: [], onmessage: null, t0: performance.now() };
  try {
    var proxy = (new URLSearchParams(location.search).get("proxy") || M.proxy || "").replace(/\/$/, "");
    var w = E.worker = new Worker("engine.worker.js?v=" + encodeURIComponent(M.bundle));
    w.onmessage = function (ev) { E.onmessage ? E.onmessage(ev) : E.queue.push(ev); };
    w.postMessage({ type: "init", manifest: M, proxyBase: proxy });
  } catch (e) { E.error = String(e && e.message || e); }
  // A Run click before app.js has loaded must not submit the form natively (reload).
  document.addEventListener("submit", function (ev) {
    if (!E.appReady) { ev.preventDefault(); E.pendingRun = true; }
  }, true);
})();
