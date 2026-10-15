// Headless end-to-end run of the built site: cold load (empty cache) -> engine ready ->
// analysis -> tabs, at desktop and phone widths. Needs `python web/build.py` and a
// running preview (node web/worker/dev-server.mjs). Writes screenshots + timings.json.
//   BASE=http://localhost:8788 OUT=/tmp/shots node web/worker/test/e2e.mjs
import { chromium } from "playwright";
import { mkdirSync, writeFileSync } from "node:fs";

const BASE = process.env.BASE || "http://localhost:8788";
const OUT = (process.env.OUT || "./e2e-out").replace(/\/$/, "") + "/";
const PREFIX = process.env.PREFIX || "inbrowser-";
mkdirSync(OUT, { recursive: true });
const fail = (m) => { console.error("E2E FAIL:", m); process.exitCode = 1; };

const browser = await chromium.launch();
const timings = {};
for (const [name, vp] of [["desktop", { width: 1440, height: 1000 }], ["mobile", { width: 390, height: 844 }]]) {
  const ctx = await browser.newContext({ viewport: vp, deviceScaleFactor: name === "mobile" ? 2 : 1 }); // fresh = cold cache
  const p = await ctx.newPage();
  const errs = [];
  let bytes = 0; // encoded bytes over the wire during cold load (all requests incl. workers)
  ctx.on("requestfinished", async (r) => { try { const z = await r.sizes(); bytes += z.responseBodySize + z.responseHeadersSize; } catch {} });
  p.on("pageerror", (e) => errs.push(e.message));
  p.on("console", (m) => m.type() === "error" && errs.push(m.text()));
  const t0 = Date.now();
  await p.goto(BASE + "/", { waitUntil: "domcontentloaded" });
  await p.waitForTimeout(2500);
  await p.screenshot({ path: OUT + `${PREFIX}${name}-0-first-load.png` });
  await p.waitForFunction(() => window.PA_TIMINGS && window.PA_TIMINGS.engineReady, null, { timeout: 180000 });
  const readyMs = Date.now() - t0;
  const coldBytes = bytes;
  const inside = await p.evaluate(() => window.PA_TIMINGS);
  await p.click("#run");
  await p.waitForTimeout(300);
  await p.screenshot({ path: OUT + `${PREFIX}${name}-1-running.png` });
  await p.waitForSelector("#loading", { state: "hidden", timeout: 120000 });
  const err = (await p.textContent("#err")).trim();
  if (err) fail(`${name}: ${err}`);
  const firstRun = await p.evaluate(() => window.PA_TIMINGS.lastRunMs);
  await p.waitForTimeout(800);
  const kpi = await p.textContent("#kpis");
  if (!/\$\d/.test(kpi)) fail(`${name}: KPI strip empty: ${kpi}`);
  await p.screenshot({ path: OUT + `${PREFIX}${name}-2-overview.png`, fullPage: true });
  for (const [i, t] of [[3, "risk"], [4, "attribution"]]) {
    await p.click(`#tabs button[data-tab=${t}]`); await p.waitForTimeout(800);
    await p.screenshot({ path: OUT + `${PREFIX}${name}-${i}-${t}.png`, fullPage: true });
  }
  // second, different portfolio: prices for new tickers + cached benchmark/^IRX/sector ETFs
  await p.click("#tabs button[data-tab=overview]");
  await p.selectOption("#preset", "sixty_forty");
  await p.click("#run");
  await p.waitForSelector("#loading", { state: "hidden", timeout: 120000 });
  const secondRun = await p.evaluate(() => window.PA_TIMINGS.lastRunMs);
  // invalid input is rejected with a message, not a crash
  await p.fill("#rows tr:first-child input.t", "BAD/SYM");
  await p.click("#run");
  await p.waitForSelector("#loading", { state: "hidden" });
  const badMsg = (await p.textContent("#err")).trim();
  if (!/Invalid symbol/.test(badMsg)) fail(`${name}: bad input message: ${badMsg}`);
  if (errs.length) fail(`${name}: page errors: ${errs.join(" | ")}`);
  // repeat visit in the same browser profile: runtime + wheels come from the HTTP cache
  const t1 = Date.now();
  await p.reload({ waitUntil: "domcontentloaded" });
  await p.waitForFunction(() => window.PA_TIMINGS && window.PA_TIMINGS.engineReady, null, { timeout: 180000 });
  const warmReadyMs = Date.now() - t1;
  timings[name] = { coldTransferMB: +(coldBytes / 1e6).toFixed(1), warmReloadEngineReadyMs: warmReadyMs, engineReadyMs: readyMs, engineInside: inside.engineInside, engineWorkerMs: inside.engineMs, firstAnalysisMs: firstRun, secondAnalysisMs: secondRun };
  await ctx.close();
}
await browser.close();
writeFileSync(OUT + `${PREFIX}timings.json`, JSON.stringify(timings, null, 2));
console.log(JSON.stringify(timings, null, 2));
