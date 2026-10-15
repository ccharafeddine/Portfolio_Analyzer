// First- and repeat-visit timings for a running preview (see dev-server.mjs).
//   BASE=http://localhost:8788 RUNS=3 node web/worker/e2e/timing.mjs
// cold   = fresh browser profile (empty HTTP cache); repeat = reload in the same profile.
// readyMs     navigation -> engine ready (Run can start computing)
// clickToResultMs  navigation -> first result when Run is clicked as soon as it is visible
import { chromium } from "playwright";

const BASE = process.env.BASE || "http://localhost:8788";
const RUNS = Number(process.env.RUNS || 3);
const med = (a) => { const s = [...a].sort((x, y) => x - y); return s[Math.floor(s.length / 2)]; };

async function visit(page) {
  const t0 = Date.now();
  await page.goto(BASE + "/", { waitUntil: "commit" });
  // usable = form populated by app.js (presets listed); click Run right then
  await page.waitForFunction(() => document.querySelectorAll("#preset option").length > 1);
  const usableMs = Date.now() - t0;
  await page.click("#run");
  await page.waitForFunction(() => window.PA_TIMINGS && window.PA_TIMINGS.engineReady, null, { timeout: 180000 });
  const readyMs = Date.now() - t0;
  await page.waitForFunction(() => window.PA_TIMINGS && window.PA_TIMINGS.lastRunMs, null, { timeout: 180000 });
  const clickToResultMs = Date.now() - t0;
  const err = (await page.textContent("#err")).trim();
  if (err) throw new Error(err);
  return { usableMs, readyMs, clickToResultMs };
}

const browser = await chromium.launch();
const out = { cold: [], repeat: [] };
for (let i = 0; i < RUNS; i++) {
  const ctx = await browser.newContext({ viewport: { width: 1440, height: 1000 } });
  const page = await ctx.newPage();
  out.cold.push(await visit(page));
  out.repeat.push(await visit(page));
  await ctx.close();
}
await browser.close();
const summary = {};
for (const k of ["cold", "repeat"]) summary[k] = Object.fromEntries(Object.keys(out[k][0]).map((m) => [m, med(out[k].map((r) => r[m]))]));
console.log(JSON.stringify({ base: BASE, runs: RUNS, median: summary, raw: out }, null, 1));
