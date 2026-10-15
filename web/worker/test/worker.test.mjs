// Unit tests for the price proxy. Runs on plain Node (>=18): `node --test test/`.
// fetch and the Cache API are stubbed, so no network or Cloudflare account is needed.
import { test, beforeEach } from "node:test";
import assert from "node:assert/strict";
import worker, { compactChart, parseSymbols, parseDates, memoryLimit, _resetMemoryLimit } from "../src/index.js";

const ORIGIN = "https://portfolio.chafic.me";
let upstream = [];
let store = new Map();

function chartBody(sym) {
  // two days, gmtoffset -5h; second close null to test filtering
  return { chart: { result: [{ meta: { symbol: sym, gmtoffset: -18000 },
    timestamp: [1704205800, 1704292200, 1704378600],
    indicators: { quote: [{ close: [10, 11, 12] }], adjclose: [{ adjclose: [9.5, null, 11.5] }] } }], error: null } };
}

beforeEach(() => {
  upstream = []; store = new Map(); _resetMemoryLimit();
  globalThis.caches = { default: {
    async match(req) { const v = store.get(req.url); return v ? new Response(v) : undefined; },
    async put(req, res) { store.set(req.url, await res.text()); },
  } };
  globalThis.fetch = async (url, init) => {
    upstream.push({ url: String(url), init });
    const u = new URL(String(url));
    if (u.pathname.includes("/v8/finance/chart/")) {
      const sym = decodeURIComponent(u.pathname.split("/").pop());
      if (sym === "NOPE") return new Response(JSON.stringify({ chart: { result: null, error: { code: "Not Found", description: "No data found, symbol may be delisted" } } }), { status: 404 });
      return new Response(JSON.stringify(chartBody(sym)), { status: 200 });
    }
    if (u.pathname === "/v1/finance/search") {
      return new Response(JSON.stringify({ quotes: [{ symbol: "AAPLX", sector: "X" }, { symbol: u.searchParams.get("q"), sector: "Technology", quoteType: "EQUITY", shortname: "Apple" }] }));
    }
    throw new Error("unexpected upstream " + url);
  };
});

const req = (path, headers = {}) => new Request("https://portfolio.chafic.me" + path, { headers: { Origin: ORIGIN, "CF-Connecting-IP": "1.2.3.4", ...headers } });
const call = async (path, headers, env = {}) => { const r = await worker.fetch(req(path, headers), env, { waitUntil() {} }); return [r, await r.json()]; };

test("chart: compact adjusted closes, dates in exchange time, nulls dropped", async () => {
  const [r, body] = await call("/v1/chart?symbols=aapl,%5EIRX&start=2024-01-01&end=2024-01-10");
  assert.equal(r.status, 200);
  assert.equal(r.headers.get("Access-Control-Allow-Origin"), ORIGIN);
  assert.deepEqual(body.data.AAPL, { d: ["2024-01-02", "2024-01-04"], c: [9.5, 11.5] });
  assert.ok(body.data["^IRX"]);
  const u = new URL(upstream[0].url);
  assert.equal(u.origin, "https://query1.finance.yahoo.com");
  assert.equal(u.searchParams.get("interval"), "1d");
});

test("chart: edge cache hit avoids a second upstream call", async () => {
  await call("/v1/chart?symbols=MSFT&start=2020-01-01&end=2020-12-31");
  await call("/v1/chart?symbols=MSFT&start=2020-01-01&end=2020-12-31");
  assert.equal(upstream.length, 1);
});

test("chart: unknown symbol reported per symbol, not a hard failure", async () => {
  const [r, body] = await call("/v1/chart?symbols=AAPL,NOPE&start=2024-01-01&end=2024-02-01");
  assert.equal(r.status, 200);
  assert.ok(body.data.AAPL);
  assert.match(body.errors.NOPE, /Unknown symbol/);
});

test("validation: bad symbols, too many, bad dates", async () => {
  for (const q of ["symbols=%3Cscript%3E&start=2024-01-01&end=2024-02-01",
    "symbols=A/B&start=2024-01-01&end=2024-02-01",
    "symbols=" + Array.from({ length: 26 }, (_, i) => "T" + i).join(",") + "&start=2024-01-01&end=2024-02-01",
    "symbols=AAPL&start=2024-13-01&end=2024-02-01", "symbols=AAPL&start=1980-01-01&end=2024-02-01",
    "symbols=AAPL&start=2024-03-01&end=2024-02-01", "symbols=&start=2024-01-01&end=2024-02-01"]) {
    const [r, body] = await call("/v1/chart?" + q);
    assert.equal(r.status, 400, q);
    assert.ok(body.error, q);
  }
  assert.equal(upstream.length, 0);
});

test("only the two allowlisted routes exist; no URL passthrough", async () => {
  for (const p of ["/v1/chart/https://evil.example", "/", "/v8/finance/chart/AAPL", "/v1/quote?url=https://x"]) {
    const [r] = await call(p);
    assert.equal(r.status, 404, p);
  }
  const post = await worker.fetch(new Request("https://x/v1/chart", { method: "POST", headers: { Origin: ORIGIN } }), {}, {});
  assert.equal(post.status, 405);
});

test("CORS: foreign origin rejected, preflight answered for allowed origin", async () => {
  const [r] = await call("/v1/chart?symbols=AAPL&start=2024-01-01&end=2024-02-01", { Origin: "https://evil.example" });
  assert.equal(r.status, 403);
  assert.equal(r.headers.get("Access-Control-Allow-Origin"), null);
  const pre = await worker.fetch(new Request("https://x/v1/chart", { method: "OPTIONS", headers: { Origin: "http://localhost:8788" } }), {}, {});
  assert.equal(pre.status, 204);
  assert.equal(pre.headers.get("Access-Control-Allow-Origin"), "http://localhost:8788");
});

test("rate limit: in-memory fallback and the Workers binding", async () => {
  const env = { RATE_LIMIT: "2" };
  const codes = [];
  for (let i = 0; i < 3; i++) codes.push((await call("/v1/profile?symbols=AAPL", {}, env))[0].status);
  assert.deepEqual(codes, [200, 200, 429]);
  const denied = { LIMITER: { limit: async ({ key }) => ({ success: key !== "1.2.3.4" }) } };
  assert.equal((await call("/v1/profile?symbols=AAPL", {}, denied))[0].status, 429);
});

test("profile: exact-symbol match from search", async () => {
  const [, body] = await call("/v1/profile?symbols=AAPL");
  assert.deepEqual(body.profiles.AAPL, { sector: "Technology", quoteType: "EQUITY", name: "Apple" });
  const [r] = await call("/v1/profile?symbols=A,B,C,D,E,F");
  assert.equal(r.status, 400);
});

test("helpers", () => {
  assert.deepEqual(parseSymbols(" spy , ^irx,SPY", 5), ["SPY", "^IRX"]);
  const { p1, p2 } = parseDates("2024-01-01", "2024-01-31", new Date("2026-01-01"));
  assert.equal(p2 - p1, 31 * 86400);
  const capped = parseDates("2024-01-01", "2030-01-01", new Date("2026-01-01T12:00:00Z"));
  assert.equal(capped.p2, Date.UTC(2026, 0, 2) / 1000);
  assert.throws(() => compactChart({ chart: { result: null, error: { description: "gone" } } }), /gone/);
  assert.equal(memoryLimit("k", 1, 1000, 0), true);
  assert.equal(memoryLimit("k", 1, 1000, 500), false);
  assert.equal(memoryLimit("k", 1, 1000, 1500), true);
});
