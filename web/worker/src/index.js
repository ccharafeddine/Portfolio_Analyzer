/**
 * Portfolio Analyzer price proxy (Cloudflare Worker).
 *
 * The in-browser demo cannot call Yahoo Finance directly (no CORS), so this Worker
 * relays exactly two read-only Yahoo endpoints and nothing else:
 *
 *   GET /v1/chart?symbols=AAPL,MSFT&start=YYYY-MM-DD&end=YYYY-MM-DD
 *       -> {data: {SYM: {d: [dates], c: [adjusted closes]}}, errors: {SYM: msg}}
 *   GET /v1/profile?symbols=AAPL
 *       -> {profiles: {SYM: {sector, quoteType, name}}}
 *
 * Upstream URLs are built here from validated symbols and dates; the client never
 * supplies a URL. Responses are cached at the edge per symbol/date range, requests
 * are rate-limited per IP, and CORS is restricted to the demo's origins. No secrets.
 */

const CHART_HOST = "https://query1.finance.yahoo.com";
const SEARCH_HOST = "https://query2.finance.yahoo.com";
const SYMBOL_RE = /^[A-Z0-9.\-^=]{1,12}$/;
const DATE_RE = /^\d{4}-\d{2}-\d{2}$/;
const MAX_SYMBOLS = 25;
const MAX_PROFILE_SYMBOLS = 5;
const EARLIEST = "1990-01-01";
const DEFAULT_ORIGINS = ["https://portfolio.chafic.me", "http://localhost:8788", "http://127.0.0.1:8788"];
const UA = "Mozilla/5.0 (compatible; PortfolioAnalyzerDemo/1.0; +https://portfolio.chafic.me)";

// Fallback limiter when no Rate Limiting binding is configured (per isolate, best effort).
const memHits = new Map();
export function memoryLimit(key, limit, windowMs, now = Date.now()) {
  const q = (memHits.get(key) || []).filter((t) => now - t < windowMs);
  if (q.length >= limit) { memHits.set(key, q); return false; }
  q.push(now); memHits.set(key, q);
  if (memHits.size > 5000) memHits.clear();
  return true;
}
export function _resetMemoryLimit() { memHits.clear(); }

function allowedOrigins(env) {
  const raw = env && env.ALLOWED_ORIGINS;
  return raw ? raw.split(",").map((s) => s.trim()).filter(Boolean) : DEFAULT_ORIGINS;
}

function corsHeaders(request, env) {
  const origin = request.headers.get("Origin");
  const h = { "Vary": "Origin" };
  if (origin && allowedOrigins(env).includes(origin)) {
    h["Access-Control-Allow-Origin"] = origin;
    h["Access-Control-Allow-Methods"] = "GET, OPTIONS";
    h["Access-Control-Max-Age"] = "86400";
  }
  return h;
}

function json(body, status, request, env, extra = {}) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json; charset=utf-8", "X-Content-Type-Options": "nosniff",
      ...corsHeaders(request, env), ...extra },
  });
}

export function parseSymbols(raw, max) {
  if (!raw) throw new Error("symbols is required");
  const syms = [...new Set(raw.split(",").map((s) => s.trim().toUpperCase()).filter(Boolean))];
  if (!syms.length) throw new Error("symbols is required");
  if (syms.length > max) throw new Error(`At most ${max} symbols per request`);
  for (const s of syms) if (!SYMBOL_RE.test(s)) throw new Error(`Invalid symbol: ${s.slice(0, 16)}`);
  return syms;
}

export function parseDates(start, end, today = new Date()) {
  if (!DATE_RE.test(start || "") || !DATE_RE.test(end || "")) throw new Error("start and end must be YYYY-MM-DD");
  const s = Date.parse(start + "T00:00:00Z"), e = Date.parse(end + "T00:00:00Z");
  if (Number.isNaN(s) || Number.isNaN(e)) throw new Error("Invalid date");
  if (start < EARLIEST) throw new Error(`start must be on or after ${EARLIEST}`);
  if (e < s) throw new Error("end must be after start");
  const max = Date.UTC(today.getUTCFullYear(), today.getUTCMonth(), today.getUTCDate()) + 86400000;
  return { p1: Math.floor(s / 1000), p2: Math.floor(Math.min(e + 86400000, max) / 1000), endMs: e };
}

/** Yahoo chart JSON -> {d, c} using adjusted closes (falls back to close). */
export function compactChart(body) {
  const r = body && body.chart && body.chart.result && body.chart.result[0];
  if (!r) {
    const err = body && body.chart && body.chart.error;
    throw new Error((err && (err.description || err.code)) || "No data");
  }
  const ts = r.timestamp || [];
  const q = (r.indicators && r.indicators.quote && r.indicators.quote[0]) || {};
  const adj = (r.indicators && r.indicators.adjclose && r.indicators.adjclose[0] && r.indicators.adjclose[0].adjclose) || q.close || [];
  const off = (r.meta && r.meta.gmtoffset) || 0;
  const d = [], c = [];
  for (let i = 0; i < ts.length; i++) {
    const v = adj[i];
    if (v === null || v === undefined || !Number.isFinite(v)) continue;
    d.push(new Date((ts[i] + off) * 1000).toISOString().slice(0, 10));
    c.push(Math.round(v * 1e6) / 1e6);
  }
  if (!d.length) throw new Error("No data");
  return { d, c };
}

async function cachedUpstream(url, ttl, ctx) {
  const cache = typeof caches !== "undefined" ? caches.default : null;
  const key = new Request(url, { method: "GET" });
  if (cache) {
    const hit = await cache.match(key);
    if (hit) return { body: await hit.json(), cached: true };
  }
  const res = await fetch(url, { headers: { "User-Agent": UA, "Accept": "application/json" }, cf: { cacheTtl: ttl, cacheEverything: true } });
  if (res.status === 404) throw new Error("Unknown symbol");
  if (!res.ok) throw new Error(`Upstream HTTP ${res.status}`);
  const body = await res.json();
  if (cache) {
    const store = new Response(JSON.stringify(body), { headers: { "Content-Type": "application/json", "Cache-Control": `public, max-age=${ttl}` } });
    const p = cache.put(key, store);
    ctx && ctx.waitUntil ? ctx.waitUntil(p) : await p;
  }
  return { body, cached: false };
}

async function handleChart(url, ctx) {
  const syms = parseSymbols(url.searchParams.get("symbols"), MAX_SYMBOLS);
  const { p1, p2, endMs } = parseDates(url.searchParams.get("start"), url.searchParams.get("end"));
  // Closed historical windows never change; windows touching the last few days refresh hourly.
  const ttl = Date.now() - endMs > 4 * 86400000 ? 7 * 86400 : 3600;
  const data = {}, errors = {};
  await Promise.all(syms.map(async (s) => {
    const up = `${CHART_HOST}/v8/finance/chart/${encodeURIComponent(s)}?period1=${p1}&period2=${p2}&interval=1d&events=div%2Csplit&includeAdjustedClose=true`;
    try { data[s] = compactChart((await cachedUpstream(up, ttl, ctx)).body); }
    catch (e) { errors[s] = String(e.message || e).slice(0, 120); }
  }));
  return { body: { data, errors }, ttl };
}

async function handleProfile(url, ctx) {
  const syms = parseSymbols(url.searchParams.get("symbols"), MAX_PROFILE_SYMBOLS);
  const profiles = {};
  await Promise.all(syms.map(async (s) => {
    const up = `${SEARCH_HOST}/v1/finance/search?q=${encodeURIComponent(s)}&quotesCount=5&newsCount=0&listsCount=0`;
    try {
      const { body } = await cachedUpstream(up, 7 * 86400, ctx);
      const q = (body.quotes || []).find((x) => String(x.symbol).toUpperCase() === s) || null;
      profiles[s] = q ? { sector: q.sector || null, quoteType: q.quoteType || null, name: q.shortname || q.longname || null } : {};
    } catch { profiles[s] = {}; }
  }));
  return { body: { profiles }, ttl: 7 * 86400 };
}

export default {
  async fetch(request, env = {}, ctx) {
    const url = new URL(request.url);
    if (request.method === "OPTIONS") return new Response(null, { status: 204, headers: corsHeaders(request, env) });
    if (request.method !== "GET") return json({ error: "Method not allowed" }, 405, request, env, { Allow: "GET, OPTIONS" });
    const route = url.pathname.replace(/\/+$/, "");
    if (route === "/v1/health") return json({ ok: true }, 200, request, env);
    if (route !== "/v1/chart" && route !== "/v1/profile") return json({ error: "Not found" }, 404, request, env);

    const origin = request.headers.get("Origin");
    if (origin && !allowedOrigins(env).includes(origin)) return json({ error: "Origin not allowed" }, 403, request, env);

    const ip = request.headers.get("CF-Connecting-IP") || "anon";
    const limit = Number(env.RATE_LIMIT || 60);
    let allowed = true;
    if (env.LIMITER && typeof env.LIMITER.limit === "function") ({ success: allowed } = await env.LIMITER.limit({ key: ip }));
    else allowed = memoryLimit(ip, limit, 60000);
    if (!allowed) return json({ error: "Rate limit exceeded, try again in a minute" }, 429, request, env, { "Retry-After": "60" });

    try {
      const { body, ttl } = route === "/v1/chart" ? await handleChart(url, ctx) : await handleProfile(url, ctx);
      return json(body, 200, request, env, { "Cache-Control": `public, max-age=${Math.min(ttl, 3600)}` });
    } catch (e) {
      return json({ error: String(e.message || e) }, 400, request, env);
    }
  },
};
