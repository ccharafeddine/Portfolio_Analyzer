// Local preview of the production layout: serves web/dist (with its _headers) and
// routes /v1/* to the price proxy, like the portfolio.chafic.me/v1/* Worker route.
//   node web/worker/dev-server.mjs                       # /v1 runs src/index.js in-process
//   UPSTREAM=http://127.0.0.1:8787 node web/worker/dev-server.mjs   # /v1 -> `wrangler dev`
import http from "node:http";
import { readFile, stat } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import worker from "./src/index.js";

const here = path.dirname(fileURLToPath(import.meta.url));
const DIST = path.resolve(here, "..", "dist");
const PORT = Number(process.env.PORT || 8788);
const UPSTREAM = process.env.UPSTREAM || "";
const TYPES = { ".html": "text/html; charset=utf-8", ".js": "text/javascript", ".css": "text/css", ".json": "application/json", ".zip": "application/zip", ".png": "image/png", ".svg": "image/svg+xml" };

if (!globalThis.caches) { // tiny Cache API shim for the in-process Worker
  const m = new Map();
  globalThis.caches = { default: { async match(r) { const v = m.get(r.url); return v ? new Response(v) : undefined; }, async put(r, res) { m.set(r.url, await res.text()); } } };
}

async function headerRules() {
  try {
    const txt = await readFile(path.join(DIST, "_headers"), "utf8");
    const rules = []; let cur = null;
    for (const line of txt.split("\n")) {
      if (!line.trim()) continue;
      if (!/^\s/.test(line)) { cur = { re: new RegExp("^" + line.trim().replace(/[.+?^${}()|[\]\\]/g, "\\$&").replace(/\*/g, ".*") + "$"), h: {} }; rules.push(cur); }
      else if (cur) { const i = line.indexOf(":"); cur.h[line.slice(0, i).trim()] = line.slice(i + 1).trim(); }
    }
    return rules;
  } catch { return []; }
}

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, `http://${req.headers.host}`);
  try {
    if (url.pathname.startsWith("/v1/")) {
      const headers = { ...req.headers, "cf-connecting-ip": req.socket.remoteAddress };
      const r = UPSTREAM ? await fetch(UPSTREAM + url.pathname + url.search, { headers: { Origin: headers.origin || "", "CF-Connecting-IP": headers["cf-connecting-ip"] } })
        : await worker.fetch(new Request(url, { headers }), {}, { waitUntil() {} });
      const h = Object.fromEntries(r.headers);
      for (const k of ["content-encoding", "content-length", "transfer-encoding", "connection", "keep-alive"]) delete h[k]; // body is already decoded
      res.writeHead(r.status, h); res.end(Buffer.from(await r.arrayBuffer())); return;
    }
    let p = path.normalize(path.join(DIST, decodeURIComponent(url.pathname)));
    if (!p.startsWith(DIST)) { res.writeHead(403); res.end(); return; }
    if ((await stat(p).catch(() => null))?.isDirectory()) p = path.join(p, "index.html");
    const body = await readFile(p);
    const h = { "Content-Type": TYPES[path.extname(p)] || "application/octet-stream" };
    for (const r of await headerRules()) if (r.re.test(url.pathname)) Object.assign(h, r.h);
    res.writeHead(200, h); res.end(body);
  } catch (e) { res.writeHead(e.code === "ENOENT" ? 404 : 500); res.end(String(e.code || e)); }
});
server.listen(PORT, () => console.log(`demo on http://localhost:${PORT} (proxy: ${UPSTREAM || "in-process"})`));
