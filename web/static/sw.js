/* Service worker: cache-first for versioned assets (the Pyodide runtime and wheels
   from the CDN, the content-hashed engine bundle) so repeat visits skip the ~30 MB
   download; network-first for the HTML/JS shell so deploys show up immediately. */
const CACHE = "pa-demo-v1";
const IMMUTABLE = [/^https:\/\/cdn\.jsdelivr\.net\/pyodide\/v[\d.]+\/full\//, /\/pa_engine\.[0-9a-f]{12}\.zip$/];

self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (e) => e.waitUntil((async () => {
  for (const k of await caches.keys()) if (k !== CACHE) await caches.delete(k);
  await self.clients.claim();
})()));

self.addEventListener("fetch", (e) => {
  const req = e.request;
  if (req.method !== "GET") return;
  const url = new URL(req.url);
  if (url.pathname.startsWith("/v1/")) return; // price proxy: leave to HTTP/edge caching
  if (IMMUTABLE.some((re) => re.test(req.url))) {
    e.respondWith(caches.open(CACHE).then(async (c) => {
      const hit = await c.match(req);
      if (hit) return hit;
      const res = await fetch(req);
      if (res.ok) c.put(req, res.clone());
      return res;
    }));
  } else if (url.origin === self.location.origin) {
    e.respondWith(fetch(req).then((res) => {
      if (res.ok) caches.open(CACHE).then((c) => c.put(req, res.clone()));
      return res;
    }).catch(() => caches.match(req)));
  }
});
