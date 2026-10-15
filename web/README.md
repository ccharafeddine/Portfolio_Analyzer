# Portfolio Analyzer · in-browser web demo

A static site for `portfolio.chafic.me` that runs the desktop app's own Python
analytics **in the visitor's browser** with [Pyodide](https://pyodide.org). The only
server piece is a small Cloudflare Worker that relays Yahoo Finance price data.

```
browser ─ index.html / app.js (UI, Plotly)            Cloudflare Pages (static, web/dist)
          └─ engine.worker.js (Web Worker)
               └─ Pyodide 0.29.3 + numpy, pandas, pydantic   (jsDelivr CDN; no scipy)
                    └─ pa_engine.<hash>.zip: src/pipeline.py, src/analytics/*, src/config/*,
                       a few src/data/* modules + web/py/pa_web (adapter)
                         └─ sync XHR ─▶ /v1/chart, /v1/profile ─▶ Worker ─▶ Yahoo (2 endpoints)
```

* `web/py/pa_web/` – `engine.py` (entry points), `fetcher.py` (ProxyClient: drop-in for
  `fetch_prices`), `yf_stub.py` (stands in for `yfinance`: sector lookups via the proxy,
  everything else falls back as on the desktop), `schemas.py` (caps: ≤20 tickers,
  ≤15 years, ≥6 months), `service.py` (lean `DemoPipeline` + JSON).
* `web/static/` – UI (Graphite), `engine.worker.js`, `sw.js` (caches the runtime and
  hashed bundle), `_headers` (CSP, caching).
* `web/worker/` – price proxy (`src/index.js`, `wrangler.toml`), unit tests (`test/`),
  local preview server, and the headless end-to-end and timing scripts (`e2e/`).
* `web/build.py` – builds `web/dist/` (stdlib only).

## Load time

* `boot.js` (in `<head>`) starts the engine Worker before the stylesheet, Plotly and
  `app.js`; the form is usable immediately and a Run clicked early waits for the engine.
* The Worker downloads the runtime, stdlib, our bundle and every wheel in parallel with
  the runtime boot, de-duplicating Pyodide's own requests (no double downloads).
* scipy is never loaded. The shared analytics import it lazily (desktop unaffected), and
  the few `scipy.stats` calls in the demo steps are served by `pa_web/stats_lite.py`,
  tested against scipy and for identical demo output. statsmodels is likewise not
  needed (its factor/scenario models are desktop-only).
* Measure with `web/worker/e2e/timing.mjs` (cold and repeat visits).

## Develop and test

```
python web/build.py                       # -> web/dist
node web/worker/dev-server.mjs            # http://localhost:8788, /v1 runs the Worker in-process
# or against the real Workers runtime:
#   (cd web/worker && npx wrangler dev --port 8787) &
#   UPSTREAM=http://127.0.0.1:8787 node web/worker/dev-server.mjs

pytest tests/test_web_demo.py             # engine + fetcher adapter (CPython, offline)
node --test web/worker/test/worker.test.mjs              # Worker unit tests (Node >= 18, no deps)
BASE=http://localhost:8788 OUT=/tmp/e2e node web/worker/e2e/e2e.mjs   # needs `npm i playwright`
```

## Deploy

1. **Worker**: `cd web/worker && npx wrangler deploy` (account from login or
   `CLOUDFLARE_ACCOUNT_ID`; nothing is committed). Then enable the commented
   `routes` entry (`portfolio.chafic.me/v1/*`, zone `chafic.me`) and deploy again so the
   page reaches the proxy on its own origin.
2. **Pages**: project `portfolio-analyzer-demo`, build command `python3 web/build.py`,
   output directory `web/dist` (or direct upload: `python web/build.py && npx wrangler
   pages deploy web/dist --project-name portfolio-analyzer-demo`).
3. **Domain**: add custom domain `portfolio.chafic.me` to the Pages project.

To use the Worker's `workers.dev` URL instead of the route, build with
`python web/build.py --proxy=https://portfolio-price-proxy.<subdomain>.workers.dev`
(already allowed by the CSP's `connect-src`).
