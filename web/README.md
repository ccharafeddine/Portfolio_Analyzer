# Portfolio Analyzer · in-browser web demo

A static site for `portfolio.chafic.me` that runs the desktop app's own Python
analytics **in the visitor's browser** with [Pyodide](https://pyodide.org). The only
server piece is a small Cloudflare Worker that relays Yahoo Finance price data.

```
browser ─ index.html / app.js (UI, Plotly)            Cloudflare Pages (static, web/dist)
          └─ engine.worker.js (Web Worker)
               └─ Pyodide 0.29.3 + numpy, pandas, scipy, pydantic   (jsDelivr CDN)
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
* `web/worker/` – price proxy (`src/index.js`, `wrangler.toml`), unit tests, local
  preview server and the headless end-to-end test.
* `web/build.py` – builds `web/dist/` (stdlib only).

statsmodels is not needed: the demo's steps use numpy/scipy OLS; the statsmodels-based
factor and scenario models are desktop-only.

## Develop and test

```
python web/build.py                       # -> web/dist
node web/worker/dev-server.mjs            # http://localhost:8788, /v1 runs the Worker in-process
# or against the real Workers runtime:
#   (cd web/worker && npx wrangler dev --port 8787) &
#   UPSTREAM=http://127.0.0.1:8787 node web/worker/dev-server.mjs

pytest tests/test_web_demo.py             # engine + fetcher adapter (CPython, offline)
node --test web/worker/test/              # Worker unit tests (Node >= 18, no deps)
BASE=http://localhost:8788 OUT=/tmp/e2e node web/worker/test/e2e.mjs   # needs `npm i playwright`
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
