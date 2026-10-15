"""Entry points called from the Web Worker (engine.worker.js).

    install(proxy_base)  -> wires the proxy fetcher into the shared pipeline
    analyze(json_text)   -> validates the request, runs it, returns JSON text
"""

from __future__ import annotations

import json
from collections import OrderedDict

from pydantic import ValidationError

from . import yf_stub
from .fetcher import ProxyClient

_client: ProxyClient | None = None
_results: "OrderedDict[str, str]" = OrderedDict()
MAX_CACHED_RESULTS = 16


def install(proxy_base: str = "", transport=None, lite_stats: bool | None = None) -> ProxyClient:
    """Route all price and metadata requests of the shared code through the proxy.

    ``lite_stats`` (default: only when scipy is not installed, i.e. in the browser)
    provides ``scipy.stats`` from pa_web.stats_lite so scipy is never loaded."""
    global _client
    if lite_stats is None:
        import importlib.util
        lite_stats = importlib.util.find_spec("scipy") is None
    if lite_stats:
        from . import stats_lite
        stats_lite.install()
    _client = ProxyClient(proxy_base, transport=transport)
    yf_stub.install(_client)  # before src.data.fetcher imports ``yfinance``
    import src.data.fetcher as fetcher_mod
    import src.pipeline as pipeline

    fetcher_mod.fetch_prices = _client.fetch_prices  # used lazily by risk_free
    pipeline.fetch_prices = _client.fetch_prices
    _results.clear()
    return _client


def _error(msg: str, kind: str = "input") -> str:
    return json.dumps({"error": msg, "kind": kind})


def analyze(payload: str) -> str:
    from .schemas import AnalyzeRequest
    from .service import AnalysisError, run_analysis

    if _client is None:
        install()
    try:
        req = AnalyzeRequest.model_validate_json(payload)
    except ValidationError as e:
        msgs = []
        for err in e.errors()[:4]:
            loc = ".".join(str(p) for p in err.get("loc", []))
            msg = str(err.get("msg", "Invalid input")).removeprefix("Value error, ")
            msgs.append(f"{loc}: {msg}" if loc else msg)
        return _error("; ".join(msgs))
    key = req.cache_key()
    if key in _results:
        _results.move_to_end(key)
        out = json.loads(_results[key])
        out["cached"] = True
        return json.dumps(out)
    try:
        result = run_analysis(req)
    except AnalysisError as e:
        return _error(str(e), "data")
    text = json.dumps({**result, "cached": False}, allow_nan=False)
    _results[key] = text
    while len(_results) > MAX_CACHED_RESULTS:
        _results.popitem(last=False)
    return text
