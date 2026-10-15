"""In-browser web demo engine (web/py/pa_web), exercised in CPython.

The proxy transport is faked with synthetic prices, so the real adapter, the yfinance
stub and the shared pipeline all run exactly as they do under Pyodide, minus the XHR.
"""

from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "web" / "py"))

from pa_web import engine, yf_stub  # noqa: E402
from pa_web.fetcher import ProxyClient, parse_series  # noqa: E402
from pa_web.schemas import MAX_TICKERS, AnalyzeRequest  # noqa: E402
from tests.conftest import _synthetic_prices  # noqa: E402

SECTORS = {"AAA": "Technology", "BBB": "Healthcare"}


class FakeProxy:
    """Answers /v1/chart and /v1/profile like the Worker, from synthetic prices."""

    def __init__(self, unknown=()):
        self.urls: list[str] = []
        self.unknown = set(unknown)

    def __call__(self, url: str) -> str:
        self.urls.append(url)
        u = urlparse(url)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        syms = q["symbols"].split(",")
        if u.path == "/v1/profile":
            return json.dumps({"profiles": {s: {"sector": SECTORS.get(s), "quoteType": "EQUITY"} for s in syms}})
        assert u.path == "/v1/chart"
        good = [s for s in syms if s not in self.unknown]
        px = _synthetic_prices(good, q["start"], q["end"]) if good else pd.DataFrame()
        data = {s: {"d": [d.strftime("%Y-%m-%d") for d in px.index], "c": [round(float(v), 6) for v in px[s]]}
                for s in good}
        return json.dumps({"data": data, "errors": {s: "Unknown symbol" for s in syms if s in self.unknown}})


@pytest.fixture
def proxy(monkeypatch):
    fake = FakeProxy(unknown={"ZZZZ"})
    import src.analytics.sector_attribution as SA
    import src.data.fetcher as fetcher_mod
    import src.pipeline as pipeline
    import src.data.meta_cache as meta

    # engine.install() patches these module globals; restore them after each test.
    monkeypatch.setattr(pipeline, "fetch_prices", pipeline.fetch_prices)
    monkeypatch.setattr(fetcher_mod, "fetch_prices", fetcher_mod.fetch_prices)
    monkeypatch.setitem(sys.modules, "yfinance", sys.modules.get("yfinance"))
    monkeypatch.setattr(meta, "_path", lambda cache_dir: Path("/nonexistent-dir-for-tests/meta.json"))
    assert SA  # sector snapshot fallback is used via the stub's funds_data error
    client = engine.install("https://proxy.test", transport=fake)
    return fake, client


def body(**over):
    b = {"holdings": [{"ticker": "AAA", "weight": 60}, {"ticker": "BBB", "weight": 40}],
         "benchmark": "SPY", "start_date": "2021-01-01", "end_date": "2024-12-31"}
    b.update(over)
    return b


# ── fetcher adapter ────────────────────────────────────────────────────────


def test_parse_series_drops_nulls_and_duplicates():
    s = parse_series({"d": ["2024-01-02", "2024-01-03", "2024-01-03", "2024-01-04"], "c": [1.0, None, 2.0, 3.0]})
    assert list(s.index.strftime("%Y-%m-%d")) == ["2024-01-02", "2024-01-03", "2024-01-04"]
    assert list(s) == [1.0, 2.0, 3.0]
    assert s.dtype == float and s.isna().sum() == 0
    assert parse_series({"d": ["2024-01-02"], "c": []}) is None
    assert parse_series({}) is None


def test_fetch_prices_frame_shape_and_memo():
    fake = FakeProxy()
    c = ProxyClient("https://p.test/", transport=fake)
    df = c.fetch_prices(["AAA", "BBB"], date(2023, 1, 1), date(2023, 6, 30))
    assert list(df.columns) == ["AAA", "BBB"] and df.index.name == "Date"
    assert df.index.min() >= pd.Timestamp("2023-01-01") and df.index.max() <= pd.Timestamp("2023-06-30")
    assert fake.urls[0].startswith("https://p.test/v1/chart?")
    c.fetch_prices(["BBB", "AAA"], date(2023, 1, 1), date(2023, 6, 30))
    assert len(fake.urls) == 1  # served from the in-session memo


def test_fetch_prices_chunks_large_requests():
    fake = FakeProxy()
    c = ProxyClient("", transport=fake)
    c.fetch_prices([f"T{i}" for i in range(30)], date(2023, 1, 1), date(2023, 3, 1))
    assert len(fake.urls) == 2


def test_fetch_prices_missing_and_all_missing():
    c = ProxyClient("", transport=FakeProxy(unknown={"ZZZZ"}))
    df = c.fetch_prices(["AAA", "ZZZZ"], date(2023, 1, 1), date(2023, 3, 1))
    assert list(df.columns) == ["AAA"] and df.attrs["missing"] == ["ZZZZ"]
    with pytest.raises(ValueError, match="Unknown symbol"):
        c.fetch_prices(["ZZZZ"], date(2023, 1, 1), date(2023, 3, 1))


def test_proxy_error_raises():
    c = ProxyClient("", transport=lambda url: json.dumps({"error": "Rate limit exceeded"}))
    with pytest.raises(ValueError, match="Rate limit"):
        c.fetch_prices(["AAA"], date(2023, 1, 1), date(2023, 3, 1))


def test_yfinance_stub_answers_info_and_refuses_the_rest(proxy):
    import yfinance
    assert yfinance is yf_stub
    assert yfinance.Ticker("AAA").info["sector"] == "Technology"
    with pytest.raises(Exception):
        yfinance.Ticker("SPY").funds_data
    with pytest.raises(Exception):
        yfinance.download("AAA")


# ── validation ─────────────────────────────────────────────────────────────


def test_weights_normalized_and_cache_key_stable():
    a = AnalyzeRequest(**body())
    b = AnalyzeRequest(**body(holdings=[{"ticker": "bbb", "weight": 4}, {"ticker": "aaa", "weight": 6}]))
    assert a.normalized_weights() == pytest.approx({"AAA": 0.6, "BBB": 0.4})
    assert a.cache_key() == b.cache_key()


@pytest.mark.parametrize("over, msg", [
    ({"holdings": [{"ticker": f"T{i}", "weight": 1} for i in range(MAX_TICKERS + 1)]}, "holdings"),
    ({"start_date": "2005-01-01", "end_date": "2024-12-31"}, "15 years"),
    ({"start_date": "2024-06-01", "end_date": "2024-08-01"}, "6 months"),
    ({"holdings": [{"ticker": "AAA", "weight": 50}, {"ticker": "AAA", "weight": 50}]}, "once"),
    ({"holdings": [{"ticker": "<script>", "weight": 100}]}, "Invalid symbol"),
    ({"holdings": [{"ticker": "AAA", "weight": -5}]}, "greater than 0"),
    ({"benchmark": "SPY;DROP"}, "Invalid symbol"),
])
def test_engine_rejects_bad_input(proxy, over, msg):
    out = json.loads(engine.analyze(json.dumps(body(**over))))
    assert out["kind"] == "input" and msg in out["error"]


def test_future_end_date_is_clamped_to_today():
    r = AnalyzeRequest(**body(start_date=str(date.today() - timedelta(days=400)),
                              end_date=str(date.today() + timedelta(days=30))))
    assert r.end_date == date.today()


# ── end to end through the adapter ────────────────────────────────────────


def test_analyze_payload(proxy):
    fake, _ = proxy
    d = json.loads(engine.analyze(json.dumps(body())))
    assert "error" not in d, d
    assert d["cached"] is False
    k = d["kpis"]
    for key in ("nav", "cagr", "vol", "sharpe", "max_dd", "beta", "tracking_error"):
        assert k[key] is not None, key
    g = d["growth"]["portfolio"]
    assert len(g["x"]) == len(g["y"]) > 100
    assert g["y"][0] == pytest.approx(100_000, rel=0.02)
    assert min(d["drawdown"]["portfolio"]["y"]) == pytest.approx(k["max_dd"], abs=1e-6)
    assert {h["Ticker"] for h in d["holdings"]} == {"AAA", "BBB", "Portfolio"}
    assert d["risk"]["var"] and d["risk"]["stress"] and d["risk"]["drawdowns"]
    assert d["attribution"]["sector_map"] == {"AAA": "Technology", "BBB": "Healthcare"}
    assert "snapshot" in d["attribution"]["source"].lower()
    assert d["meta"]["risk_free"]["source"]  # ^IRX came through the proxy too
    paths = {urlparse(u).path for u in fake.urls}
    assert paths == {"/v1/chart", "/v1/profile"}
    assert any("%5EIRX" in u for u in fake.urls) and any("XLK" in u for u in fake.urls)
    json.loads(json.dumps(d), parse_constant=lambda c: pytest.fail(f"non-finite {c}"))


def test_repeat_run_served_from_cache(proxy):
    fake, _ = proxy
    engine.analyze(json.dumps(body()))
    n = len(fake.urls)
    again = json.loads(engine.analyze(json.dumps(body(holdings=[{"ticker": "BBB", "weight": 2}, {"ticker": "AAA", "weight": 3}]))))
    assert again["cached"] is True and len(fake.urls) == n


def test_unknown_ticker_is_a_clean_error(proxy):
    out = json.loads(engine.analyze(json.dumps(body(holdings=[{"ticker": "ZZZZ", "weight": 1}]))))
    assert out["kind"] == "data" and "Could not build" in out["error"]


def test_partial_unknown_ticker_is_reported(proxy):
    d = json.loads(engine.analyze(json.dumps(body(holdings=[{"ticker": "AAA", "weight": 1}, {"ticker": "ZZZZ", "weight": 1}]))))
    assert "error" not in d
    assert "ZZZZ" in d["missing"] or any("ZZZZ" in w for w in d["warnings"])


def test_bundle_contains_only_needed_modules():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "web"))
    import build
    arcs = {arc for _, arc in build.bundle_files()}
    assert "src/pipeline.py" in arcs and "pa_web/engine.py" in arcs
    assert not any(a.startswith(("src/ui/", "src/reports/", "src/charts/")) for a in arcs)


def test_static_site_has_no_secrets():
    import re
    root = Path(__file__).resolve().parent.parent / "web"
    files = [p for d in ("static", "worker/src") for p in (root / d).rglob("*") if p.is_file()]
    files.append(root / "worker" / "wrangler.toml")
    text = "".join(p.read_text(encoding="utf-8") for p in files)
    assert not re.search(r"(api[_-]?key|secret|token|password)\s*[:=]", text, re.I)
    assert not re.search(r"^\s*account_id\s*=", text, re.M)
