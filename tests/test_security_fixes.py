"""Security-hardening regressions (Qt-free).

Covers the audit fixes: symbol charset validation in PortfolioConfig, the
inline-<script> JSON escaper, and HTML-report autoescape being enabled.
"""

from datetime import date

import pytest

from src.config.models import PortfolioConfig
from src.ui.formatting import js_embed


def _cfg(**kw):
    base = dict(
        tickers=["AAPL"],
        weights={"AAPL": 1.0},
        start_date=date(2020, 1, 1),
        end_date=date(2021, 1, 1),
    )
    base.update(kw)
    return PortfolioConfig(**base)


# ── Symbol charset validation ──────────────────────────────────────
def test_real_world_symbols_are_accepted():
    syms = ["BTC-USD", "^GSPC", "ES=F", "EURUSD=X", "BRK-B", "BF.B", "0700.HK"]
    cfg = _cfg(tickers=syms, weights={"BTC-USD": 1.0})
    assert cfg.tickers == syms  # all preserved, uppercased


def test_injection_ticker_is_rejected():
    with pytest.raises(Exception):  # pydantic ValidationError wrapping ValueError
        _cfg(
            tickers=["<img src=x onerror=alert(1)>"],
            weights={"<img src=x onerror=alert(1)>": 1.0},
        )


def test_injection_in_weight_key_is_rejected():
    with pytest.raises(Exception):
        _cfg(tickers=["AAPL"], weights={"</script>": 1.0})


def test_injection_in_cost_basis_key_is_rejected():
    with pytest.raises(Exception):
        _cfg(cost_basis={"A<b>": 100.0})


def test_benchmark_label_with_slash_is_allowed():
    # With a blended benchmark, ``benchmark`` is a free-text label (e.g. "60/40")
    # and is intentionally NOT charset-validated — output escaping covers it.
    cfg = _cfg(benchmark="60/40", benchmark_weights={"SPY": 0.6, "AGG": 0.4})
    assert cfg.benchmark == "60/40"


# ── Inline-<script> JSON escaper ───────────────────────────────────
def test_js_embed_neutralizes_script_breakout():
    payload = '{"label": "</script><img src=x onerror=alert(1)>"}'
    out = js_embed(payload)
    assert "</script>" not in out
    assert "<" not in out and ">" not in out
    assert "\\u003c/script\\u003e" in out


def test_js_embed_leaves_safe_json_intact():
    assert js_embed('{"a": 1, "b": [2, 3]}') == '{"a": 1, "b": [2, 3]}'


# ── HTML report autoescape ─────────────────────────────────────────
def test_html_report_template_autoescapes():
    from src.reports.html_builder import _HTML_TEMPLATE

    assert _HTML_TEMPLATE.environment.autoescape is True
