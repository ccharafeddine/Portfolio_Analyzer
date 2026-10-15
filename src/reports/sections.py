"""Report-ready tables for the analytics added in Phase 3/4.

Each section is a plain dict so the HTML, PDF and PowerPoint builders render
the same content: ``{"id", "title", "note", "tables": [{"title", "columns",
"align", "rows"}]}`` where rows are already-formatted strings and ``align`` is
"l"/"r" per column.
"""

from __future__ import annotations

import math

import pandas as pd


def _pct(v, d=2, signed=False) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    s = f"{v * 100:+.{d}f}%" if signed else f"{v * 100:.{d}f}%"
    return s.replace("-", "−")


def _num(v, d=2) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v:,.{d}f}".replace("-", "−")


def _date(v) -> str:
    return "Open" if v is None or pd.isna(v) else pd.Timestamp(v).strftime("%d %b %Y")


def _money(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"−${-v:,.0f}" if v < 0 else f"${v:,.0f}"


def _table(title, columns, align, rows, signed=False):
    """``signed``: colour +/− values green/red (only where sign carries meaning,
    e.g. active returns; never for VaR levels or factor loadings)."""
    return {"title": title, "columns": columns, "align": align, "rows": rows, "signed": signed}


def sign_class(value: str, signed: bool) -> str:
    """'pos' / 'neg' / '' for a formatted cell; zeros stay neutral."""
    if not signed or not value or value.strip("+−-0.%$") == "":
        return ""
    return "pos" if value.startswith("+") else "neg" if value.startswith(("−", "-")) else ""


def data_quality_section(results):
    rep = getattr(results, "data_quality", None)
    if rep is None or rep.table.empty:
        return None
    rows = [[r.Ticker, _date(r.First), _date(r.Last), str(int(r.Missing)), str(int(r.Gaps)),
             str(int(r.Splits)), r.Status] for r in rep.table.itertuples()]
    note = ("No data issues found." if not rep.issues
            else "Issues: " + "; ".join(rep.issues[:6]) + ("…" if len(rep.issues) > 6 else ""))
    return {"id": "dataquality", "title": "Data Quality", "note": note, "tables": [
        _table("", ["Ticker", "First", "Last", "Missing", "Gaps", "Split?", "Status"],
               "lllrrrl", rows)]}


def var_section(results):
    vt = getattr(results, "var_table", None)
    if vt is None or vt.empty:
        return None
    tables = [_table("Value at Risk and Expected Shortfall (1-day)",
                     ["Method", "VaR 95%", "ES 95%", "VaR 99%", "ES 99%"], "lrrrr",
                     [[r["Method"]] + [_pct(r[c]) for c in ("VaR 95%", "ES 95%", "VaR 99%", "ES 99%")]
                      for _, r in vt.iterrows()])]
    bt = getattr(results, "var_backtest", None)
    if bt is not None and not bt.empty:
        tables.append(_table("Backtest: 99% VaR, 250-day rolling window",
                             ["Method", "Days", "Breaches", "Expected", "Rate", "Kupiec p", "Verdict"],
                             "lrrrrrl",
                             [[r["Method"], f"{int(r['Days']):,}", str(int(r["Breaches"])),
                               _num(r["Expected"], 1), _pct(r["Breach rate"]), _num(r["Kupiec p"], 3),
                               r["Verdict"]] for _, r in bt.iterrows()]))
    return {"id": "var", "title": "Value at Risk", "tables": tables,
            "note": "Historical, normal, Cornish-Fisher and Student-t Monte Carlo estimates of "
                    "one-day loss. The backtest forecasts each day from the prior 250 only; a "
                    "Kupiec p-value below 0.05 rejects the model."}


def drawdown_section(results):
    dd = getattr(results, "drawdown_table", None)
    if dd is None or dd.empty:
        return None
    rows = [[_date(r["Peak"]), _date(r["Trough"]), _date(r["Recovery"]), _pct(r["Depth"], 1),
             str(int(r["Decline days"])),
             "—" if pd.isna(r["Recovery days"]) else str(int(r["Recovery days"])),
             str(int(r["Length"]))] for _, r in dd.iterrows()]
    return {"id": "drawdowns", "title": "Largest Drawdowns",
            "note": "Lengths are in trading days. 'Open' means the prior peak has not been regained.",
            "tables": [_table("", ["Peak", "Trough", "Recovery", "Depth", "Decline", "Recover",
                                   "Length"], "lllrrrr", rows)]}


def stress_section(results):
    srs = getattr(results, "stress_results", None) or []
    if not srs:
        return None
    rows = []
    for r in srs:
        ok = r.data_available
        rows.append([r.name, f"{r.start} → {r.end}",
                     _pct(r.portfolio_return, 1, True) if ok else "N/A",
                     _pct(r.benchmark_return, 1, True) if ok else "N/A",
                     _pct(r.max_drawdown, 1) if ok else "N/A",
                     r.basis if ok else "No data"])
    tables = [_table("Historical scenarios", ["Scenario", "Period", "Portfolio", "Benchmark",
                                              "Max DD", "Basis"], "llrrrl", rows, signed=True)]
    bs = getattr(results, "beta_shocks", None)
    if bs is not None and not bs.empty:
        tables.append(_table(f"Market shocks via beta ({bs['Beta'].iloc[0]:.2f})",
                             ["Market move", "Portfolio", "P&L"], "rrr",
                             [[_pct(r["Market shock"], 0, True), _pct(r["Portfolio"], 1, True),
                               _money(r["P&L"])] for _, r in bs.iterrows()], signed=True))
    return {"id": "stress", "title": "Stress Tests", "tables": tables,
            "note": "Replay rows apply today's weights to each holding's own price history "
                    "for scenarios before the analysis window."}


def sector_section(results):
    sa = getattr(results, "sector_attribution", None)
    if sa is None or sa.empty:
        return None
    meta = getattr(results, "sector_attribution_meta", None) or {}
    rows = [[r["Sector"], _pct(r["Port weight"], 1), _pct(r["Bench weight"], 1),
             _pct(r["Allocation"], 2, True), _pct(r["Selection"], 2, True),
             _pct(r["Interaction"], 2, True), _pct(r["Total"], 2, True)] for _, r in sa.iterrows()]
    tot = sa[["Allocation", "Selection", "Interaction", "Total"]].sum()
    rows.append(["Total", "100.0%", "100.0%"] + [_pct(tot[c], 2, True) for c in tot.index])
    return {"id": "sectors", "title": "Sector Attribution (Brinson-Fachler)",
            "note": (f"Portfolio {_pct(meta.get('port_return'), 1, True)} vs sector benchmark "
                     f"{_pct(meta.get('bench_return'), 1, True)}; monthly, Carino-linked. "
                     f"Benchmark weights: {meta.get('source', 'n/a')}; sector returns from SPDR "
                     "sector ETFs."),
            "tables": [_table("", ["Sector", "Port. wt", "Bench wt", "Allocation", "Selection",
                                   "Interaction", "Total"], "lrrrrrr", rows, signed=True)]}


def factor_section(results):
    fm = getattr(results, "factor_models", None) or {}
    df = fm.get("FF3") if "FF3" in fm else (next(iter(fm.values())) if fm else None)
    if df is None or df.empty:
        return None
    factors = [c[:-5] for c in df.columns if c.endswith("_coef") and c != "const_coef"]
    rows = []
    for _, r in df.iterrows():
        row = [r["Asset"], f"{_pct((1 + r['const_coef']) ** 252 - 1, 1, True)} ({_num(r.get('const_t', math.nan), 1)})"]
        row += [f"{_num(r[f + '_coef'])} ({_num(r.get(f + '_t', math.nan), 1)})" for f in factors]
        row.append(_num(r["R2"]))
        rows.append(row)
    return {"id": "factors", "title": "Factor Exposures (Fama-French 3)",
            "note": "Daily regressions on Ken French factors; t-statistics in parentheses. "
                    "|t| above 2 is conventionally significant.",
            "tables": [_table("", ["Asset", "Alpha ann. (t)"] + [f"{f} (t)" for f in factors] + ["R²"],
                              "l" + "r" * (len(factors) + 2), rows)]}


def analytics_sections(results) -> list[dict]:
    out = []
    for fn in (var_section, drawdown_section, stress_section, sector_section, factor_section,
               data_quality_section):
        try:
            sec = fn(results)
        except Exception:  # a report must never fail on one section
            sec = None
        if sec:
            out.append(sec)
    return out
