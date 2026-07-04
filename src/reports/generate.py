"""Headless report generation — the engine behind automated / scheduled reports.

Qt-free: runs the analysis pipeline for a config and writes HTML/PDF reports to a
folder. Reused by the Data tab (interactive), the in-app scheduler, and the CLI.
"""

from __future__ import annotations

import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from src.charts import plotly_charts as charts
from src.reports.html_builder import build_html_report
from src.reports.pdf_builder import build_pdf_report


def report_figures(results) -> dict:
    """Chart set embedded in reports (Qt-free; mirrors the Data tab)."""
    figs: dict = {}
    try:
        growth = {}
        for k, ps in (("Active", results.active), ("Passive", results.passive),
                      ("ORP", results.orp)):
            if ps is not None:
                growth[k] = ps.values
        if growth:
            figs["growth"] = charts.growth_chart(growth, results.config.capital)

        dd = {}
        for k, ps in (("Active", results.active), ("Passive", results.passive)):
            if ps is not None:
                dd[k] = ps.values
        if dd:
            figs["drawdown"] = charts.drawdown_chart(dd)

        if results.correlation_matrix is not None:
            figs["correlation"] = charts.correlation_heatmap(results.correlation_matrix)

        orp = results.orp_optimization
        if orp:
            rets_m = results.monthly_returns
            cols = [t for t in results.config.tickers
                    if t in rets_m.columns and t != results.config.benchmark]
            mu = ((1 + rets_m[cols].mean()) ** 12 - 1) if cols else None
            vol = (rets_m[cols].std() * np.sqrt(12)) if cols else None
            figs["frontier"] = charts.efficient_frontier_chart(
                orp.frontier_vols, orp.frontier_returns, orp.expected_vol,
                orp.expected_return, results.config.risk_free_rate,
                asset_vols=vol, asset_returns=mu,
            )

        # Active portfolio allocation — holdings scaled by the risky fraction,
        # plus a Cash slice when a cash balance is held (cash as a holding).
        cap = float(results.config.capital or 0.0)
        cash = float(getattr(results.config, "cash", 0.0) or 0.0)
        alloc = {t: float(w) for t, w in (results.config.weights or {}).items()}
        if cash > 0 and cap > 0:
            y = max(cap - cash, 0.0) / cap
            alloc = {t: w * y for t, w in alloc.items()}
            alloc["Cash"] = cash / cap
        if alloc:
            figs["active_allocation"] = charts.allocation_donut(
                alloc, "Active Portfolio Allocation"
            )
    except Exception:
        pass
    return figs


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", (name or "portfolio").strip()) or "portfolio"


def generate_report(
    config,
    out_dir,
    formats=("pdf",),
    name: Optional[str] = None,
    progress: Optional[Callable[[str, float], None]] = None,
    stamp: Optional[str] = None,
) -> list[Path]:
    """Run the pipeline for ``config`` and write the requested report formats.

    Returns the list of written file paths. ``formats`` may include ``"html"`` /
    ``"pdf"``. Filenames are ``<name>_<stamp>.<ext>``.
    """
    from src.pipeline import AnalysisPipeline

    results = AnalysisPipeline(config, output_dir=tempfile.mkdtemp()).run(progress=progress)
    figs = report_figures(results)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    base = _safe(name)
    stamp = stamp or datetime.now().strftime("%Y%m%d_%H%M")

    written: list[Path] = []
    if "html" in formats:
        p = out / f"{base}_{stamp}.html"
        p.write_text(build_html_report(results, chart_figures=figs), encoding="utf-8")
        written.append(p)
    if "pdf" in formats:
        p = out / f"{base}_{stamp}.pdf"
        p.write_bytes(build_pdf_report(results, chart_figures=figs))
        written.append(p)
    return written
