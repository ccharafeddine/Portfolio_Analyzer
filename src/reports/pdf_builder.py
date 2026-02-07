"""
PDF report builder for Portfolio Analyzer v2.

Generates a functional PDF report using reportlab with:
- Cover page
- Table of contents
- Embedded chart images (via kaleido, optional)
- Data tables
- Interpretation text from the interpreter engine
"""

from __future__ import annotations

import io
from datetime import datetime
from typing import TYPE_CHECKING, Optional

import numpy as np

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak,
    KeepTogether,
)

if TYPE_CHECKING:
    from src.pipeline import AnalysisResults

from src.reports.interpreter import generate_full_interpretation


# ---------------------------------------------------------------
# Constants
# ---------------------------------------------------------------

NAVY = colors.HexColor("#0B1120")
BLUE = colors.HexColor("#3B82F6")
DARK_SLATE = colors.HexColor("#1E293B")
LIGHT_BG = colors.HexColor("#F8FAFC")
TEXT_COLOR = colors.HexColor("#334155")
MUTED = colors.HexColor("#64748B")
WHITE = colors.white


# ---------------------------------------------------------------
# Styles
# ---------------------------------------------------------------

def _build_styles():
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        "CoverTitle",
        parent=styles["Title"],
        fontSize=28,
        textColor=WHITE,
        spaceAfter=12,
        leading=34,
    ))
    styles.add(ParagraphStyle(
        "CoverSub",
        parent=styles["Normal"],
        fontSize=12,
        textColor=colors.HexColor("#94A3B8"),
        spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        "SectionTitle",
        parent=styles["Heading1"],
        fontSize=18,
        textColor=NAVY,
        spaceBefore=24,
        spaceAfter=8,
        borderWidth=0,
        borderPadding=0,
    ))
    styles.add(ParagraphStyle(
        "SubTitle",
        parent=styles["Heading2"],
        fontSize=13,
        textColor=DARK_SLATE,
        spaceBefore=16,
        spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        "InterpText",
        parent=styles["Normal"],
        fontSize=10,
        textColor=TEXT_COLOR,
        leading=15,
        spaceBefore=4,
        spaceAfter=12,
        leftIndent=8,
        borderWidth=0,
    ))
    styles.add(ParagraphStyle(
        "BodyText2",
        parent=styles["Normal"],
        fontSize=10,
        textColor=TEXT_COLOR,
        leading=14,
        spaceAfter=8,
    ))

    return styles


# ---------------------------------------------------------------
# Chart image helper
# ---------------------------------------------------------------

def _fig_to_image(fig, width=6.5 * inch, height=3.2 * inch) -> Optional[Image]:
    """Convert a Plotly figure to a reportlab Image. Returns None if kaleido unavailable."""
    try:
        img_bytes = fig.to_image(format="png", width=900, height=440, scale=2)
        buf = io.BytesIO(img_bytes)
        return Image(buf, width=width, height=height)
    except Exception:
        return None


# ---------------------------------------------------------------
# Table builder
# ---------------------------------------------------------------

def _make_table(headers: list[str], rows: list[list[str]], col_widths=None) -> Table:
    """Build a styled reportlab Table."""
    data = [headers] + rows

    if col_widths is None:
        n = len(headers)
        available = 7.0 * inch
        col_widths = [available / n] * n

    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 8),
        ("FONTSIZE", (0, 1), (-1, -1), 8),
        ("TEXTCOLOR", (0, 1), (-1, -1), TEXT_COLOR),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT_BG]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    return t


# ---------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------

def _pct(v, decimals=2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    return f"{v * 100:.{decimals}f}%"


def _dollar(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    if abs(v) >= 1e6:
        return f"${v / 1e6:,.2f}M"
    return f"${v:,.0f}"


# ---------------------------------------------------------------
# PDF builder
# ---------------------------------------------------------------

def build_pdf_report(
    results: AnalysisResults,
    chart_figures: Optional[dict] = None,
) -> bytes:
    """
    Build a PDF report from AnalysisResults.

    Parameters
    ----------
    results : AnalysisResults from pipeline
    chart_figures : optional dict of chart_name -> plotly Figure

    Returns
    -------
    PDF bytes
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=letter,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
    )

    styles = _build_styles()
    interp = generate_full_interpretation(results)
    chart_figures = chart_figures or {}
    elements = []

    # ── Cover page ──
    elements.append(Spacer(1, 2 * inch))
    elements.append(Paragraph("Portfolio Analysis Report", styles["CoverTitle"]))
    elements.append(Paragraph(
        ", ".join(results.config.tickers),
        styles["CoverSub"],
    ))
    elements.append(Spacer(1, 24))
    meta_lines = [
        f"Benchmark: {results.config.benchmark}",
        f"Period: {results.config.start_date} to {results.config.end_date}",
        f"Capital: ${results.config.capital:,.0f}",
        f"Risk-Free Rate: {_pct(results.config.risk_free_rate)}",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
    ]
    for line in meta_lines:
        elements.append(Paragraph(line, styles["CoverSub"]))
    elements.append(PageBreak())

    # ── Table of Contents ──
    elements.append(Paragraph("Table of Contents", styles["SectionTitle"]))
    toc_items = [
        "1. Executive Summary",
        "2. Performance Analysis",
        "3. Risk Analysis",
        "4. Drawdown Analysis",
        "5. CAPM Regression",
        "6. Portfolio Optimization",
        "7. Stress Testing",
        "8. Income Analysis",
        "9. Correlation & Diversification",
        "10. Monte Carlo Simulation",
        "11. Holdings Detail",
    ]
    for item in toc_items:
        elements.append(Paragraph(item, styles["BodyText2"]))
    elements.append(PageBreak())

    # ── 1. Executive Summary ──
    elements.append(Paragraph("1. Executive Summary", styles["SectionTitle"]))
    elements.append(Paragraph(interp["executive_summary"], styles["InterpText"]))

    # Summary metrics as a mini table
    if results.active:
        metrics_data = [
            ["Total Return", _pct(results.active.ann_return)],
            ["Sharpe Ratio", f"{results.active.sharpe:.2f}" if not np.isnan(results.active.sharpe) else "N/A"],
            ["Max Drawdown", _pct(results.active.max_dd)],
            ["Volatility", _pct(results.active.ann_vol)],
        ]
        if results.capm_results:
            avg_alpha = float(np.mean([r.alpha for r in results.capm_results]))
            avg_beta = float(np.mean([r.beta for r in results.capm_results]))
            metrics_data.append(["Alpha (ann.)", f"{avg_alpha * 12:.2%}"])
            metrics_data.append(["Beta", f"{avg_beta:.2f}"])

        t = _make_table(
            ["Metric", "Value"],
            metrics_data,
            col_widths=[3.5 * inch, 3.5 * inch],
        )
        elements.append(t)
        elements.append(Spacer(1, 12))

    # ── 2. Performance ──
    elements.append(Paragraph("2. Performance Analysis", styles["SectionTitle"]))
    elements.append(Paragraph(interp["performance"], styles["InterpText"]))

    perf_rows = []
    for ps in [results.active, results.passive, results.orp, results.hrp, results.rebalanced, results.complete]:
        if ps is None:
            continue
        perf_rows.append([
            ps.name,
            _pct(ps.ann_return),
            _pct(ps.ann_vol),
            f"{ps.sharpe:.2f}" if not np.isnan(ps.sharpe) else "N/A",
            _pct(ps.max_dd),
        ])
    if perf_rows:
        t = _make_table(
            ["Portfolio", "Ann. Return", "Ann. Vol", "Sharpe", "Max DD"],
            perf_rows,
        )
        elements.append(t)

    # Growth chart
    if "growth" in chart_figures:
        img = _fig_to_image(chart_figures["growth"])
        if img:
            elements.append(Spacer(1, 8))
            elements.append(img)

    # ── 3. Risk ──
    elements.append(Paragraph("3. Risk Analysis", styles["SectionTitle"]))
    elements.append(Paragraph(interp["risk"], styles["InterpText"]))

    if results.drawdown_metrics is not None:
        risk_rows = []
        for _, r in results.drawdown_metrics.iterrows():
            risk_rows.append([
                r["Portfolio"],
                _pct(r["MaxDrawdown"]),
                _pct(r["VaR_95"]),
                _pct(r["CVaR_95"]),
                _pct(r["VaR_99"]),
                _pct(r["CVaR_99"]),
            ])
        t = _make_table(
            ["Portfolio", "Max DD", "VaR 95%", "CVaR 95%", "VaR 99%", "CVaR 99%"],
            risk_rows,
        )
        elements.append(t)

    # ── 4. Drawdown ──
    elements.append(Paragraph("4. Drawdown Analysis", styles["SectionTitle"]))
    elements.append(Paragraph(interp["drawdown"], styles["InterpText"]))

    if "drawdown" in chart_figures:
        img = _fig_to_image(chart_figures["drawdown"])
        if img:
            elements.append(img)

    # ── 5. CAPM ──
    elements.append(Paragraph("5. CAPM Regression", styles["SectionTitle"]))
    elements.append(Paragraph(interp["capm"], styles["InterpText"]))

    if results.capm_results:
        capm_rows = []
        for r in results.capm_results:
            capm_rows.append([
                r.ticker,
                f"{r.alpha:.4f}",
                f"{r.beta:.2f}",
                f"{r.t_alpha:.2f}",
                f"{r.t_beta:.2f}",
                f"{r.r_squared:.3f}",
            ])
        t = _make_table(
            ["Asset", "Alpha", "Beta", "t(a)", "t(b)", "R2"],
            capm_rows,
        )
        elements.append(t)

    # ── 6. Optimization ──
    elements.append(Paragraph("6. Portfolio Optimization", styles["SectionTitle"]))
    elements.append(Paragraph(interp["optimization"], styles["InterpText"]))

    if results.orp_optimization:
        headers = ["Asset", "ORP Weight"]
        if results.hrp_weights is not None:
            headers.append("HRP Weight")
        opt_rows = []
        for asset, w in results.orp_optimization.weights.items():
            row = [asset, _pct(float(w))]
            if results.hrp_weights is not None:
                hrp_val = float(results.hrp_weights.get(asset, 0))
                row.append(_pct(hrp_val))
            opt_rows.append(row)
        t = _make_table(headers, opt_rows)
        elements.append(t)

    if "frontier" in chart_figures:
        img = _fig_to_image(chart_figures["frontier"])
        if img:
            elements.append(Spacer(1, 8))
            elements.append(img)

    # ── 7. Stress Testing ──
    elements.append(Paragraph("7. Stress Testing", styles["SectionTitle"]))
    elements.append(Paragraph(interp["stress_tests"], styles["InterpText"]))

    if results.stress_df is not None and not results.stress_df.empty:
        stress_rows = []
        for _, r in results.stress_df.iterrows():
            stress_rows.append([
                str(r.get("Scenario", "")),
                str(r.get("Period", "")),
                str(r.get("Portfolio", "N/A")),
                str(r.get("Benchmark", "N/A")),
            ])
        t = _make_table(
            ["Scenario", "Period", "Portfolio", "Benchmark"],
            stress_rows,
        )
        elements.append(t)

    # ── 8. Income ──
    elements.append(Paragraph("8. Income Analysis", styles["SectionTitle"]))
    elements.append(Paragraph(interp["income"], styles["InterpText"]))

    if results.income_summary is not None and not results.income_summary.empty:
        inc_rows = []
        for _, r in results.income_summary.iterrows():
            inc_rows.append([
                r["Ticker"],
                f"{r['Shares']:.0f}",
                f"${r['AnnualDividendPerShare']:.4f}",
                f"${r['AnnualIncome']:,.2f}",
                _pct(r["YieldOnCost"]),
                _pct(r["CurrentYield"]),
            ])
        t = _make_table(
            ["Ticker", "Shares", "Div/Share", "Ann. Income", "YoC", "Cur. Yield"],
            inc_rows,
        )
        elements.append(t)

    # ── 9. Correlation ──
    elements.append(Paragraph("9. Correlation & Diversification", styles["SectionTitle"]))
    elements.append(Paragraph(interp["correlation"], styles["InterpText"]))

    if "correlation" in chart_figures:
        img = _fig_to_image(chart_figures["correlation"])
        if img:
            elements.append(img)

    # ── 10. Monte Carlo ──
    elements.append(Paragraph("10. Monte Carlo Simulation", styles["SectionTitle"]))
    elements.append(Paragraph(interp["simulation"], styles["InterpText"]))

    if results.simulations:
        sim_rows = []
        for sim in results.simulations:
            sim_rows.append([
                sim.name,
                _dollar(sim.expected_value),
                _pct(sim.prob_loss),
                _dollar(sim.percentiles["P5"]),
                _dollar(sim.percentiles["P50"]),
                _dollar(sim.percentiles["P95"]),
            ])
        t = _make_table(
            ["Method", "Expected", "P(Loss)", "P5", "P50", "P95"],
            sim_rows,
        )
        elements.append(t)

    # ── 11. Holdings ──
    elements.append(Paragraph("11. Holdings Detail", styles["SectionTitle"]))

    if results.holdings is not None:
        h_rows = []
        for _, r in results.holdings.iterrows():
            h_rows.append([
                r["Ticker"],
                _pct(r.get("TargetWeight", 0)),
                _pct(r.get("RealizedWeight", 0)),
                f"{r['Shares']:.0f}",
                f"${r['PurchasePrice']:.2f}",
                _dollar(r["Invested"]),
            ])
        t = _make_table(
            ["Ticker", "Target Wt", "Realized Wt", "Shares", "Price", "Invested"],
            h_rows,
        )
        elements.append(t)

    # Footer
    elements.append(Spacer(1, 24))
    elements.append(Paragraph(
        f"Portfolio Analyzer v2 | Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        ParagraphStyle("Footer", parent=styles["Normal"], fontSize=8, textColor=MUTED, alignment=1),
    ))

    doc.build(elements)
    return buf.getvalue()
