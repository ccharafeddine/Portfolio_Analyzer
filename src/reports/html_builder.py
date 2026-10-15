"""
HTML report builder for Portfolio Analyzer v2.

Generates a standalone HTML report using Jinja2 templates
with embedded CSS. Dark navy headers, clean typography,
print-friendly layout.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
from jinja2 import Template

if TYPE_CHECKING:
    from src.pipeline import AnalysisResults

from src.reports.interpreter import generate_full_interpretation


# ---------------------------------------------------------------
# Chart image export (optional kaleido)
# ---------------------------------------------------------------

def _export_chart_base64(fig) -> Optional[str]:
    """Export a Plotly figure to a base64-encoded PNG. Returns None if kaleido unavailable."""
    try:
        import base64
        img_bytes = fig.to_image(format="png", width=900, height=450, scale=2)
        return base64.b64encode(img_bytes).decode("utf-8")
    except Exception:
        return None


# ---------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------

_HTML_TEMPLATE = Template("""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta http-equiv="Content-Security-Policy" content="script-src 'none'; object-src 'none'; base-uri 'none'">
<title>Portfolio Analysis Report</title>
<style>
  /* Graphite, print edition: neutral greys, hairlines, one blue accent,
     tabular numerals; green/red reserved for signed values. */
  * { margin: 0; padding: 0; box-sizing: border-box; }
  :root { --ink:#15191E; --ink2:#3A4250; --muted:#6B7480; --line:#E3E6EA;
          --soft:#F5F6F8; --accent:#2F6FEB; --green:#1A7F4B; --red:#C93C37; }
  body { font-family: 'Inter','Segoe UI','Helvetica Neue',Arial,sans-serif;
         color: var(--ink); background:#fff; line-height:1.55; font-size:13px;
         font-variant-numeric: tabular-nums; }
  .cover { border-bottom: 1px solid var(--line); padding: 56px 60px 36px; margin-bottom: 28px; }
  .cover .eyebrow { font-size: 11px; letter-spacing:.08em; text-transform:uppercase;
                    color: var(--muted); margin-bottom: 10px; }
  .cover h1 { font-size: 28px; font-weight: 600; letter-spacing:-.01em; margin-bottom: 6px; }
  .cover .subtitle { font-size: 15px; color: var(--ink2); margin-bottom: 18px; }
  .cover .meta { font-size: 12px; color: var(--muted); line-height: 1.8; }
  .container { max-width: 920px; margin: 0 auto; padding: 0 40px; }
  h2 { font-size: 17px; font-weight: 600; color: var(--ink); margin: 40px 0 10px;
       padding-bottom: 6px; border-bottom: 1px solid var(--ink); }
  h3 { font-size: 11px; font-weight: 600; color: var(--muted); text-transform: uppercase;
       letter-spacing: .06em; margin: 22px 0 6px; }
  p, .interpretation { margin-bottom: 14px; color: var(--ink2); }
  .interpretation { border-left: 2px solid var(--accent); padding: 4px 0 4px 14px;
                    font-size: 12.5px; line-height: 1.65; }
  .note { color: var(--muted); font-size: 11.5px; margin: 4px 0 10px; }
  table { width: 100%; border-collapse: collapse; margin: 6px 0 18px; font-size: 12px; }
  th { color: var(--muted); font-weight: 600; font-size: 10.5px; text-transform: uppercase;
       letter-spacing: .05em; text-align: left; padding: 6px 10px;
       border-bottom: 1px solid var(--ink); }
  td { padding: 5px 10px; border-bottom: 1px solid var(--line); color: var(--ink); }
  th.r, td.r { text-align: right; }
  td.pos { color: var(--green); } td.neg { color: var(--red); }
  tr.tot td { font-weight: 600; border-top: 1px solid var(--ink); }
  .metrics-grid { display: grid; grid-template-columns: repeat(4, 1fr); margin: 14px 0 18px;
                  border-top: 1px solid var(--line); border-left: 1px solid var(--line); }
  .metric-card { padding: 10px 12px; border-right: 1px solid var(--line);
                 border-bottom: 1px solid var(--line); }
  .metric-card .label { font-size: 10px; font-weight: 600; text-transform: uppercase;
                        letter-spacing: .06em; color: var(--muted); margin-bottom: 2px; }
  .metric-card .value { font-size: 18px; font-weight: 600; color: var(--ink); }
  .chart-img { width: 100%; max-width: 900px; margin: 10px auto 18px; display: block; }
  .toc { border: 1px solid var(--line); padding: 16px 24px; margin: 20px 0; columns: 2; }
  .toc h3 { margin-top: 0; column-span: all; }
  .toc ol { padding-left: 18px; }
  .toc li { margin: 3px 0; color: var(--muted); }
  .toc a { color: var(--ink); text-decoration: none; }
  .toc a:hover { color: var(--accent); }
  .footer { margin-top: 48px; padding: 16px 0; border-top: 1px solid var(--line);
            text-align: center; font-size: 11px; color: var(--muted); }
  @media print {
    .cover { page-break-after: always; }
    h2 { page-break-before: always; }
    table { page-break-inside: avoid; }
    .chart-img { page-break-inside: avoid; }
  }
</style>
</head>
<body>

<!-- Cover -->
<div class="cover">
  <div class="eyebrow">Portfolio Analyzer</div>
  <h1>Portfolio Analysis Report</h1>
  <div class="subtitle">{{ tickers_str }}</div>
  <div class="meta">
    Benchmark: {{ benchmark }}<br>
    Period: {{ start_date }} to {{ end_date }}<br>
    {{ capital_line }}<br>
    Risk-Free Rate: {{ rf_rate }}<br>
    Generated: {{ generated_date }}
  </div>
</div>

<div class="container">

<!-- Table of Contents -->
<div class="toc">
  <h3>Table of Contents</h3>
  <ol>
    <li><a href="#summary">Executive Summary</a></li>
    <li><a href="#performance">Performance Analysis</a></li>
    <li><a href="#risk">Risk Analysis</a></li>
    <li><a href="#drawdown">Drawdown Analysis</a></li>
    <li><a href="#capm">CAPM Regression</a></li>
    <li><a href="#optimization">Portfolio Optimization</a></li>
    {% for sec in sections %}<li><a href="#{{ sec.id }}">{{ sec.title|e }}</a></li>{% endfor %}
    <li><a href="#income">Income Analysis</a></li>
    <li><a href="#correlation">Correlation & Diversification</a></li>
    <li><a href="#simulation">Monte Carlo Simulation</a></li>
    <li><a href="#perfmeasure">Performance Measurement</a></li>
    {% if tax_metrics %}<li><a href="#tax">Tax Analysis</a></li>{% endif %}
    {% if plan %}<li><a href="#retirement">Retirement Planning</a></li>{% endif %}
    <li><a href="#holdings">Holdings Detail</a></li>
    <li><a href="#disclosures">Important Disclosures</a></li>
  </ol>
</div>

<!-- Executive Summary -->
<h2 id="summary">Executive Summary</h2>
<div class="interpretation">{{ interp.executive_summary }}</div>

{% if summary_metrics %}
<div class="metrics-grid">
  {% for m in summary_metrics %}
  <div class="metric-card">
    <div class="label">{{ m.label }}</div>
    <div class="value">{{ m.value }}</div>
  </div>
  {% endfor %}
</div>
{% endif %}

<!-- Performance -->
<h2 id="performance">Performance Analysis</h2>
<div class="interpretation">{{ interp.performance }}</div>

{% if perf_table %}
<h3>Performance Summary</h3>
<table>
  <thead><tr><th>Portfolio</th><th>Ann. Return</th><th>Ann. Vol</th><th>Sharpe</th><th>Max DD</th></tr></thead>
  <tbody>
  {% for row in perf_table %}
  <tr><td>{{ row.name }}</td><td>{{ row.ret }}</td><td>{{ row.vol }}</td><td>{{ row.sharpe }}</td><td>{{ row.dd }}</td></tr>
  {% endfor %}
  </tbody>
</table>
{% endif %}

{% if growth_chart_img %}
<img class="chart-img" src="data:image/png;base64,{{ growth_chart_img }}" alt="Growth Chart">
{% endif %}

{% if active_alloc_img %}
<h3>Portfolio Allocation</h3>
<img class="chart-img" src="data:image/png;base64,{{ active_alloc_img }}" alt="Active Portfolio Allocation">
{% endif %}

<!-- Risk -->
<h2 id="risk">Risk Analysis</h2>
<div class="interpretation">{{ interp.risk }}</div>

{% if risk_table %}
<h3>Tail Risk Metrics</h3>
<table>
  <thead><tr><th>Portfolio</th><th>Max DD</th><th>VaR 95%</th><th>CVaR 95%</th><th>VaR 99%</th><th>CVaR 99%</th></tr></thead>
  <tbody>
  {% for row in risk_table %}
  <tr><td>{{ row.portfolio }}</td><td>{{ row.max_dd }}</td><td>{{ row.var95 }}</td><td>{{ row.cvar95 }}</td><td>{{ row.var99 }}</td><td>{{ row.cvar99 }}</td></tr>
  {% endfor %}
  </tbody>
</table>
{% endif %}

<!-- Drawdown -->
<h2 id="drawdown">Drawdown Analysis</h2>
<div class="interpretation">{{ interp.drawdown }}</div>

{% if drawdown_chart_img %}
<img class="chart-img" src="data:image/png;base64,{{ drawdown_chart_img }}" alt="Drawdown Chart">
{% endif %}

<!-- CAPM -->
<h2 id="capm">CAPM Regression</h2>
<div class="interpretation">{{ interp.capm }}</div>

{% if capm_table %}
<h3>CAPM Results by Asset</h3>
<table>
  <thead><tr><th>Asset</th><th>Alpha (mo.)</th><th>Beta</th><th>t(a)</th><th>t(b)</th><th>R2</th></tr></thead>
  <tbody>
  {% for row in capm_table %}
  <tr><td>{{ row.ticker }}</td><td>{{ row.alpha }}</td><td>{{ row.beta }}</td><td>{{ row.t_alpha }}</td><td>{{ row.t_beta }}</td><td>{{ row.r2 }}</td></tr>
  {% endfor %}
  </tbody>
</table>
{% endif %}

<!-- Optimization -->
<h2 id="optimization">Portfolio Optimization</h2>
<div class="interpretation">{{ interp.optimization }}</div>

{% if orp_weights_table %}
<h3>ORP Weights</h3>
<table>
  <thead><tr><th>Asset</th><th>ORP Weight</th>{% if has_hrp %}<th>HRP Weight</th>{% endif %}</tr></thead>
  <tbody>
  {% for row in orp_weights_table %}
  <tr><td>{{ row.asset }}</td><td>{{ row.orp }}</td>{% if has_hrp %}<td>{{ row.hrp }}</td>{% endif %}</tr>
  {% endfor %}
  </tbody>
</table>
{% endif %}

{% if frontier_chart_img %}
<img class="chart-img" src="data:image/png;base64,{{ frontier_chart_img }}" alt="Efficient Frontier">
{% endif %}

<!-- Risk & attribution analytics (src/reports/sections.py) -->
{% for sec in sections %}
<h2 id="{{ sec.id }}">{{ sec.title|e }}</h2>
{% if sec.id == 'stress' and interp.stress_tests %}<div class="interpretation">{{ interp.stress_tests }}</div>{% endif %}
{% if sec.note %}<p class="note">{{ sec.note|e }}</p>{% endif %}
{% for t in sec.tables %}
{% if t.title %}<h3>{{ t.title|e }}</h3>{% endif %}
<table>
  <thead><tr>{% for c in t.columns %}<th class="{{ 'r' if t.align[loop.index0] == 'r' else '' }}">{{ c|e }}</th>{% endfor %}</tr></thead>
  <tbody>
  {% for row in t.rows %}
  <tr class="{{ 'tot' if row[0] == 'Total' else '' }}">{% for v in row %}{% set al = t.align[loop.index0] %}<td class="{{ 'r' if al == 'r' else '' }} {{ sign_class(v, t.signed) }}">{{ v|e }}</td>{% endfor %}</tr>
  {% endfor %}
  </tbody>
</table>
{% endfor %}
{% endfor %}

<!-- Income -->
<h2 id="income">Income Analysis</h2>
<div class="interpretation">{{ interp.income }}</div>

{% if income_table %}
<h3>Dividend Income by Holding</h3>
<table>
  <thead><tr><th>Ticker</th><th>Shares</th><th>Ann. Div/Share</th><th>Ann. Income</th><th>Yield on Cost</th><th>Current Yield</th></tr></thead>
  <tbody>
  {% for row in income_table %}
  <tr><td>{{ row.ticker }}</td><td>{{ row.shares }}</td><td>{{ row.div_ps }}</td><td>{{ row.income }}</td><td>{{ row.yoc }}</td><td>{{ row.cur_yield }}</td></tr>
  {% endfor %}
  </tbody>
</table>
{% endif %}

<!-- Correlation -->
<h2 id="correlation">Correlation & Diversification</h2>
<div class="interpretation">{{ interp.correlation }}</div>

{% if corr_chart_img %}
<img class="chart-img" src="data:image/png;base64,{{ corr_chart_img }}" alt="Correlation Heatmap">
{% endif %}

<!-- Simulation -->
<h2 id="simulation">Monte Carlo Simulation</h2>
<div class="interpretation">{{ interp.simulation }}</div>

{% if sim_table %}
<h3>Simulation Comparison</h3>
<table>
  <thead><tr><th>Method</th><th>Expected Value</th><th>P(Loss)</th><th>P5</th><th>P50</th><th>P95</th></tr></thead>
  <tbody>
  {% for row in sim_table %}
  <tr><td>{{ row.method }}</td><td>{{ row.expected }}</td><td>{{ row.ploss }}</td><td>{{ row.p5 }}</td><td>{{ row.p50 }}</td><td>{{ row.p95 }}</td></tr>
  {% endfor %}
  </tbody>
</table>
{% endif %}

<!-- Performance Measurement -->
<h2 id="perfmeasure">Performance Measurement</h2>
<p>Benchmark-relative measures on the active portfolio: how much of the market's up
and down moves it captured, and how consistently it beat the benchmark.</p>
{% if capture_grid %}
<div class="metrics-grid">
  {% for m in capture_grid %}
  <div class="metric-card">
    <div class="label">{{ m.label }}</div>
    <div class="value">{{ m.value }}</div>
  </div>
  {% endfor %}
</div>
{% else %}
<p><em>Benchmark-relative measures were not available for this run.</em></p>
{% endif %}

{% if tax_metrics %}
<!-- Tax -->
<h2 id="tax">Tax Analysis</h2>
<p>Unrealized gains/losses on current holdings, tax-loss-harvesting candidates, and
estimated tax on gains realized by rebalancing (simple average-cost basis).</p>
<div class="metrics-grid">
  <div class="metric-card"><div class="label">Unrealized Gain</div><div class="value">{{ tax_metrics.unrealized }}</div></div>
  <div class="metric-card"><div class="label">Harvestable Losses</div><div class="value">{{ tax_metrics.harvest }}</div></div>
  <div class="metric-card"><div class="label">Loss Candidates</div><div class="value">{{ tax_metrics.n_harvest }}</div></div>
  <div class="metric-card"><div class="label">Est. Tax (realized)</div><div class="value">{{ tax_metrics.est_tax }}</div></div>
</div>
{% if tax_detail %}
<h3>Position Detail</h3>
<table>
  <thead><tr><th>Ticker</th><th>Shares</th><th>Avg Cost</th><th>Price</th><th>Market Value</th><th>Unrealized</th><th>Return</th></tr></thead>
  <tbody>
  {% for row in tax_detail %}
  <tr><td>{{ row.ticker }}</td><td>{{ row.shares }}</td><td>{{ row.avg_cost }}</td><td>{{ row.price }}</td><td>{{ row.mv }}</td><td>{{ row.unreal }}</td><td>{{ row.ret }}</td></tr>
  {% endfor %}
  </tbody>
</table>
{% endif %}
{% endif %}

{% if plan %}
<!-- Retirement -->
<h2 id="retirement">Retirement Planning</h2>
<p>Monte Carlo projection of the portfolio over a {{ plan.horizon }}-year horizon, including
any planned contributions and withdrawals.</p>
<div class="metrics-grid">
  <div class="metric-card"><div class="label">Success Probability</div><div class="value">{{ plan.success }}</div></div>
  <div class="metric-card"><div class="label">Depletion Risk</div><div class="value">{{ plan.depletion }}</div></div>
  <div class="metric-card"><div class="label">Median Ending</div><div class="value">{{ plan.median }}</div></div>
  <div class="metric-card"><div class="label">Worst Case (P5)</div><div class="value">{{ plan.p5 }}</div></div>
  {% if plan.swr %}<div class="metric-card"><div class="label">Safe Withdrawal Rate</div><div class="value">{{ plan.swr }}</div></div>{% endif %}
  {% if plan.goal %}<div class="metric-card"><div class="label">Goal Probability</div><div class="value">{{ plan.goal }}</div></div>{% endif %}
</div>
{% endif %}

<!-- Holdings -->
<h2 id="holdings">Holdings Detail</h2>

{% if holdings_table %}
<table>
  <thead><tr><th>Ticker</th><th>Target Wt</th><th>Realized Wt</th><th>Shares</th><th>Purchase Price</th><th>Invested</th></tr></thead>
  <tbody>
  {% for row in holdings_table %}
  <tr><td>{{ row.ticker }}</td><td>{{ row.target }}</td><td>{{ row.realized }}</td><td>{{ row.shares }}</td><td>{{ row.price }}</td><td>{{ row.invested }}</td></tr>
  {% endfor %}
  </tbody>
</table>
{% endif %}

<!-- Disclosures -->
<h2 id="disclosures">Important Disclosures</h2>
<div class="interpretation">{{ disclosures }}</div>

<div class="footer">
  Portfolio Analyzer &nbsp;|&nbsp; Generated {{ generated_date }}
</div>

</div>
</body>
</html>
""", autoescape=True)


# ---------------------------------------------------------------
# Builder
# ---------------------------------------------------------------

def _fmt_pct(v, decimals=2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    return f"{v * 100:.{decimals}f}%"


def _fmt_dollar(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    if abs(v) >= 1e6:
        return f"${v / 1e6:,.2f}M"
    return f"${v:,.0f}"


def _fmt_shares(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    return f"{v:,.0f}" if abs(v - round(v)) < 1e-6 else f"{v:,.4f}"


def _fmt_ratio(v, d=2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    return f"{v:.{d}f}"


DISCLOSURES = (
    "This report is generated by Portfolio Analyzer for informational and educational "
    "purposes only. It is not investment, tax, or financial advice, and is not an offer or "
    "solicitation to buy or sell any security. Past performance and backtested or simulated "
    "results do not guarantee future results; simulated results have inherent limitations "
    "because they are prepared with the benefit of hindsight and do not reflect actual "
    "trading. Figures rely on third-party market data that may contain errors or gaps. "
    "Tax figures use a simplified average-cost model and do not constitute tax advice. "
    "Consult a qualified professional before making investment decisions."
)


def build_html_report(
    results: AnalysisResults,
    chart_figures: Optional[dict] = None,
) -> str:
    """
    Build a standalone HTML report from AnalysisResults.

    Parameters
    ----------
    results : AnalysisResults from pipeline
    chart_figures : optional dict of chart_name -> plotly Figure for image embedding

    Returns
    -------
    HTML string
    """
    interp = generate_full_interpretation(results)
    chart_figures = chart_figures or {}

    # Summary metrics
    summary_metrics = []
    if results.active:
        summary_metrics.append({"label": "Total Return", "value": _fmt_pct(results.active.ann_return)})
        summary_metrics.append({"label": "Sharpe Ratio", "value": f"{results.active.sharpe:.2f}" if not np.isnan(results.active.sharpe) else "N/A"})
        summary_metrics.append({"label": "Max Drawdown", "value": _fmt_pct(results.active.max_dd)})
        summary_metrics.append({"label": "Volatility", "value": _fmt_pct(results.active.ann_vol)})

    if results.capm_results:
        avg_alpha = float(np.mean([r.alpha for r in results.capm_results]))
        avg_beta = float(np.mean([r.beta for r in results.capm_results]))
        summary_metrics.append({"label": "Alpha (ann.)", "value": f"{avg_alpha * 12:.2%}"})
        summary_metrics.append({"label": "Beta", "value": f"{avg_beta:.2f}"})

    # Performance table
    perf_table = []
    for ps in [results.active, results.passive, results.orp, results.hrp, results.rebalanced, results.complete]:
        if ps is None:
            continue
        perf_table.append({
            "name": ps.name,
            "ret": _fmt_pct(ps.ann_return),
            "vol": _fmt_pct(ps.ann_vol),
            "sharpe": f"{ps.sharpe:.2f}" if not np.isnan(ps.sharpe) else "N/A",
            "dd": _fmt_pct(ps.max_dd),
        })

    # Risk table
    risk_table = []
    if results.drawdown_metrics is not None:
        for _, r in results.drawdown_metrics.iterrows():
            risk_table.append({
                "portfolio": r["Portfolio"],
                "max_dd": _fmt_pct(r["MaxDrawdown"]),
                "var95": _fmt_pct(r["VaR_95"]),
                "cvar95": _fmt_pct(r["CVaR_95"]),
                "var99": _fmt_pct(r["VaR_99"]),
                "cvar99": _fmt_pct(r["CVaR_99"]),
            })

    # CAPM table
    capm_table = []
    for r in results.capm_results:
        capm_table.append({
            "ticker": r.ticker,
            "alpha": f"{r.alpha:.4f}",
            "beta": f"{r.beta:.2f}",
            "t_alpha": f"{r.t_alpha:.2f}",
            "t_beta": f"{r.t_beta:.2f}",
            "r2": f"{r.r_squared:.3f}",
        })

    # ORP weights table
    orp_weights_table = []
    has_hrp = results.hrp_weights is not None
    if results.orp_optimization:
        for asset, w in results.orp_optimization.weights.items():
            row = {"asset": asset, "orp": _fmt_pct(float(w))}
            if has_hrp and asset in results.hrp_weights.index:
                row["hrp"] = _fmt_pct(float(results.hrp_weights[asset]))
            else:
                row["hrp"] = "N/A"
            orp_weights_table.append(row)

    # Income table
    income_table = []
    if results.income_summary is not None and not results.income_summary.empty:
        for _, r in results.income_summary.iterrows():
            income_table.append({
                "ticker": r["Ticker"],
                "shares": f"{r['Shares']:.0f}",
                "div_ps": f"${r['AnnualDividendPerShare']:.4f}",
                "income": f"${r['AnnualIncome']:,.2f}",
                "yoc": _fmt_pct(r["YieldOnCost"]),
                "cur_yield": _fmt_pct(r["CurrentYield"]),
            })

    # Simulation table
    sim_table = []
    for sim in results.simulations:
        sim_table.append({
            "method": sim.name,
            "expected": _fmt_dollar(sim.expected_value),
            "ploss": _fmt_pct(sim.prob_loss),
            "p5": _fmt_dollar(sim.percentiles["P5"]),
            "p50": _fmt_dollar(sim.percentiles["P50"]),
            "p95": _fmt_dollar(sim.percentiles["P95"]),
        })

    # Holdings table
    holdings_table = []
    if results.holdings is not None:
        for _, r in results.holdings.iterrows():
            holdings_table.append({
                "ticker": r["Ticker"],
                "target": _fmt_pct(r.get("TargetWeight", 0)),
                "realized": _fmt_pct(r.get("RealizedWeight", 0)),
                "shares": _fmt_shares(r["Shares"]),
                "price": f"${r['PurchasePrice']:.2f}",
                "invested": _fmt_dollar(r["Invested"]),
            })

    # Performance-measurement capture grid
    capture_grid = []
    cap = results.capture_metrics
    if cap:
        capture_grid = [
            {"label": "Up Capture", "value": _fmt_pct(cap.get("up_capture"))},
            {"label": "Down Capture", "value": _fmt_pct(cap.get("down_capture"))},
            {"label": "Capture Ratio", "value": _fmt_ratio(cap.get("capture_ratio"))},
            {"label": "Batting Avg", "value": _fmt_pct(cap.get("batting_avg"))},
        ]

    # Tax section
    tax_metrics_ctx = None
    tax_detail_ctx = []
    if results.tax_metrics:
        tm = results.tax_metrics
        tax_metrics_ctx = {
            "unrealized": _fmt_dollar(tm.get("unrealized_gain", 0)),
            "harvest": _fmt_dollar(tm.get("harvest_potential", 0)),
            "n_harvest": str(tm.get("n_harvest", 0)),
            "est_tax": _fmt_dollar(tm.get("estimated_tax", 0)),
        }
        td = results.tax_detail
        if td is not None and not td.empty:
            for _, r in td.iterrows():
                tax_detail_ctx.append({
                    "ticker": r["Ticker"],
                    "shares": _fmt_shares(r["Shares"]),
                    "avg_cost": f"${r['AvgCost']:,.2f}" if r["AvgCost"] == r["AvgCost"] else "—",
                    "price": f"${r['CurrentPrice']:,.2f}" if r["CurrentPrice"] == r["CurrentPrice"] else "—",
                    "mv": _fmt_dollar(r["MarketValue"]),
                    "unreal": _fmt_dollar(r["UnrealizedGain"]) if r["UnrealizedGain"] == r["UnrealizedGain"] else "—",
                    "ret": _fmt_pct(r["Return"]) if r["Return"] == r["Return"] else "—",
                })

    # Retirement section
    plan_ctx = None
    plan = results.plan_result
    if plan is not None:
        plan_ctx = {
            "horizon": plan.horizon_years,
            "success": f"{plan.success_prob:.0%}",
            "depletion": f"{plan.depletion_prob:.0%}",
            "median": _fmt_dollar(plan.median_terminal),
            "p5": _fmt_dollar(plan.percentiles.get("P5")),
            "swr": f"{plan.safe_withdrawal_rate:.1%}" if plan.safe_withdrawal_rate is not None else None,
            "goal": f"{plan.goal_prob:.0%}" if plan.goal_amount else None,
        }

    # Export chart images
    growth_chart_img = _export_chart_base64(chart_figures["growth"]) if "growth" in chart_figures else None
    drawdown_chart_img = _export_chart_base64(chart_figures["drawdown"]) if "drawdown" in chart_figures else None
    frontier_chart_img = _export_chart_base64(chart_figures["frontier"]) if "frontier" in chart_figures else None
    corr_chart_img = _export_chart_base64(chart_figures["correlation"]) if "correlation" in chart_figures else None
    active_alloc_img = _export_chart_base64(chart_figures["active_allocation"]) if "active_allocation" in chart_figures else None

    _cash = float(getattr(results.config, "cash", 0.0) or 0.0)
    if _cash > 0:
        _invested = results.config.capital - _cash
        capital_line = (
            f"Total Capital: {_fmt_dollar(results.config.capital)} "
            f"(invested {_fmt_dollar(_invested)} + cash {_fmt_dollar(_cash)})"
        )
    else:
        capital_line = f"Initial Capital: {_fmt_dollar(results.config.capital)}"

    from src.reports.sections import analytics_sections, sign_class

    return _HTML_TEMPLATE.render(
        sections=analytics_sections(results), sign_class=sign_class,
        tickers_str=", ".join(results.config.tickers),
        benchmark=results.config.benchmark,
        start_date=str(results.config.start_date),
        end_date=str(results.config.end_date),
        capital_line=capital_line,
        rf_rate=_fmt_pct(results.config.risk_free_rate),
        generated_date=datetime.now().strftime("%Y-%m-%d %H:%M"),
        interp=interp,
        summary_metrics=summary_metrics,
        perf_table=perf_table,
        risk_table=risk_table,
        capm_table=capm_table,
        orp_weights_table=orp_weights_table,
        has_hrp=has_hrp,
        income_table=income_table,
        sim_table=sim_table,
        holdings_table=holdings_table,
        capture_grid=capture_grid,
        tax_metrics=tax_metrics_ctx,
        tax_detail=tax_detail_ctx,
        plan=plan_ctx,
        disclosures=DISCLOSURES,
        growth_chart_img=growth_chart_img,
        drawdown_chart_img=drawdown_chart_img,
        frontier_chart_img=frontier_chart_img,
        corr_chart_img=corr_chart_img,
        active_alloc_img=active_alloc_img,
    )
