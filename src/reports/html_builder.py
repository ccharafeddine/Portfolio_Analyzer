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
<title>Portfolio Analysis Report</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }

  body {
    font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
    color: #1e293b;
    background: #ffffff;
    line-height: 1.6;
    font-size: 14px;
  }

  .cover {
    background: linear-gradient(135deg, #0B1120 0%, #1E3A5F 100%);
    color: #F1F5F9;
    padding: 80px 60px;
    margin-bottom: 40px;
  }
  .cover h1 {
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 12px;
    letter-spacing: -0.02em;
  }
  .cover .subtitle {
    font-size: 1.1rem;
    color: #94A3B8;
    margin-bottom: 30px;
  }
  .cover .meta {
    font-size: 0.9rem;
    color: #64748B;
    line-height: 1.8;
  }

  .container {
    max-width: 900px;
    margin: 0 auto;
    padding: 0 40px;
  }

  h2 {
    font-size: 1.5rem;
    font-weight: 700;
    color: #0B1120;
    margin-top: 48px;
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 2px solid #3B82F6;
  }

  h3 {
    font-size: 1.15rem;
    font-weight: 600;
    color: #1E3A5F;
    margin-top: 28px;
    margin-bottom: 8px;
  }

  p, .interpretation {
    margin-bottom: 16px;
    color: #334155;
  }

  .interpretation {
    background: #F8FAFC;
    border-left: 3px solid #3B82F6;
    padding: 16px 20px;
    border-radius: 0 8px 8px 0;
    font-size: 0.9rem;
    line-height: 1.7;
  }

  table {
    width: 100%;
    border-collapse: collapse;
    margin: 16px 0 24px 0;
    font-size: 0.85rem;
  }
  th {
    background: #0B1120;
    color: #F1F5F9;
    padding: 10px 14px;
    text-align: left;
    font-weight: 600;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }
  td {
    padding: 8px 14px;
    border-bottom: 1px solid #E2E8F0;
    color: #334155;
  }
  tr:nth-child(even) { background: #F8FAFC; }
  tr:hover { background: #EFF6FF; }

  .metrics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 16px;
    margin: 20px 0;
  }
  .metric-card {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 16px;
    text-align: center;
  }
  .metric-card .label {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #64748B;
    margin-bottom: 4px;
  }
  .metric-card .value {
    font-size: 1.4rem;
    font-weight: 700;
    color: #0B1120;
    font-family: 'JetBrains Mono', 'Consolas', monospace;
  }

  .chart-img {
    width: 100%;
    max-width: 860px;
    margin: 16px auto;
    display: block;
    border-radius: 8px;
  }

  .toc {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 24px 32px;
    margin: 24px 0;
  }
  .toc h3 { margin-top: 0; color: #0B1120; }
  .toc ol { padding-left: 20px; }
  .toc li {
    margin: 6px 0;
    color: #3B82F6;
  }
  .toc a { color: #3B82F6; text-decoration: none; }
  .toc a:hover { text-decoration: underline; }

  .footer {
    margin-top: 60px;
    padding: 20px 0;
    border-top: 1px solid #E2E8F0;
    text-align: center;
    font-size: 0.8rem;
    color: #94A3B8;
  }

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
  <h1>Portfolio Analysis Report</h1>
  <div class="subtitle">{{ tickers_str }}</div>
  <div class="meta">
    Benchmark: {{ benchmark }}<br>
    Period: {{ start_date }} to {{ end_date }}<br>
    Initial Capital: {{ capital }}<br>
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
    <li><a href="#stress">Stress Testing</a></li>
    <li><a href="#income">Income Analysis</a></li>
    <li><a href="#correlation">Correlation & Diversification</a></li>
    <li><a href="#simulation">Monte Carlo Simulation</a></li>
    <li><a href="#holdings">Holdings Detail</a></li>
  </ol>
</div>

<!-- Executive Summary -->
<h2 id="summary">1. Executive Summary</h2>
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
<h2 id="performance">2. Performance Analysis</h2>
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

<!-- Risk -->
<h2 id="risk">3. Risk Analysis</h2>
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
<h2 id="drawdown">4. Drawdown Analysis</h2>
<div class="interpretation">{{ interp.drawdown }}</div>

{% if drawdown_chart_img %}
<img class="chart-img" src="data:image/png;base64,{{ drawdown_chart_img }}" alt="Drawdown Chart">
{% endif %}

<!-- CAPM -->
<h2 id="capm">5. CAPM Regression</h2>
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
<h2 id="optimization">6. Portfolio Optimization</h2>
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

<!-- Stress Testing -->
<h2 id="stress">7. Stress Testing</h2>
<div class="interpretation">{{ interp.stress_tests }}</div>

{% if stress_table %}
<h3>Historical Scenario Results</h3>
<table>
  <thead><tr><th>Scenario</th><th>Period</th><th>Portfolio</th><th>Benchmark</th></tr></thead>
  <tbody>
  {% for row in stress_table %}
  <tr><td>{{ row.scenario }}</td><td>{{ row.period }}</td><td>{{ row.portfolio }}</td><td>{{ row.benchmark }}</td></tr>
  {% endfor %}
  </tbody>
</table>
{% endif %}

<!-- Income -->
<h2 id="income">8. Income Analysis</h2>
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
<h2 id="correlation">9. Correlation & Diversification</h2>
<div class="interpretation">{{ interp.correlation }}</div>

{% if corr_chart_img %}
<img class="chart-img" src="data:image/png;base64,{{ corr_chart_img }}" alt="Correlation Heatmap">
{% endif %}

<!-- Simulation -->
<h2 id="simulation">10. Monte Carlo Simulation</h2>
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

<!-- Holdings -->
<h2 id="holdings">11. Holdings Detail</h2>

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

<div class="footer">
  Portfolio Analyzer v2 | Generated {{ generated_date }}
</div>

</div>
</body>
</html>
""")


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

    # Stress table
    stress_table = []
    if results.stress_df is not None:
        for _, r in results.stress_df.iterrows():
            stress_table.append({
                "scenario": r.get("Scenario", ""),
                "period": r.get("Period", ""),
                "portfolio": r.get("Portfolio", "N/A"),
                "benchmark": r.get("Benchmark", "N/A"),
            })

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
                "shares": f"{r['Shares']:.0f}",
                "price": f"${r['PurchasePrice']:.2f}",
                "invested": _fmt_dollar(r["Invested"]),
            })

    # Export chart images
    growth_chart_img = _export_chart_base64(chart_figures["growth"]) if "growth" in chart_figures else None
    drawdown_chart_img = _export_chart_base64(chart_figures["drawdown"]) if "drawdown" in chart_figures else None
    frontier_chart_img = _export_chart_base64(chart_figures["frontier"]) if "frontier" in chart_figures else None
    corr_chart_img = _export_chart_base64(chart_figures["correlation"]) if "correlation" in chart_figures else None

    return _HTML_TEMPLATE.render(
        tickers_str=", ".join(results.config.tickers),
        benchmark=results.config.benchmark,
        start_date=str(results.config.start_date),
        end_date=str(results.config.end_date),
        capital=_fmt_dollar(results.config.capital),
        rf_rate=_fmt_pct(results.config.risk_free_rate),
        generated_date=datetime.now().strftime("%Y-%m-%d %H:%M"),
        interp=interp,
        summary_metrics=summary_metrics,
        perf_table=perf_table,
        risk_table=risk_table,
        capm_table=capm_table,
        orp_weights_table=orp_weights_table,
        has_hrp=has_hrp,
        stress_table=stress_table,
        income_table=income_table,
        sim_table=sim_table,
        holdings_table=holdings_table,
        growth_chart_img=growth_chart_img,
        drawdown_chart_img=drawdown_chart_img,
        frontier_chart_img=frontier_chart_img,
        corr_chart_img=corr_chart_img,
    )
