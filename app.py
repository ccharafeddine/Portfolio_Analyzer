"""
Portfolio Analyzer v2 — Streamlit App

Bloomberg-style dark theme, interactive Plotly charts,
direct pipeline execution (no subprocess), tabbed layout.
"""

import io
import json
import os
import sys
import zipfile
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ── Ensure project root is on path for imports ──
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.models import PortfolioConfig, BLView, BLConfig, CompletePortfolioConfig
from src.pipeline import AnalysisPipeline, AnalysisResults
from src.charts import plotly_charts as charts
from src.data import transforms as T
from src.reports.html_builder import build_html_report
from src.reports.pdf_builder import build_pdf_report


# ══════════════════════════════════════════════════════════════
# Page config & theme
# ══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Portfolio Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Bloomberg-style dark CSS ──
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,500;0,9..40,700;1,9..40,400&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Global */
    .stApp {
        background-color: #0B1120;
        font-family: 'DM Sans', sans-serif;
    }

    /* Header bar */
    .app-header {
        background: linear-gradient(135deg, #0D1526 0%, #131B2E 100%);
        border-bottom: 1px solid #1E293B;
        padding: 16px 24px;
        margin: -1rem -1rem 1.5rem -1rem;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .app-header h1 {
        color: #F1F5F9;
        font-size: 1.5rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .app-header .badge {
        background: #1E3A5F;
        color: #3B82F6;
        font-size: 0.7rem;
        padding: 3px 10px;
        border-radius: 100px;
        font-weight: 600;
        letter-spacing: 0.05em;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #151D2E 0%, #1A2438 100%);
        border: 1px solid #2D3A50;
        border-radius: 12px;
        padding: 18px 22px;
        transition: border-color 0.2s;
    }
    div[data-testid="stMetric"]:hover {
        border-color: #3B82F6;
    }
    div[data-testid="stMetric"] label {
        color: #64748B !important;
        font-size: 0.8rem !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #F1F5F9 !important;
        font-size: 1.7rem !important;
        font-weight: 700 !important;
        font-family: 'JetBrains Mono', monospace !important;
    }
    div[data-testid="stMetricDelta"] > div {
        font-family: 'JetBrains Mono', monospace !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background-color: transparent;
        border-bottom: 1px solid #1E293B;
        padding-bottom: 0;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        color: #64748B;
        font-weight: 500;
        font-size: 0.9rem;
        border-bottom: 2px solid transparent;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(59, 130, 246, 0.08);
        color: #3B82F6 !important;
        border-bottom: 2px solid #3B82F6;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #94A3B8;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0D1526;
        border-right: 1px solid #1E293B;
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #E2E8F0;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #1E3A5F 0%, #2563EB 100%);
        color: #F1F5F9;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #2563EB 0%, #3B82F6 100%);
        box-shadow: 0 0 20px rgba(59, 130, 246, 0.3);
    }

    /* Section dividers */
    .section-label {
        color: #475569;
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin: 24px 0 8px 0;
        padding-bottom: 6px;
        border-bottom: 1px solid #1E293B;
    }

    /* DataFrames */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }

    /* Plotly chart containers */
    .stPlotlyChart {
        border-radius: 12px;
        overflow: hidden;
    }

    /* Hide Streamlit chrome (keep sidebar toggle visible) */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header {
        background: transparent !important;
    }
    header [data-testid="stDecoration"] {
        display: none;
    }

    /* Download buttons */
    .stDownloadButton > button {
        background: #151D2E;
        border: 1px solid #2D3A50;
        color: #94A3B8;
        font-size: 0.85rem;
    }
    .stDownloadButton > button:hover {
        border-color: #3B82F6;
        color: #F1F5F9;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background-color: #151D2E;
        border-radius: 8px;
    }

    /* Progress bar */
    .stProgress > div > div {
        background-color: #3B82F6;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# Session state
# ══════════════════════════════════════════════════════════════

if "results" not in st.session_state:
    st.session_state.results = None
if "run_status" not in st.session_state:
    st.session_state.run_status = None
if "run_error" not in st.session_state:
    st.session_state.run_error = None


# ══════════════════════════════════════════════════════════════
# Header
# ══════════════════════════════════════════════════════════════

st.markdown("""
<div class="app-header">
    <h1>Portfolio Analyzer</h1>
    <span class="badge">v2.0</span>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# Sidebar: Configuration
# ══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### Configuration")

    # ── Tickers ──
    st.markdown('<div class="section-label">Universe</div>', unsafe_allow_html=True)
    tickers_text = st.text_area(
        "Tickers (one per line)",
        value="AAPL\nMSFT\nNVDA\nGLD\nTLT",
        height=150,
        help="Enter stock/ETF/crypto tickers, one per line",
    )
    tickers = [t.strip().upper() for t in tickers_text.split("\n") if t.strip()]

    # ── Weights ──
    equal_weights = st.checkbox("Equal weights", value=True)

    weights_dict: dict[str, float] = {}
    if tickers:
        if equal_weights:
            w = 1.0 / len(tickers)
            weights_dict = {t: round(w, 6) for t in tickers}
        else:
            st.markdown('<div class="section-label">Weights</div>', unsafe_allow_html=True)
            cols = st.columns(2)
            for i, t in enumerate(tickers):
                default_w = round(1.0 / len(tickers), 4)
                w = cols[i % 2].number_input(
                    t, min_value=0.0, max_value=2.0, value=default_w,
                    step=0.01, format="%.4f", key=f"w_{t}",
                )
                weights_dict[t] = float(w)

            total_w = sum(weights_dict.values())
            if abs(total_w - 1.0) > 0.01:
                st.error(f"Weights sum to {total_w:.4f} — must be ≈ 1.0")

    # ── Dates ──
    st.markdown('<div class="section-label">Date Range</div>', unsafe_allow_html=True)
    col_d1, col_d2 = st.columns(2)
    start_date = col_d1.date_input("Start", date(2020, 1, 1))
    end_date = col_d2.date_input("End", date(2025, 12, 31))

    # ── Settings ──
    st.markdown('<div class="section-label">Settings</div>', unsafe_allow_html=True)
    benchmark = st.text_input("Benchmark", value="SPY")
    capital = st.number_input("Capital ($)", value=1_000_000, step=100_000, format="%d")
    rf_rate = st.number_input("Risk-free rate", value=0.04, step=0.005, format="%.3f")

    col_s1, col_s2 = st.columns(2)
    allow_shorts = col_s1.checkbox("Allow shorts", value=False)
    include_orp = col_s2.checkbox("Include ORP", value=True)

    max_bound = st.slider("Max weight bound", 0.5, 2.0, 1.0, 0.1)
    y_cp = st.slider("Complete portfolio: % in ORP", 0.0, 1.0, 0.8, 0.05)

    # ── Run ──
    st.markdown("---")
    run_clicked = st.button("▶ Run Analysis", use_container_width=True, type="primary")


# ══════════════════════════════════════════════════════════════
# Run pipeline
# ══════════════════════════════════════════════════════════════

if run_clicked and tickers and weights_dict:
    try:
        config = PortfolioConfig(
            tickers=tickers,
            weights=weights_dict,
            benchmark=benchmark.strip().upper(),
            start_date=start_date,
            end_date=end_date,
            capital=float(capital),
            risk_free_rate=float(rf_rate),
            short_sales=allow_shorts,
            max_weight_bound=max_bound,
            include_orp=include_orp,
            include_complete=True,
            complete_portfolio=CompletePortfolioConfig(y=y_cp),
        )

        # Save config for legacy modules that read config.json
        config.save("config.json")

        progress_bar = st.progress(0.0)
        status_text = st.empty()

        def on_progress(label: str, frac: float):
            progress_bar.progress(frac)
            status_text.caption(f"{label}...")

        pipeline = AnalysisPipeline(config, output_dir="outputs")
        results = pipeline.run(progress=on_progress)

        st.session_state.results = results
        st.session_state.run_status = "ok"
        st.session_state.run_error = None

        progress_bar.progress(1.0)
        status_text.caption("Analysis complete")

    except Exception as e:
        st.session_state.run_status = "error"
        st.session_state.run_error = str(e)
        st.error(f"Pipeline error: {e}")


# ══════════════════════════════════════════════════════════════
# Display results
# ══════════════════════════════════════════════════════════════

results: AnalysisResults | None = st.session_state.get("results")

if results is None:
    st.markdown("""
    <div style="
        text-align: center;
        padding: 80px 20px;
        color: #475569;
    ">
        <p style="font-size: 1.1rem; font-weight: 500;">Configure your portfolio in the sidebar and click <b>Run Analysis</b></p>
        <p style="font-size: 0.85rem; margin-top: 8px;">
            Enter tickers, set weights, choose dates, and hit the button.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ── Headline metrics ──
def _fmt_pct(v, decimals=2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v*100:.{decimals}f}%"

def _fmt_dollar(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if abs(v) >= 1e6:
        return f"${v/1e6:,.2f}M"
    return f"${v:,.0f}"

def _delta_str(v):
    """Format a delta value -- return None if not meaningful."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return f"{v*100:+.2f}%"


def _interpretation_card(text: str) -> None:
    """Render an interpretation card with styled HTML."""
    if not text:
        return
    st.markdown(
        f'<div style="background: #151D2E; border-left: 3px solid #3B82F6; '
        f'padding: 16px 20px; border-radius: 0 8px 8px 0; margin: 16px 0; '
        f'color: #94A3B8; font-size: 0.9rem; line-height: 1.6;">{text}</div>',
        unsafe_allow_html=True,
    )


m1, m2, m3, m4, m5, m6 = st.columns(6)

active_ret = results.active.ann_return if results.active else None
passive_ret = results.passive.ann_return if results.passive else None
excess = (active_ret - passive_ret) if (active_ret is not None and passive_ret is not None) else None

m1.metric(
    "Total Return",
    _fmt_pct(active_ret),
    delta=_delta_str(excess),
    help="Annualized geometric return of the active portfolio",
)
m2.metric(
    "Sharpe Ratio",
    f"{results.active.sharpe:.2f}" if results.active and not np.isnan(results.active.sharpe) else "—",
    help="Annualized Sharpe ratio (excess return / volatility)",
)
m3.metric(
    "Max Drawdown",
    _fmt_pct(results.active.max_dd if results.active else None),
    help="Maximum peak-to-trough decline",
)
m4.metric(
    "Volatility",
    _fmt_pct(results.active.ann_vol if results.active else None),
    help="Annualized portfolio volatility",
)

alpha_val = None
beta_val = None
if results.capm_results:
    alpha_val = float(np.mean([r.alpha for r in results.capm_results]))
    beta_val = float(np.mean([r.beta for r in results.capm_results]))

m5.metric(
    "Alpha",
    f"{alpha_val*12:.2%}" if alpha_val is not None else "—",
    help="Average monthly CAPM alpha, annualized",
)
m6.metric(
    "Beta",
    f"{beta_val:.2f}" if beta_val is not None else "—",
    help="Average CAPM beta vs benchmark",
)


# ══════════════════════════════════════════════════════════════
# Tabs
# ══════════════════════════════════════════════════════════════

tab_overview, tab_risk, tab_attribution, tab_income, tab_optimization, tab_forecast, tab_data = st.tabs([
    "Overview",
    "Risk",
    "Attribution",
    "Income",
    "Optimization",
    "Forecast",
    "Data",
])


# ──────────────────────────────────────────────────────────────
# TAB: Overview
# ──────────────────────────────────────────────────────────────

with tab_overview:
    # Interpretation card
    if results.interpretations and "performance" in results.interpretations:
        _interpretation_card(results.interpretations["performance"])

    # Growth chart
    growth_series = {}
    if results.active:
        growth_series["Active"] = results.active.values
    if results.passive:
        growth_series["Passive"] = results.passive.values
    if results.orp:
        growth_series["ORP"] = results.orp.values
    if results.hrp:
        growth_series["HRP"] = results.hrp.values
    if results.rebalanced:
        growth_series["Rebalanced"] = results.rebalanced.values
    if results.complete:
        growth_series["Complete"] = results.complete.values

    if growth_series:
        st.plotly_chart(
            charts.growth_chart(growth_series, results.config.capital),
            use_container_width=True,
        )

    # Outperformance
    if results.active and results.passive:
        st.plotly_chart(
            charts.outperformance_chart(results.active.values, results.passive.values),
            use_container_width=True,
        )

    # Summary stats table
    if results.active and results.passive:
        st.markdown("#### Performance Summary")
        summary_data = []
        for ps in [results.active, results.passive, results.orp, results.hrp, results.rebalanced, results.complete]:
            if ps is None:
                continue
            summary_data.append({
                "Portfolio": ps.name,
                "Ann. Return": f"{ps.ann_return:.2%}" if not np.isnan(ps.ann_return) else "—",
                "Ann. Volatility": f"{ps.ann_vol:.2%}" if not np.isnan(ps.ann_vol) else "—",
                "Sharpe": f"{ps.sharpe:.2f}" if not np.isnan(ps.sharpe) else "—",
                "Max Drawdown": f"{ps.max_dd:.2%}" if not np.isnan(ps.max_dd) else "—",
            })
        st.dataframe(pd.DataFrame(summary_data), hide_index=True, use_container_width=True)


# ──────────────────────────────────────────────────────────────
# TAB: Risk
# ──────────────────────────────────────────────────────────────

with tab_risk:
    # Interpretation card
    if results.interpretations and "risk" in results.interpretations:
        _interpretation_card(results.interpretations["risk"])

    # Drawdown chart
    dd_series = {}
    if results.active:
        dd_series["Active"] = results.active.values
    if results.passive:
        dd_series["Passive"] = results.passive.values
    if results.orp:
        dd_series["ORP"] = results.orp.values

    if dd_series:
        st.plotly_chart(
            charts.drawdown_chart(dd_series),
            use_container_width=True,
        )

    col_r1, col_r2 = st.columns(2)

    # VaR histogram
    with col_r1:
        if results.active:
            var95, cvar95 = T.var_cvar(results.active.daily_returns, 0.95)
            st.plotly_chart(
                charts.return_distribution_chart(
                    results.active.daily_returns,
                    var_95=var95,
                    cvar_95=cvar95,
                    title="Active Portfolio: Daily Returns",
                ),
                use_container_width=True,
            )

    # Correlation heatmap
    with col_r2:
        if results.correlation_matrix is not None:
            st.plotly_chart(
                charts.correlation_heatmap(results.correlation_matrix),
                use_container_width=True,
            )

    # Correlation regime detection
    if results.correlation_regime is not None and not results.correlation_regime.empty:
        st.plotly_chart(
            charts.correlation_regime_chart(results.correlation_regime),
            use_container_width=True,
        )

    # Rolling metrics
    if not results.monthly_returns.empty:
        asset_cols = [
            t for t in results.config.tickers
            if t in results.monthly_returns.columns
        ]
        if asset_cols and len(results.monthly_returns) >= 12:
            st.plotly_chart(
                charts.rolling_metrics_chart(
                    results.monthly_returns[asset_cols],
                    window=12,
                    rf_annual=results.config.risk_free_rate,
                ),
                use_container_width=True,
            )

    # Drawdown metrics table
    if results.drawdown_metrics is not None:
        st.markdown("#### Tail Risk Metrics")
        st.dataframe(results.drawdown_metrics, hide_index=True, use_container_width=True)

    # Detailed tail risk metrics
    if results.tail_risk:
        st.markdown("#### Extended Risk Statistics")
        tr = results.tail_risk
        col_t1, col_t2, col_t3, col_t4 = st.columns(4)
        col_t1.metric("Sortino Ratio", f"{tr.get('Sortino', np.nan):.2f}" if not np.isnan(tr.get('Sortino', np.nan)) else "N/A")
        col_t2.metric("Calmar Ratio", f"{tr.get('Calmar', np.nan):.2f}" if not np.isnan(tr.get('Calmar', np.nan)) else "N/A")
        col_t3.metric("Skewness", f"{tr.get('Skewness', np.nan):.3f}" if not np.isnan(tr.get('Skewness', np.nan)) else "N/A")
        col_t4.metric("Excess Kurtosis", f"{tr.get('Excess_Kurtosis', np.nan):.3f}" if not np.isnan(tr.get('Excess_Kurtosis', np.nan)) else "N/A")

        col_t5, col_t6, col_t7, col_t8 = st.columns(4)
        col_t5.metric("Worst Day", f"{tr.get('Worst_Day', np.nan):.2%}" if not np.isnan(tr.get('Worst_Day', np.nan)) else "N/A")
        col_t6.metric("Best Day", f"{tr.get('Best_Day', np.nan):.2%}" if not np.isnan(tr.get('Best_Day', np.nan)) else "N/A")
        col_t7.metric("Gain-to-Pain", f"{tr.get('Gain_to_Pain', np.nan):.2f}" if not np.isnan(tr.get('Gain_to_Pain', np.nan)) else "N/A")
        col_t8.metric("Max Drawdown", f"{tr.get('Max_Drawdown', np.nan):.2%}" if not np.isnan(tr.get('Max_Drawdown', np.nan)) else "N/A")

    # Stress testing
    if results.stress_df is not None and not results.stress_df.empty:
        st.markdown("#### Stress Testing: Historical Scenarios")
        st.plotly_chart(
            charts.stress_test_chart(results.stress_df),
            use_container_width=True,
        )
        with st.expander("Stress Test Details"):
            st.dataframe(results.stress_df, hide_index=True, use_container_width=True)


# ──────────────────────────────────────────────────────────────
# TAB: Attribution
# ──────────────────────────────────────────────────────────────

with tab_attribution:
    # Interpretation card
    if results.interpretations and "capm" in results.interpretations:
        _interpretation_card(results.interpretations["capm"])

    if results.asset_attribution is not None and not results.asset_attribution.empty:
        st.plotly_chart(
            charts.attribution_chart(
                results.asset_attribution,
                title="Brinson–Fachler Attribution (Assets)",
            ),
            use_container_width=True,
        )
        with st.expander("Asset Attribution Data"):
            st.dataframe(results.asset_attribution, use_container_width=True)
    else:
        st.info("Asset-level attribution data not available for this run.")

    if results.sector_attribution is not None and not results.sector_attribution.empty:
        st.plotly_chart(
            charts.attribution_chart(
                results.sector_attribution,
                title="Brinson–Fachler Attribution (Sectors)",
            ),
            use_container_width=True,
        )
        with st.expander("Sector Attribution Data"):
            st.dataframe(results.sector_attribution, use_container_width=True)

    # Sector & Factor Exposure
    if results.sector_weights is not None or results.factor_tilts is not None:
        st.markdown("#### Sector & Factor Exposure")
        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            if results.sector_weights is not None and not results.sector_weights.empty:
                st.plotly_chart(
                    charts.sector_donut_chart(results.sector_weights),
                    use_container_width=True,
                )
                from src.analytics.exposure import effective_n_sectors
                eff_n = effective_n_sectors(results.sector_weights)
                st.metric("Effective N Sectors", f"{eff_n:.1f}")
        with col_exp2:
            if results.factor_tilts is not None and not results.factor_tilts.empty:
                st.plotly_chart(
                    charts.factor_tilts_chart(results.factor_tilts),
                    use_container_width=True,
                )

    # CAPM results
    if results.capm_results:
        st.markdown("#### CAPM Regression Results")
        capm_df = pd.DataFrame([
            {
                "Asset": r.ticker,
                "Alpha (monthly)": f"{r.alpha:.4f}",
                "Beta": f"{r.beta:.2f}",
                "t(α)": f"{r.t_alpha:.2f}",
                "t(β)": f"{r.t_beta:.2f}",
                "R²": f"{r.r_squared:.3f}",
            }
            for r in results.capm_results
        ])
        st.dataframe(capm_df, hide_index=True, use_container_width=True)

        # Individual CAPM scatters
        if len(results.capm_results) <= 12:
            n_cols = min(3, len(results.capm_results))
            cols = st.columns(n_cols)
            rets_m = results.monthly_returns
            rf_m = (1 + results.config.risk_free_rate) ** (1/12) - 1
            benchmark = results.config.benchmark

            for i, r in enumerate(results.capm_results):
                if r.ticker in rets_m.columns and benchmark in rets_m.columns:
                    df_aligned = pd.concat(
                        [rets_m[r.ticker], rets_m[benchmark]], axis=1
                    ).dropna()
                    if not df_aligned.empty:
                        with cols[i % n_cols]:
                            fig = charts.capm_scatter(
                                r.ticker,
                                df_aligned.iloc[:, 0] - rf_m,
                                df_aligned.iloc[:, 1] - rf_m,
                                r.alpha, r.beta,
                            )
                            fig.update_layout(height=320)
                            st.plotly_chart(fig, use_container_width=True)

    # Factor regressions
    factor_models = {
        "FF3": "factor_regression_ff3.csv",
        "Carhart 4-Factor": "factor_regression_carhart4.csv",
        "FF5": "factor_regression_ff5.csv",
        "Quality & Low-Vol": "factor_regression_quality_lowvol.csv",
    }
    for model_name, fname in factor_models.items():
        fpath = Path("outputs") / fname
        if fpath.exists():
            try:
                df = pd.read_csv(fpath)
                st.markdown(f"#### {model_name} Factor Regression")
                st.plotly_chart(
                    charts.factor_loadings_chart(df, model_name),
                    use_container_width=True,
                )
                with st.expander(f"{model_name} Data"):
                    st.dataframe(df, use_container_width=True)
            except Exception:
                pass


# ──────────────────────────────────────────────────────────────
# TAB: Income
# ──────────────────────────────────────────────────────────────

with tab_income:
    # Interpretation card
    if results.interpretations and "income" in results.interpretations:
        _interpretation_card(results.interpretations["income"])

    if results.income_metrics is not None:
        im = results.income_metrics
        col_i1, col_i2, col_i3, col_i4 = st.columns(4)
        col_i1.metric("Annual Income", f"${im.get('total_annual_income', 0):,.2f}")
        col_i2.metric("Portfolio Yield", f"{im.get('portfolio_yield', 0):.2%}")
        col_i3.metric("Avg Yield on Cost", f"{im.get('avg_yield_on_cost', 0):.2%}")
        col_i4.metric("Dividend Payers", f"{im.get('n_payers', 0)}")

    if results.income_summary is not None and not results.income_summary.empty:
        col_ic1, col_ic2 = st.columns(2)
        with col_ic1:
            st.plotly_chart(
                charts.income_bar_chart(results.income_summary),
                use_container_width=True,
            )
        with col_ic2:
            if results.cumulative_income is not None and not results.cumulative_income.empty:
                st.plotly_chart(
                    charts.cumulative_income_chart(results.cumulative_income),
                    use_container_width=True,
                )

        st.markdown("#### Dividend Income by Position")
        display_cols = ["Ticker", "Shares", "AnnualDividendPerShare", "AnnualIncome", "YieldOnCost", "CurrentYield", "IncomeGrowthRate"]
        available_cols = [c for c in display_cols if c in results.income_summary.columns]
        st.dataframe(results.income_summary[available_cols], hide_index=True, use_container_width=True)
    else:
        st.info("Dividend income data not available. Run analysis to compute income analytics.")


# ──────────────────────────────────────────────────────────────
# TAB: Optimization
# ──────────────────────────────────────────────────────────────

with tab_optimization:
    # Interpretation card
    if results.interpretations and "optimization" in results.interpretations:
        _interpretation_card(results.interpretations["optimization"])

    orp_opt = results.orp_optimization

    if orp_opt is not None:
        # Efficient frontier
        # Get individual asset stats for plotting
        rets_m = results.monthly_returns
        asset_cols = [t for t in results.config.tickers if t in rets_m.columns and t != results.config.benchmark]
        asset_mu = ((1 + rets_m[asset_cols].mean()) ** 12 - 1) if asset_cols else None
        asset_vol = (rets_m[asset_cols].std() * np.sqrt(12)) if asset_cols else None

        st.plotly_chart(
            charts.efficient_frontier_chart(
                orp_opt.frontier_vols,
                orp_opt.frontier_returns,
                orp_opt.expected_vol,
                orp_opt.expected_return,
                results.config.risk_free_rate,
                asset_vols=asset_vol,
                asset_returns=asset_mu,
            ),
            use_container_width=True,
        )

        # Weights comparison
        col_o1, col_o2 = st.columns(2)

        with col_o1:
            st.plotly_chart(
                charts.weights_bar(orp_opt.weights, "ORP Weights (Max-Sharpe)"),
                use_container_width=True,
            )

        with col_o2:
            # Complete portfolio donut
            if results.config.include_complete:
                cp_weights = {
                    t: float(results.config.complete_portfolio.y) * float(w)
                    for t, w in orp_opt.weights.items()
                    if abs(w) > 1e-6
                }
                cp_weights["Risk-Free"] = 1.0 - results.config.complete_portfolio.y
                st.plotly_chart(
                    charts.allocation_donut(
                        cp_weights,
                        f"Complete Portfolio (y={results.config.complete_portfolio.y:.0%})",
                    ),
                    use_container_width=True,
                )

        # HRP weights side-by-side with ORP
        if results.hrp_weights is not None:
            col_h1, col_h2 = st.columns(2)
            with col_h1:
                st.plotly_chart(
                    charts.weights_bar(results.hrp_weights, "HRP Weights"),
                    use_container_width=True,
                )
            with col_h2:
                if results.hrp_linkage is not None:
                    asset_cols = [t for t in results.config.tickers if t in results.monthly_returns.columns and t != results.config.benchmark]
                    st.plotly_chart(
                        charts.dendrogram_chart(results.hrp_linkage, asset_cols),
                        use_container_width=True,
                    )

        # Concentration metrics
        from src.analytics.risk import herfindahl_index, effective_n_bets, concentration_ratio
        st.markdown("#### Concentration Metrics")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown("**Active Portfolio**")
            if results.holdings is not None:
                active_w = results.holdings["RealizedWeight"].values
                c1, c2, c3 = st.columns(3)
                c1.metric("HHI", f"{herfindahl_index(active_w):.4f}")
                c2.metric("Eff. N Bets", f"{effective_n_bets(active_w):.1f}")
                c3.metric("Top-3 Conc.", f"{concentration_ratio(active_w, 3):.2%}")
        with col_c2:
            st.markdown("**ORP**")
            if orp_opt is not None:
                orp_w = orp_opt.weights.values
                c4, c5, c6 = st.columns(3)
                c4.metric("HHI", f"{herfindahl_index(orp_w):.4f}")
                c5.metric("Eff. N Bets", f"{effective_n_bets(orp_w):.1f}")
                c6.metric("Top-3 Conc.", f"{concentration_ratio(orp_w, 3):.2%}")

        # Weight drift chart
        if results.weight_drift is not None and not results.weight_drift.empty:
            st.plotly_chart(
                charts.weight_drift_chart(results.weight_drift),
                use_container_width=True,
            )

        # Rebalanced vs unrebalanced comparison
        if results.rebalanced is not None and results.active is not None:
            st.markdown("#### Rebalanced vs Buy-and-Hold")
            rebal_data = []
            for ps in [results.active, results.rebalanced]:
                rebal_data.append({
                    "Strategy": ps.name,
                    "Ann. Return": f"{ps.ann_return:.2%}" if not np.isnan(ps.ann_return) else "--",
                    "Ann. Volatility": f"{ps.ann_vol:.2%}" if not np.isnan(ps.ann_vol) else "--",
                    "Sharpe": f"{ps.sharpe:.2f}" if not np.isnan(ps.sharpe) else "--",
                    "Max Drawdown": f"{ps.max_dd:.2%}" if not np.isnan(ps.max_dd) else "--",
                })
            st.dataframe(pd.DataFrame(rebal_data), hide_index=True, use_container_width=True)

        # Turnover
        if results.turnover_table is not None and not results.turnover_table.empty:
            with st.expander("Quarterly Turnover"):
                st.dataframe(results.turnover_table, hide_index=True, use_container_width=True)

        # ORP stats
        st.markdown("#### Optimal Risky Portfolio Statistics")
        st.dataframe(pd.DataFrame([{
            "Expected Return": f"{orp_opt.expected_return:.2%}",
            "Expected Volatility": f"{orp_opt.expected_vol:.2%}",
            "Sharpe Ratio": f"{orp_opt.sharpe:.3f}",
        }]), hide_index=True, use_container_width=True)

        # Risk contribution
        if results.risk_contribution is not None:
            st.plotly_chart(
                charts.risk_contribution_chart(
                    results.risk_contribution,
                    "ORP Risk Contribution by Asset",
                ),
                use_container_width=True,
            )

    else:
        st.info("ORP optimization was not included in this run.")


# ──────────────────────────────────────────────────────────────
# TAB: Forecast
# ──────────────────────────────────────────────────────────────

with tab_forecast:
    # Interpretation card
    if results.interpretations and "simulation" in results.interpretations:
        _interpretation_card(results.interpretations["simulation"])

    if results.simulations:
        for sim in results.simulations:
            st.plotly_chart(
                charts.simulation_fan_chart(
                    starting_value=sim.starting_value,
                    paths=sim.paths,
                    horizon_days=sim.horizon_days,
                    method_name=sim.name,
                ),
                use_container_width=True,
            )

        # Simulation comparison table
        if results.simulation_summary is not None:
            st.markdown("#### Simulation Comparison")
            st.dataframe(results.simulation_summary, hide_index=True, use_container_width=True)

        # Probability metrics
        st.markdown("#### Probability Analysis")
        cols_sim = st.columns(len(results.simulations))
        for i, sim in enumerate(results.simulations):
            with cols_sim[i]:
                st.markdown(f"**{sim.name}**")
                st.metric("Expected Value", f"${sim.expected_value:,.0f}")
                st.metric("P(Loss)", f"{sim.prob_loss:.1%}")
                st.metric("Median (P50)", f"${sim.percentiles['P50']:,.0f}")
                st.metric("Worst Case (P5)", f"${sim.percentiles['P5']:,.0f}")
                st.metric("Best Case (P95)", f"${sim.percentiles['P95']:,.0f}")
    else:
        st.info("Monte Carlo simulations not available. Run analysis to generate forecasts.")

    # Historical stats as fallback
    if results.active:
        st.markdown("#### Active Portfolio: Historical Risk Statistics")
        rets = results.active.daily_returns
        gtp = T.gain_to_pain(rets)
        var95, cvar95 = T.var_cvar(rets, 0.95)

        col_f1, col_f2, col_f3 = st.columns(3)
        col_f1.metric("Gain-to-Pain", f"{gtp:.2f}" if gtp else "N/A")
        col_f2.metric("Daily VaR (95%)", f"{var95:.2%}" if not np.isnan(var95) else "N/A")
        col_f3.metric("Daily CVaR (95%)", f"{cvar95:.2%}" if not np.isnan(cvar95) else "N/A")


# ──────────────────────────────────────────────────────────────
# TAB: Data
# ──────────────────────────────────────────────────────────────

with tab_data:
    # Executive summary interpretation
    if results.interpretations and "executive_summary" in results.interpretations:
        _interpretation_card(results.interpretations["executive_summary"])

    # ── Report Downloads (prominent) ──
    st.markdown("#### Reports")
    col_rpt1, col_rpt2 = st.columns(2)

    # Build chart figures dict for report embedding
    _report_charts = {}
    try:
        growth_series_rpt = {}
        if results.active:
            growth_series_rpt["Active"] = results.active.values
        if results.passive:
            growth_series_rpt["Passive"] = results.passive.values
        if results.orp:
            growth_series_rpt["ORP"] = results.orp.values
        if growth_series_rpt:
            _report_charts["growth"] = charts.growth_chart(growth_series_rpt, results.config.capital)

        dd_rpt = {}
        if results.active:
            dd_rpt["Active"] = results.active.values
        if results.passive:
            dd_rpt["Passive"] = results.passive.values
        if dd_rpt:
            _report_charts["drawdown"] = charts.drawdown_chart(dd_rpt)

        if results.correlation_matrix is not None:
            _report_charts["correlation"] = charts.correlation_heatmap(results.correlation_matrix)

        orp_opt_rpt = results.orp_optimization
        if orp_opt_rpt:
            rets_m_rpt = results.monthly_returns
            asset_cols_rpt = [t for t in results.config.tickers if t in rets_m_rpt.columns and t != results.config.benchmark]
            asset_mu_rpt = ((1 + rets_m_rpt[asset_cols_rpt].mean()) ** 12 - 1) if asset_cols_rpt else None
            asset_vol_rpt = (rets_m_rpt[asset_cols_rpt].std() * np.sqrt(12)) if asset_cols_rpt else None
            _report_charts["frontier"] = charts.efficient_frontier_chart(
                orp_opt_rpt.frontier_vols, orp_opt_rpt.frontier_returns,
                orp_opt_rpt.expected_vol, orp_opt_rpt.expected_return,
                results.config.risk_free_rate,
                asset_vols=asset_vol_rpt, asset_returns=asset_mu_rpt,
            )
    except Exception:
        pass

    with col_rpt1:
        try:
            html_report = build_html_report(results, chart_figures=_report_charts)
            st.download_button(
                "Download HTML Report",
                data=html_report.encode("utf-8"),
                file_name="portfolio_report.html",
                mime="text/html",
                use_container_width=True,
                type="primary",
            )
        except Exception as e:
            st.caption(f"HTML report unavailable: {e}")

    with col_rpt2:
        try:
            pdf_bytes = build_pdf_report(results, chart_figures=_report_charts)
            st.download_button(
                "Download PDF Report",
                data=pdf_bytes,
                file_name="portfolio_report.pdf",
                mime="application/pdf",
                use_container_width=True,
                type="primary",
            )
        except Exception as e:
            st.caption(f"PDF report unavailable: {e}")

    st.markdown("---")

    # Holdings
    if results.holdings is not None:
        st.markdown("#### Holdings")
        st.dataframe(results.holdings, hide_index=True, use_container_width=True)

    # Config summary
    with st.expander("Run Configuration"):
        st.json(results.config.model_dump(mode="json"), expanded=False)

    # CSV downloads
    st.markdown("#### Data Exports")

    output_dir = Path("outputs")
    if output_dir.exists():
        all_files = sorted(output_dir.iterdir())
        data_files = [f for f in all_files if f.is_file()]

        if data_files:
            # ZIP of everything
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                for f in data_files:
                    zf.write(f, arcname=f.name)
            zip_buffer.seek(0)

            st.download_button(
                "Download All Outputs (ZIP)",
                data=zip_buffer,
                file_name="portfolio_analysis_outputs.zip",
                mime="application/zip",
                use_container_width=True,
            )

            # Individual files
            csv_files = [f for f in data_files if f.suffix == ".csv"]
            if csv_files:
                cols = st.columns(3)
                for i, f in enumerate(csv_files):
                    with cols[i % 3]:
                        with open(f, "rb") as fh:
                            st.download_button(
                                f"{f.name}",
                                data=fh.read(),
                                file_name=f.name,
                                mime="text/csv",
                                key=f"dl_{f.name}",
                            )
