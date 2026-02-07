"""
Portfolio Analyzer v2 - Streamlit App
Bloomberg-style dark theme with tabbed layout.
Calls src/pipeline.py directly (no subprocess).
"""

import io
import os
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.config.models import PortfolioConfig, BLConfig, BLView, CompletePortfolioConfig
from src.pipeline import AnalysisPipeline, AnalysisResults
from src.data import transforms as T

# ─────────────────────────────────────────────────────────────
# Page config (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Portfolio Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# Bloomberg-style dark theme CSS
# ─────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0B1120; }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #151D2E 0%, #1A2438 100%);
        border: 1px solid #2D3A50;
        border-radius: 12px;
        padding: 16px 20px;
    }
    div[data-testid="stMetric"] label {
        color: #94A3B8 !important;
        font-size: 0.85rem !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #F1F5F9 !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }

    /* Positive/negative delta colors */
    div[data-testid="stMetricDelta"] svg { display: none; }
    div[data-testid="stMetricDelta"] div {
        font-weight: 600 !important;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #0B1120;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #151D2E;
        border-radius: 8px;
        padding: 8px 16px;
        color: #94A3B8;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E3A5F;
        color: #3B82F6;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0D1526;
        border-right: 1px solid #1E293B;
    }

    /* Tables */
    .stDataFrame { border-radius: 8px; overflow: hidden; }

    /* Hide Streamlit branding */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

CONFIG_PATH = Path("config.json")
OUTPUT_DIR = Path("outputs")


# ─────────────────────────────────────────────────────────────
# Helper: format numbers
# ─────────────────────────────────────────────────────────────

def fmt_pct(val, decimals=2):
    """Format a decimal as percentage string."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"{val * 100:.{decimals}f}%"


def fmt_dollar(val, decimals=0):
    """Format a number as dollar string."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"${val:,.{decimals}f}"


def fmt_ratio(val, decimals=2):
    """Format a ratio."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"{val:.{decimals}f}"


# ─────────────────────────────────────────────────────────────
# Title
# ─────────────────────────────────────────────────────────────

st.markdown(
    '<h1 style="color: #F1F5F9; margin-bottom: 0;">📊 Portfolio Analyzer</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p style="color: #64748B; margin-top: 0;">Advanced portfolio optimization & risk analytics</p>',
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────
# Sidebar: Configuration
# ─────────────────────────────────────────────────────────────

# Load legacy config for defaults
legacy_cfg = {}
if CONFIG_PATH.exists():
    with open(CONFIG_PATH) as f:
        legacy_cfg = json.load(f)

ap = legacy_cfg.get("active_portfolio", {}) or {}

st.sidebar.markdown("## Configuration")

# -- Tickers --
st.sidebar.markdown("### Tickers")
default_tickers = legacy_cfg.get("tickers", ap.get("tickers", ["AAPL", "MSFT"]))
tickers_text = st.sidebar.text_area(
    "Enter tickers (one per line)",
    value="\n".join(default_tickers),
    height=120,
)
tickers = [t.strip().upper() for t in tickers_text.split("\n") if t.strip()]

# -- Weights --
st.sidebar.markdown("### Weights")
equal_weights = st.sidebar.checkbox("Equal Weights", value=True)

weights_dict = {}
if tickers:
    if equal_weights:
        w = round(1.0 / len(tickers), 6)
        weights_dict = {t: w for t in tickers}
        # Adjust last ticker to ensure sum = 1.0
        remainder = 1.0 - w * (len(tickers) - 1)
        weights_dict[tickers[-1]] = round(remainder, 6)
    else:
        default_w = round(1.0 / len(tickers), 4)
        for t in tickers:
            old_w = ap.get("weights", {}).get(t, default_w)
            weights_dict[t] = st.sidebar.number_input(
                f"{t} weight",
                min_value=-2.0,
                max_value=2.0,
                value=float(old_w),
                step=0.01,
                format="%.4f",
                key=f"weight_{t}",
            )

    wt_sum = sum(weights_dict.values())
    if abs(wt_sum - 1.0) > 0.01:
        st.sidebar.error(f"Weights sum to {wt_sum:.4f} (must be ~1.0)")

st.sidebar.markdown("---")

# -- Dates --
st.sidebar.markdown("### Date Range")
start_default = (
    legacy_cfg.get("start")
    or legacy_cfg.get("start_date")
    or ap.get("start_date")
    or "2020-01-01"
)
end_default = (
    legacy_cfg.get("end")
    or legacy_cfg.get("end_date")
    or ap.get("end_date")
    or "2025-12-31"
)

start_date = st.sidebar.date_input(
    "Start Date",
    pd.to_datetime(str(start_default)),
)
end_date = st.sidebar.date_input(
    "End Date",
    pd.to_datetime(str(end_default)),
)

# -- Capital & Risk-Free Rate --
st.sidebar.markdown("### Capital & Risk")
capital = st.sidebar.number_input(
    "Initial Capital ($)",
    min_value=1000,
    value=int(ap.get("capital", legacy_cfg.get("initial_capital", 1_000_000))),
    step=10000,
)

benchmark = st.sidebar.text_input(
    "Benchmark Ticker",
    value=legacy_cfg.get("benchmark") or legacy_cfg.get("passive_benchmark", "SPY"),
)

rf_rate = st.sidebar.number_input(
    "Risk-Free Rate (annual)",
    min_value=0.0,
    max_value=0.25,
    value=float(legacy_cfg.get("risk_free_rate", 0.04)),
    step=0.005,
    format="%.4f",
)

st.sidebar.markdown("---")

# -- Optimization Settings --
st.sidebar.markdown("### Optimization")
allow_shorts = st.sidebar.checkbox(
    "Allow Short Sales",
    value=bool(legacy_cfg.get("short_sales", False)),
)

max_bound = st.sidebar.slider(
    "Max Weight Bound",
    min_value=0.5,
    max_value=2.0,
    value=1.0,
    step=0.1,
)

y_cp = st.sidebar.slider(
    "Complete Portfolio: Risky %",
    min_value=0.0,
    max_value=1.0,
    value=float(
        legacy_cfg.get("complete_portfolio", {}).get(
            "y", legacy_cfg.get("y_cp", 0.8)
        )
    ),
    step=0.05,
)

st.sidebar.markdown("---")

# -- Black-Litterman --
st.sidebar.markdown("### Black-Litterman")
bl_cfg = legacy_cfg.get("black_litterman", {}) or {}
bl_enabled = st.sidebar.checkbox(
    "Enable BL Model",
    value=bool(bl_cfg.get("enabled", False)),
)

bl_tau = 0.05
bl_views_list = []

if bl_enabled:
    bl_tau = st.sidebar.number_input(
        "Tau (model uncertainty)",
        min_value=0.001,
        max_value=1.0,
        value=float(bl_cfg.get("tau", 0.05)),
        step=0.005,
        format="%.3f",
    )

    existing_views = bl_cfg.get("views", []) or []
    max_views = 3

    for i in range(max_views):
        prior = existing_views[i] if i < len(existing_views) else {}
        with st.sidebar.expander(f"View {i + 1}", expanded=bool(prior)):
            enabled_this = st.sidebar.checkbox(
                "Enable",
                value=bool(prior),
                key=f"bl_v_enable_{i}",
            )
            if not enabled_this:
                continue

            vtype = st.sidebar.selectbox(
                "Type",
                ["absolute", "relative"],
                index=0 if prior.get("type", "absolute") == "absolute" else 1,
                key=f"bl_v_type_{i}",
            )

            confidence = st.sidebar.selectbox(
                "Confidence",
                ["low", "medium", "high"],
                index=["low", "medium", "high"].index(
                    prior.get("confidence", "medium")
                ),
                key=f"bl_v_conf_{i}",
            )

            if vtype == "absolute" and tickers:
                asset = st.sidebar.selectbox(
                    "Asset",
                    tickers,
                    index=min(
                        tickers.index(prior["asset"]) if prior.get("asset") in tickers else 0,
                        len(tickers) - 1,
                    ),
                    key=f"bl_v_asset_{i}",
                )
                q = st.sidebar.number_input(
                    "Expected excess return",
                    value=float(prior.get("q", 0.0)),
                    step=0.01,
                    format="%.4f",
                    key=f"bl_v_q_{i}",
                )
                bl_views_list.append({
                    "type": "absolute",
                    "asset": asset,
                    "q": float(q),
                    "confidence": confidence,
                })
            elif vtype == "relative" and len(tickers) >= 2:
                asset_long = st.sidebar.selectbox(
                    "Long Asset",
                    tickers,
                    index=0,
                    key=f"bl_v_long_{i}",
                )
                asset_short = st.sidebar.selectbox(
                    "Short Asset",
                    tickers,
                    index=min(1, len(tickers) - 1),
                    key=f"bl_v_short_{i}",
                )
                q = st.sidebar.number_input(
                    "Expected outperformance",
                    value=float(prior.get("q", 0.0)),
                    step=0.01,
                    format="%.4f",
                    key=f"bl_v_qr_{i}",
                )
                bl_views_list.append({
                    "type": "relative",
                    "asset_long": asset_long,
                    "asset_short": asset_short,
                    "q": float(q),
                    "confidence": confidence,
                })

st.sidebar.markdown("---")

# ─────────────────────────────────────────────────────────────
# Build PortfolioConfig from sidebar inputs
# ─────────────────────────────────────────────────────────────


def build_config() -> PortfolioConfig:
    """Build a validated PortfolioConfig from the sidebar state."""
    bl_config = BLConfig(
        enabled=bl_enabled,
        tau=bl_tau,
        views=[BLView(**v) for v in bl_views_list] if bl_views_list else [],
    )

    return PortfolioConfig(
        tickers=tickers,
        weights=weights_dict,
        benchmark=benchmark.strip().upper(),
        start_date=str(start_date),
        end_date=str(end_date),
        capital=float(capital),
        risk_free_rate=float(rf_rate),
        short_sales=allow_shorts,
        max_weight_bound=max_bound,
        frontier_points=50,
        include_orp=True,
        include_complete=True,
        use_dividends=False,
        complete_portfolio=CompletePortfolioConfig(y=y_cp),
        black_litterman=bl_config,
    )


# ─────────────────────────────────────────────────────────────
# Run button
# ─────────────────────────────────────────────────────────────

run_clicked = st.sidebar.button("Run Analysis", type="primary", use_container_width=True)

# Save config button
if st.sidebar.button("Save Config", use_container_width=True):
    try:
        cfg = build_config()
        cfg.save(CONFIG_PATH)
        st.sidebar.success("Config saved")
    except Exception as e:
        st.sidebar.error(f"Invalid config: {e}")


# ─────────────────────────────────────────────────────────────
# Run pipeline
# ─────────────────────────────────────────────────────────────

if run_clicked:
    if not tickers:
        st.error("Enter at least one ticker in the sidebar.")
        st.stop()

    try:
        config = build_config()
    except Exception as e:
        st.error(f"Configuration error: {e}")
        st.stop()

    # Save config so legacy modules (attribution, factors) can read it
    config.save(CONFIG_PATH)

    # Clear old outputs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for f in OUTPUT_DIR.iterdir():
        if f.is_file():
            f.unlink()

    # Run pipeline with progress bar
    progress_bar = st.progress(0, text="Initializing...")

    def on_progress(label: str, fraction: float):
        progress_bar.progress(fraction, text=label)

    pipeline = AnalysisPipeline(config, output_dir=str(OUTPUT_DIR))
    results = pipeline.run(progress=on_progress)

    progress_bar.progress(1.0, text="Complete")

    # Store results in session state
    st.session_state["results"] = results
    st.session_state["run_complete"] = True
    st.rerun()


# ─────────────────────────────────────────────────────────────
# Display results (if available)
# ─────────────────────────────────────────────────────────────

if not st.session_state.get("run_complete"):
    st.info("Configure your portfolio in the sidebar and click **Run Analysis**.")
    st.stop()

results: AnalysisResults = st.session_state["results"]

# ─────────────────────────────────────────────────────────────
# Metric Cards Row
# ─────────────────────────────────────────────────────────────

st.markdown("---")

mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)

active = results.active
passive = results.passive
orp_opt = results.orp_optimization

with mc1:
    val = active.ann_return if active else None
    st.metric(
        "Total Return (Ann.)",
        fmt_pct(val),
        delta=fmt_pct(val - passive.ann_return) + " vs bench" if active and passive else None,
    )

with mc2:
    st.metric(
        "Annualized Vol",
        fmt_pct(active.ann_vol if active else None),
    )

with mc3:
    st.metric(
        "Sharpe Ratio",
        fmt_ratio(active.sharpe if active else None),
        delta=fmt_ratio(
            active.sharpe - passive.sharpe
        ) + " vs bench" if active and passive and not np.isnan(passive.sharpe) else None,
    )

with mc4:
    st.metric(
        "Max Drawdown",
        fmt_pct(active.max_dd if active else None),
    )

with mc5:
    if results.capm_results:
        avg_alpha = np.mean([r.alpha for r in results.capm_results])
        st.metric("Avg Alpha (monthly)", fmt_pct(avg_alpha, 4))
    else:
        st.metric("Avg Alpha", "N/A")

with mc6:
    if results.capm_results:
        avg_beta = np.mean([r.beta for r in results.capm_results])
        st.metric("Avg Beta", fmt_ratio(avg_beta))
    else:
        st.metric("Avg Beta", "N/A")


# ─────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────

tab_overview, tab_risk, tab_attribution, tab_optimization, tab_forecast, tab_data = st.tabs([
    "Overview",
    "Risk",
    "Attribution",
    "Optimization",
    "Forecast",
    "Data",
])


# ─────────────────────────────────────────────────────────────
# Helper: show an image from outputs if it exists
# ─────────────────────────────────────────────────────────────

def show_image(filename, caption=None):
    """Display an output image if it exists."""
    path = OUTPUT_DIR / filename
    if path.exists():
        st.image(str(path), caption=caption or filename, use_container_width=True)
        return True
    return False


# ═════════════════════════════════════════════════════════════
# OVERVIEW TAB
# ═════════════════════════════════════════════════════════════

with tab_overview:
    st.markdown("### Portfolio Growth")
    show_image("growth_active_vs_passive.png", "Growth: Active vs Passive")
    show_image("growth_all_portfolios.png", "Growth: All Portfolios")

    st.markdown("### Cumulative Outperformance")
    show_image("active_minus_passive_cumulative.png", "Active minus Passive (cumulative)")

    # Summary stats table
    if active and passive:
        st.markdown("### Performance Summary")
        summary_data = {
            "Metric": [
                "Annualized Return",
                "Annualized Volatility",
                "Sharpe Ratio",
                "Max Drawdown",
            ],
            "Active": [
                fmt_pct(active.ann_return),
                fmt_pct(active.ann_vol),
                fmt_ratio(active.sharpe),
                fmt_pct(active.max_dd),
            ],
            "Passive (Benchmark)": [
                fmt_pct(passive.ann_return),
                fmt_pct(passive.ann_vol),
                fmt_ratio(passive.sharpe),
                fmt_pct(passive.max_dd),
            ],
        }
        if results.orp:
            summary_data["ORP"] = [
                fmt_pct(results.orp.ann_return),
                fmt_pct(results.orp.ann_vol),
                fmt_ratio(results.orp.sharpe),
                fmt_pct(results.orp.max_dd),
            ]
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════════════════════
# RISK TAB
# ═════════════════════════════════════════════════════════════

with tab_risk:
    st.markdown("### Drawdown")
    show_image("drawdown_curves.png", "Portfolio Drawdown Curves")

    st.markdown("### Value-at-Risk Distribution")
    show_image("loss_histogram_active.png", "Daily Return Distribution with VaR/CVaR")

    # Risk metrics table
    if results.drawdown_metrics is not None and not results.drawdown_metrics.empty:
        st.markdown("### Risk Metrics")
        st.dataframe(results.drawdown_metrics, use_container_width=True, hide_index=True)

    st.markdown("### Rolling Risk Analytics")
    show_image("rolling_metrics.png", "Rolling Volatility, Sharpe, Beta")

    # Rolling correlation GIF
    corr_gif = OUTPUT_DIR / "rolling_corr_heatmap.gif"
    if corr_gif.exists():
        st.image(str(corr_gif), caption="Rolling Correlation Heatmap")

    st.markdown("### Correlation Matrix")
    show_image("correlation_matrix.png", "Asset Correlation Matrix")

    # CAPM scatter plots
    capm_pngs = sorted(OUTPUT_DIR.glob("capm_*.png"))
    if capm_pngs:
        st.markdown("### CAPM Scatter Plots")
        cols = st.columns(min(len(capm_pngs), 3))
        for idx, p in enumerate(capm_pngs):
            with cols[idx % 3]:
                st.image(str(p), caption=p.stem, use_container_width=True)

    # CAPM results table
    if results.capm_results:
        st.markdown("### CAPM Regression Results")
        capm_rows = [
            {
                "Ticker": r.ticker,
                "Alpha": f"{r.alpha:.6f}",
                "Beta": f"{r.beta:.4f}",
                "t(Alpha)": f"{r.t_alpha:.3f}",
                "t(Beta)": f"{r.t_beta:.3f}",
                "R-squared": f"{r.r_squared:.4f}",
            }
            for r in results.capm_results
        ]
        st.dataframe(pd.DataFrame(capm_rows), use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════════════════════
# ATTRIBUTION TAB
# ═════════════════════════════════════════════════════════════

with tab_attribution:
    st.markdown("### Asset-Level Attribution (Brinson-Fachler)")
    show_image("performance_attribution.png", "Allocation, Selection, Interaction Effects")

    if results.asset_attribution is not None and not results.asset_attribution.empty:
        st.dataframe(results.asset_attribution, use_container_width=True, hide_index=True)

    st.markdown("### Sector-Level Attribution")
    show_image("performance_attribution_sector.png", "Sector Attribution")

    if results.sector_attribution is not None and not results.sector_attribution.empty:
        st.dataframe(results.sector_attribution, use_container_width=True, hide_index=True)

    # Factor regressions
    factor_models = {
        "Fama-French 3-Factor": "factor_regression_ff3.csv",
        "Carhart 4-Factor": "factor_regression_carhart4.csv",
        "Fama-French 5-Factor": "factor_regression_ff5.csv",
        "Quality & Low-Vol": "factor_regression_quality_lowvol.csv",
    }

    has_factors = False
    for title, fname in factor_models.items():
        fpath = OUTPUT_DIR / fname
        if fpath.exists():
            if not has_factors:
                st.markdown("### Multi-Factor Regressions")
                has_factors = True
            st.markdown(f"**{title}**")
            try:
                st.dataframe(pd.read_csv(fpath), use_container_width=True, hide_index=True)
            except Exception:
                pass

    # Factor beta charts
    factor_chart_dir = OUTPUT_DIR / "factor_charts"
    if factor_chart_dir.exists():
        factor_pngs = sorted(factor_chart_dir.rglob("*.png"))
        if factor_pngs:
            st.markdown("### Factor Beta Charts")
            cols = st.columns(min(len(factor_pngs), 3))
            for idx, p in enumerate(factor_pngs):
                with cols[idx % 3]:
                    st.image(str(p), caption=p.stem, use_container_width=True)


# ═════════════════════════════════════════════════════════════
# OPTIMIZATION TAB
# ═════════════════════════════════════════════════════════════

with tab_optimization:
    st.markdown("### Efficient Frontier")
    show_image("efficient_frontier.png", "Mean-Variance Efficient Frontier with ORP")

    st.markdown("### Capital Allocation Line")
    show_image("CAL.png", "Capital Allocation Line")

    # ORP Weights
    if orp_opt is not None:
        st.markdown("### ORP Weights (Max-Sharpe)")
        orp_w = orp_opt.weights.copy()
        orp_df = pd.DataFrame({
            "Ticker": orp_w.index,
            "Weight": orp_w.values,
            "Weight %": (orp_w.values * 100).round(2),
        }).sort_values("Weight", ascending=False)
        st.dataframe(orp_df, use_container_width=True, hide_index=True)

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("ORP Expected Return", fmt_pct(orp_opt.expected_return))
        with col_b:
            st.metric("ORP Expected Vol", fmt_pct(orp_opt.expected_vol))
        with col_c:
            st.metric("ORP Sharpe", fmt_ratio(orp_opt.sharpe))

    show_image("orp_real_vs_expected.png", "ORP: Realized vs Expected Return")

    # HRP
    st.markdown("### Hierarchical Risk Parity")
    show_image("hrp_cluster_tree.png", "HRP Cluster Dendrogram")

    hrp_path = OUTPUT_DIR / "hrp_weights.csv"
    if hrp_path.exists():
        try:
            hrp_df = pd.read_csv(hrp_path, index_col=0)
            st.dataframe(hrp_df, use_container_width=True)
        except Exception:
            pass

    # Black-Litterman
    st.markdown("### Black-Litterman")
    show_image("black_litterman_efficient_frontier.png", "BL Efficient Frontier")

    bl_path = OUTPUT_DIR / "black_litterman_weights.csv"
    if bl_path.exists():
        try:
            bl_df = pd.read_csv(bl_path, index_col=0)
            st.markdown("**BL Optimal Weights**")
            st.dataframe(bl_df, use_container_width=True)
        except Exception:
            pass

    # Complete Portfolio
    st.markdown("### Complete Portfolio")
    show_image("complete_portfolio_pie.png", "Complete Portfolio Allocation")


# ═════════════════════════════════════════════════════════════
# FORECAST TAB
# ═════════════════════════════════════════════════════════════

with tab_forecast:
    st.markdown("### Monte Carlo Forward Simulation")
    show_image("forward_scenarios.png", "Forward Scenarios (Monte Carlo)")

    # Style analysis
    style_path = OUTPUT_DIR / "style_regression_summary.csv"
    if style_path.exists():
        st.markdown("### Style Analysis")
        try:
            st.dataframe(pd.read_csv(style_path), use_container_width=True, hide_index=True)
        except Exception:
            pass


# ═════════════════════════════════════════════════════════════
# DATA TAB
# ═════════════════════════════════════════════════════════════

with tab_data:
    # Holdings
    if results.holdings is not None and not results.holdings.empty:
        st.markdown("### Holdings")
        st.dataframe(results.holdings, use_container_width=True, hide_index=True)

    # Summary JSON
    summary_path = OUTPUT_DIR / "summary.json"
    if summary_path.exists():
        st.markdown("### Summary Metrics")
        with open(summary_path) as f:
            summary = json.load(f)
        st.json(summary)

    # CSV Downloads
    st.markdown("### Download Data Files")
    csv_files = sorted(OUTPUT_DIR.glob("*.csv"))
    if csv_files:
        for csv_file in csv_files:
            with open(csv_file, "rb") as f:
                st.download_button(
                    label=f"Download {csv_file.name}",
                    data=f.read(),
                    file_name=csv_file.name,
                    mime="text/csv",
                    key=f"dl_{csv_file.name}",
                )

    # Reports
    st.markdown("### Reports")
    for ext in ["*.md", "*.pdf"]:
        for report_file in sorted(OUTPUT_DIR.glob(ext)):
            mime = "text/markdown" if report_file.suffix == ".md" else "application/pdf"
            with open(report_file, "rb") as f:
                st.download_button(
                    label=f"Download {report_file.name}",
                    data=f.read(),
                    file_name=report_file.name,
                    mime=mime,
                    key=f"dl_{report_file.name}",
                )

    # ZIP of everything
    st.markdown("### Download All Outputs")
    all_output_files = [f for f in OUTPUT_DIR.iterdir() if f.is_file()]
    if all_output_files:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in all_output_files:
                zf.write(f, arcname=f.name)
        zip_buffer.seek(0)

        st.download_button(
            label="Download All Outputs (ZIP)",
            data=zip_buffer,
            file_name="portfolio_analysis_outputs.zip",
            mime="application/zip",
            key="dl_all_zip",
        )
