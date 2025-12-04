import streamlit as st
import json
import subprocess
import os
import sys
from datetime import datetime
import io
import zipfile
import pandas as pd

CONFIG_PATH = "config.json"
OUTPUT_DIR = "outputs"


def load_config():
    if not os.path.exists(CONFIG_PATH):
        return {}
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


def save_config(cfg):
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=4)
    return True


def clear_output_dir():
    if os.path.exists(OUTPUT_DIR):
        for fname in os.listdir(OUTPUT_DIR):
            fpath = os.path.join(OUTPUT_DIR, fname)
            if os.path.isfile(fpath):
                os.remove(fpath)


st.set_page_config(page_title="Portfolio Analyzer", layout="wide")
st.title("Portfolio Analyzer App")

cfg = load_config()
ap = cfg.get("active_portfolio", {}) or {}

# -------------------------
# Session state for run status
# -------------------------
if "run_status" not in st.session_state:
    st.session_state["run_status"] = None
if "run_error" not in st.session_state:
    st.session_state["run_error"] = None

tickers: list[str] = []

# =========================
# Sidebar: Global Settings
# =========================
st.sidebar.header("⚙️ Global Settings")

initial_capital = st.sidebar.number_input(
    "Initial Capital",
    min_value=1000,
    value=int(ap.get("capital", cfg.get("initial_capital", 1_000_000))),
)

start_default = (
    cfg.get("start")
    or cfg.get("start_date")
    or ap.get("start_date")
    or "2020-01-01"
)
end_default = (
    cfg.get("end")
    or cfg.get("end_date")
    or ap.get("end_date")
    or "2025-12-31"
)

start_date = st.sidebar.date_input(
    "Start Date", datetime.fromisoformat(str(start_default))
)
end_date = st.sidebar.date_input(
    "End Date", datetime.fromisoformat(str(end_default))
)

benchmark = st.sidebar.text_input(
    "Benchmark Ticker",
    value=cfg.get("benchmark") or cfg.get("passive_benchmark", "SPY"),
)

rf_rate = st.sidebar.number_input(
    "Risk-free rate (annual, e.g. 0.04 for 4%)",
    value=float(cfg.get("risk_free_rate", 0.04)),
    step=0.005,
    format="%.4f",
)

include_orp = st.sidebar.checkbox(
    "Include ORP (Optimal Risky Portfolio)",
    value=cfg.get("include_orp", True),
)

include_complete = st.sidebar.checkbox(
    "Include Complete Portfolio (ORP + Treasuries)",
    value=cfg.get("include_complete", True),
)

y_cp = st.sidebar.slider(
    "Complete Portfolio: ORP % (Rest goes to Treasuries)",
    min_value=0.0,
    max_value=1.0,
    value=float(cfg.get("complete_portfolio", {}).get("y", cfg.get("y_cp", 0.8))),
)

# -------------------------
# Black-Litterman (optional, new)
# -------------------------
bl_cfg = cfg.get("black_litterman", {}) or {}
bl_enabled = st.sidebar.checkbox(
    "Enable Black-Litterman Model",
    value=bool(bl_cfg.get("enabled", False)),
)
if bl_enabled:
    tau = st.sidebar.number_input(
        "Black-Litterman τ (model uncertainty)",
        min_value=0.001,
        max_value=1.0,
        value=float(bl_cfg.get("tau", 0.05)),
        step=0.005,
        format="%.3f",
        help="Controls how much weight is placed on the prior vs your views. "
        "Smaller τ = more weight on views.",
    )
else:
    # Keep previous tau as default so you don't lose it when disabling
    tau = float(bl_cfg.get("tau", 0.05))

# ------------------------------------------------------------------
# PER-ASSET BOUND DEFAULTS
# ------------------------------------------------------------------
DEFAULT_MAX_BOUND = 1.0

# =========================
# Active Portfolio
# =========================
st.subheader("📈 Active Portfolio")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### Tickers (one per line)")
    # Default: EMPTY textarea so each new session/user starts fresh
    tickers_text = st.text_area(
        "Tickers (one per line)",
        value="",
        height=200,
        label_visibility="collapsed",
    )

tickers = [t.strip().upper() for t in tickers_text.split("\n") if t.strip()]

with col2:
    # spacer so checkboxes visually align with text box
    st.markdown("<br><br>", unsafe_allow_html=True)

    # Default: equal weights ON for a fresh session
    equal_weights = st.checkbox(
        "Use Equal Weights",
        value=True,
    )
    use_dividends = st.checkbox(
        "Use Dividends",
        value=ap.get("use_dividends", False),
    )

    # Allow Shorting + per-asset bound
    allow_shorts_initial = ap.get("allow_shorts", cfg.get("short_sales", False))
    allow_shorts = st.checkbox(
        "Allow Shorting",
        value=allow_shorts_initial,
    )

    max_bound = st.slider(
        "Per-asset weight bound (|wᵢ| ≤ ...)",
        min_value=0.5,
        max_value=1.5,  # upper limit the user can choose
        value=DEFAULT_MAX_BOUND,  # always start at 1.0 on app load
        step=0.1,
        help=(
            "Sets the maximum absolute position per asset.\n\n"
            "• 1.0 = no asset can exceed 100% of portfolio value.\n"
            "• Values above 1.0 allow leverage, where a single asset "
            "can be larger than the portfolio (long or short)."
        ),
    )

    # If user pushes bound above 1.0, show a gentle leverage warning
    if max_bound > 1.0:
        st.warning(
            "Per-asset bound is above 1.0 — you’re now allowing **leverage**.\n\n"
            "This means a single position can be larger than your total capital "
            "(or a short position can be more than 100% of the portfolio). "
            "Make sure this is intentional."
        )

# =========================
# Weights
# =========================
weights_list: list[float] = []

if equal_weights:
    if tickers:
        weights_list = [1.0 / len(tickers)] * len(tickers)
else:
    st.markdown("### Weights")

    weight_cols = st.columns(3)

    # Start custom weights at equal-weight by default
    default_weights = [1.0 / len(tickers)] * len(tickers) if tickers else []

    for idx, t in enumerate(tickers):
        col = weight_cols[idx % 3]

        if idx < len(default_weights):
            default_val = float(default_weights[idx])
        else:
            default_val = 1.0 / len(tickers) if tickers else 0.0

        w = col.number_input(
            f"{t} weight",
            min_value=-max_bound if allow_shorts else 0.0,
            max_value=max_bound,
            value=default_val,
            step=0.01,
            format="%.4f",
        )
        weights_list.append(float(w))

# Basic guard: if no tickers, do nothing further
if not tickers:
    st.warning("Please enter at least one ticker.")
    st.stop()

if equal_weights and not weights_list:
    weights_list = [1.0 / len(tickers)] * len(tickers)

weights_dict = {t: w for t, w in zip(tickers, weights_list)}

# ----- Enforce custom weights sum exactly to 1 with red popup -----
weights_sum = sum(weights_list) if weights_list else 0.0
weights_invalid = (not equal_weights) and (abs(weights_sum - 1.0) > 1e-6)

if not equal_weights:
    st.markdown(
        f"**Total custom weight (must equal 1.0000):** {weights_sum:.4f}"
    )
    if weights_invalid:
        st.error(
            "Custom weights must sum to 1.0000. "
            f"Current total is {weights_sum:.4f}. "
            "Please adjust one or more weights."
        )

# ----- Leverage health bar (gross exposure) -----
# Gross exposure = sum of absolute weights. 1.0x means fully invested, no leverage.
gross_exposure = sum(abs(w) for w in weights_list) if weights_list else 0.0

st.markdown("### Leverage Health")
st.markdown(f"**Gross exposure:** {gross_exposure:.2f}×")

# Color-coded health messages
if gross_exposure <= 1.0 + 1e-6:
    st.success("Conservative: gross exposure ≤ 1.0× (no leverage).")
elif gross_exposure <= 1.5 + 1e-6:
    st.info(
        "Aggressive: gross exposure between 1.0× and 1.5×. "
        "You are using some leverage; expect higher volatility."
    )
elif gross_exposure <= 2.0 + 1e-6:
    st.warning(
        "High leverage: gross exposure between 1.5× and 2.0×. "
        "Large swings and deep drawdowns are likely."
    )
else:
    st.error(
        "Extreme leverage: gross exposure above 2.0×. "
        "Portfolios at this level can blow up quickly; proceed with caution."
    )

# =========================
# Black-Litterman Views (optional, new)
# =========================
bl_views: list[dict] = []

if bl_enabled:
    st.subheader("🧠 Black-Litterman Views (optional)")

    if not tickers:
        st.info("Enter tickers above to define Black-Litterman views.")
    else:
        existing_views = []
        if isinstance(bl_cfg, dict):
            existing_views = bl_cfg.get("views", []) or []

        max_views = 3  # simple V1: up to 3 views
        conf_index_map = {"low": 0, "medium": 1, "high": 2}

        for i in range(max_views):
            prior = existing_views[i] if i < len(existing_views) else {}

            with st.expander(f"View {i + 1}", expanded=(i == 0 and not existing_views)):
                enabled_this = st.checkbox(
                    "Enable this view",
                    value=bool(prior),
                    key=f"bl_view_enable_{i}",
                )
                if not enabled_this:
                    continue

                # View type
                prior_type = str(prior.get("type", "absolute")).lower()
                vtype = st.selectbox(
                    "View type",
                    options=["absolute", "relative"],
                    index=0 if prior_type == "absolute" else 1,
                    key=f"bl_view_type_{i}",
                )

                # Confidence
                prior_conf = str(prior.get("confidence", "medium")).lower()
                conf_idx = conf_index_map.get(prior_conf, 1)
                confidence = st.selectbox(
                    "Confidence",
                    options=["low", "medium", "high"],
                    index=conf_idx,
                    key=f"bl_view_conf_{i}",
                )

                if vtype == "absolute":
                    # Absolute: pick one asset and an expected annual excess return
                    default_asset = prior.get("asset", tickers[0])
                    try:
                        default_idx = tickers.index(default_asset)
                    except ValueError:
                        default_idx = 0

                    asset = st.selectbox(
                        "Asset",
                        options=tickers,
                        index=default_idx,
                        key=f"bl_view_asset_{i}",
                    )

                    q_val = float(prior.get("q", 0.0))
                    q = st.number_input(
                        "Expected annual excess return (decimal, e.g. 0.08 = 8%)",
                        value=q_val,
                        step=0.01,
                        format="%.4f",
                        key=f"bl_view_q_abs_{i}",
                    )

                    bl_views.append(
                        {
                            "type": "absolute",
                            "asset": asset,
                            "q": float(q),
                            "confidence": confidence,
                        }
                    )
                else:
                    # Relative: long vs short
                    default_long = prior.get("asset_long", tickers[0])
                    try:
                        long_idx = tickers.index(default_long)
                    except ValueError:
                        long_idx = 0

                    default_short = prior.get(
                        "asset_short",
                        tickers[1] if len(tickers) > 1 else tickers[0],
                    )
                    try:
                        short_idx = tickers.index(default_short)
                    except ValueError:
                        short_idx = 1 if len(tickers) > 1 else 0

                    asset_long = st.selectbox(
                        "Asset (long)",
                        options=tickers,
                        index=long_idx,
                        key=f"bl_view_long_{i}",
                    )
                    asset_short = st.selectbox(
                        "Asset (short)",
                        options=tickers,
                        index=short_idx,
                        key=f"bl_view_short_{i}",
                    )

                    q_val = float(prior.get("q", 0.0))
                    q = st.number_input(
                        "Expected outperformance of long vs short (decimal, e.g. 0.03 = 3%)",
                        value=q_val,
                        step=0.01,
                        format="%.4f",
                        key=f"bl_view_q_rel_{i}",
                    )

                    bl_views.append(
                        {
                            "type": "relative",
                            "asset_long": asset_long,
                            "asset_short": asset_short,
                            "q": float(q),
                            "confidence": confidence,
                        }
                    )

# =========================
# Build config.json
# =========================
cfg["tickers"] = tickers
cfg["benchmark"] = benchmark

cfg["start"] = str(start_date)
cfg["end"] = str(end_date)

cfg["risk_free_rate"] = float(rf_rate)

# Shorting + bounds logic (uses current slider value)
cfg["short_sales"] = bool(allow_shorts)
if allow_shorts:
    cfg["max_allocation_bounds"] = [-max_bound, max_bound]
else:
    cfg["max_allocation_bounds"] = [0.0, max_bound]

cfg["frontier_points"] = int(cfg.get("frontier_points", 50))

cfg["include_orp"] = include_orp
cfg["include_complete"] = include_complete

ap["tickers"] = tickers
ap["weights"] = weights_dict
ap["use_dividends"] = use_dividends
ap["allow_shorts"] = allow_shorts
ap["capital"] = float(initial_capital)
ap["start_date"] = str(start_date)
ap["end_date"] = str(end_date)
cfg["active_portfolio"] = ap
cfg["tickers"] = tickers  # keep top-level tickers in sync

cfg["passive_portfolio"] = {
    "capital": float(initial_capital),
    "start_date": str(start_date),
}

cfg["complete_portfolio"] = {"y": float(y_cp)}

# --- Black-Litterman config wiring (new) ---
bl_cfg_out = {"enabled": bool(bl_enabled)}
if bl_enabled and tickers:
    bl_cfg_out["tau"] = float(tau)
    bl_cfg_out["views"] = bl_views
cfg["black_litterman"] = bl_cfg_out

# -------------------------
# Run button & status
# -------------------------
status = st.session_state.get("run_status")
error_msg = st.session_state.get("run_error")

if st.button("💾 Save Config 💾"):
    if weights_invalid:
        st.error(
            "Cannot save config: custom weights must sum to exactly 1. "
            "Please adjust them first."
        )
    else:
        save_config(cfg)
        st.success("Configuration saved to config.json")

st.markdown("## 🏃 Run Analysis")

if st.button("▶️ Run Portfolio Analysis ▶️"):
    if weights_invalid:
        st.error(
            "Cannot run analysis: custom weights must sum to exactly 1. "
            "Please adjust them first."
        )
    else:
        st.session_state["run_status"] = "running"
        st.session_state["run_error"] = None

        clear_output_dir()

        python_exec = sys.executable

        with st.spinner("Running portfolio analysis... this may take a moment ⏳"):
            result = subprocess.run(
                [python_exec, "main.py", "--config", "config.json"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=".",
            )

        if result.returncode != 0:
            st.session_state["run_status"] = "error"
            st.session_state["run_error"] = result.stderr
        else:
            st.session_state["run_status"] = "ok"

status = st.session_state.get("run_status")
error_msg = st.session_state.get("run_error")

if status == "running":
    st.info("Running analysis… please wait…")
elif status == "ok":
    st.success("Analysis complete!")
elif status == "error":
    st.error(f"Error running analyzer:\n\n{error_msg}")

# =========================
# Outputs & Stats (ordered)
# =========================
if os.path.exists(OUTPUT_DIR) and st.session_state.get("run_status") == "ok":
    all_files = sorted(os.listdir(OUTPUT_DIR))
    st.write("### Output Files")

    # ---------- ZIP of everything ----------
    if all_files:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in all_files:
                full_path = os.path.join(OUTPUT_DIR, f)
                if os.path.isfile(full_path):
                    zf.write(full_path, arcname=f)
        zip_buffer.seek(0)

        st.download_button(
            label="⬇️ Download all outputs as ZIP ⬇️",
            data=zip_buffer,
            file_name="outputs.zip",
            mime="application/zip",
            key="download_all_zip",
        )

    # ---------- Categorize files ----------
    reports = []
    priority_pngs = {}
    capm_pngs = []
    other_pngs = []
    csv_files = []
    other_files = []
    gif_files = []

    rolling_metrics_path = None
    rolling_corr_name = None
    rolling_corr_path = None
    drawdown_curves_path = None
    loss_histogram_path = None

    attribution_png_path = None
    attribution_csv_path = None
    sector_attribution_png_path = None
    sector_attribution_csv_path = None

    # NOTE: Black-Litterman frontier inserted right after ORP frontier
    priority_order = [
        "correlation_matrix.png",
        "efficient_frontier.png",
        "black_litterman_efficient_frontier.png",
        "cal.png",
        "orp_real_vs_expected.png",
        "complete_portfolio_pie.png",
        "growth_all_portfolios.png",
        "forward_scenarios.png",
    ]

    for f in all_files:
        full_path = os.path.join(OUTPUT_DIR, f)
        if not os.path.isfile(full_path):
            continue

        lower = f.lower()

        if lower.endswith(".png"):
            if lower in priority_order:
                priority_pngs[lower] = full_path
            elif lower == "rolling_metrics.png":
                rolling_metrics_path = full_path
            elif lower == "drawdown_curves.png":
                drawdown_curves_path = full_path
            elif lower == "loss_histogram_active.png":
                loss_histogram_path = full_path
            elif lower == "performance_attribution.png":
                attribution_png_path = full_path
            elif lower == "performance_attribution_sector.png":
                sector_attribution_png_path = full_path
            elif lower.startswith("capm_"):
                capm_pngs.append((f, full_path))
            else:
                other_pngs.append((f, full_path))
        elif lower.endswith(".gif"):
            gif_files.append((f, full_path))
        elif lower.endswith(".csv"):
            if lower == "performance_attribution.csv":
                attribution_csv_path = full_path
                csv_files.append((f, full_path))
            elif lower == "performance_attribution_sector.csv":
                sector_attribution_csv_path = full_path
                csv_files.append((f, full_path))
            else:
                csv_files.append((f, full_path))
        elif lower.endswith(".pdf") or lower.endswith(".md"):
            if "report" in lower:
                reports.append((f, full_path))
            else:
                other_files.append((f, full_path))
        else:
            other_files.append((f, full_path))

    def pop_gif_by_name(target_name: str):
        target_name = target_name.lower()
        for idx, (gf, gpath) in enumerate(gif_files):
            if gf.lower() == target_name:
                gif_files.pop(idx)
                return gf, gpath
        return None, None

    for candidate in ["rolling_corr_heatmap.gif", "rolling_cor_heatmap.gif"]:
        gf, gpath = pop_gif_by_name(candidate)
        if gpath:
            rolling_corr_name, rolling_corr_path = gf, gpath
            break

    # 1) Reports
    if reports:
        st.write("### Report Files")
        for f, full_path in reports:
            if f.lower().endswith(".pdf"):
                mime = "application/pdf"
            elif f.lower().endswith(".md"):
                mime = "text/markdown"
            else:
                mime = "application/octet-stream"

            with open(full_path, "rb") as fh:
                data = fh.read()

            st.download_button(
                label=f"Download {f}",
                data=data,
                file_name=f,
                mime=mime,
                key=f"download_report_{f}",
            )

    # 2) Holdings + CAPM summary
    holdings_path = os.path.join(OUTPUT_DIR, "holdings_table.csv")
    capm_path = os.path.join(OUTPUT_DIR, "capm_summary.csv")

    if os.path.exists(holdings_path):
        st.write("### Holdings Table")
        df_hold = pd.read_csv(holdings_path, index_col=0)
        st.dataframe(df_hold)

    if os.path.exists(capm_path):
        st.write("### CAPM Regression Results")
        df_capm = pd.read_csv(capm_path, index_col=0)
        st.dataframe(df_capm)

    # ORP weights + BL weights from summary.json
    orp_df = None
    bl_df = None
    summary_path = os.path.join(OUTPUT_DIR, "summary.json")
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r") as f:
                summary = json.load(f)
            w_dict = summary.get("max_sharpe_weights", {})
            if isinstance(w_dict, dict) and w_dict:
                orp_df = (
                    pd.Series(w_dict, name="weight")
                    .to_frame()
                    .assign(weight_pct=lambda df: df["weight"] * 100)
                    .sort_values("weight", ascending=False)
                )
                orp_df["weight"] = orp_df["weight"].round(4)
                orp_df["weight_pct"] = orp_df["weight_pct"].round(2)

            bl_info = summary.get("black_litterman")
            if isinstance(bl_info, dict) and bl_info.get("enabled"):
                w_bl = bl_info.get("weights", {})
                if isinstance(w_bl, dict) and w_bl:
                    bl_df = (
                        pd.Series(w_bl, name="weight")
                        .to_frame()
                        .assign(weight_pct=lambda df: df["weight"] * 100)
                        .sort_values("weight", ascending=False)
                    )
                    bl_df["weight"] = bl_df["weight"].round(4)
                    bl_df["weight_pct"] = bl_df["weight_pct"].round(2)

        except Exception:
            orp_df = None
            bl_df = None

    # 3) Key Charts
    st.write("### Key Charts")
    for name in priority_order:
        if name in priority_pngs:
            # nicer captions but filenames still drive PNG titles via plotting.py
            if name == "efficient_frontier.png":
                caption = "Efficient Frontier (ORP universe)"
            elif name == "black_litterman_efficient_frontier.png":
                caption = "Black-Litterman Efficient Frontier"
            elif name == "cal.png":
                caption = "Capital Allocation Line (CAL)"
            else:
                caption = name

            st.image(priority_pngs[name], caption=caption)

            # ORP weights under ORP frontier
            if name == "efficient_frontier.png" and orp_df is not None:
                st.write("### ORP (Optimal Risky Portfolio) Weights")
                st.dataframe(orp_df)

            # BL frontier + BL weights right underneath
            if name == "black_litterman_efficient_frontier.png" and bl_df is not None:
                st.write("### Black-Litterman Portfolio Weights")
                st.dataframe(bl_df)

    # 4) Drawdown & Tail Risk
    if drawdown_curves_path is not None or loss_histogram_path is not None:
        st.write("### Drawdown & Tail Risk")
        if drawdown_curves_path is not None:
            st.image(
                drawdown_curves_path,
                caption="Portfolio drawdown curves",
            )
        if loss_histogram_path is not None:
            st.image(
                loss_histogram_path,
                caption="Daily return distribution with VaR/CVaR",
            )
        metrics_path = os.path.join(OUTPUT_DIR, "drawdown_tail_metrics.csv")
        if os.path.exists(metrics_path):
            try:
                dd_metrics_df = pd.read_csv(metrics_path)
                st.dataframe(dd_metrics_df)
            except Exception as e:
                st.caption(f"Could not load drawdown_tail_metrics.csv: {e}")

    # 5) Rolling Risk Analytics
    if rolling_corr_path is not None or rolling_metrics_path is not None:
        st.write("### Rolling Risk Analytics")
        if rolling_corr_path is not None:
            st.image(
                rolling_corr_path,
                caption=rolling_corr_name or "rolling_corr_heatmap.gif",
            )
        if rolling_metrics_path is not None:
            st.image(rolling_metrics_path, caption="rolling_metrics.png")

    # 6) Performance Attribution (Brinson–Fachler)
    if (
        attribution_png_path is not None
        or attribution_csv_path is not None
        or sector_attribution_png_path is not None
        or sector_attribution_csv_path is not None
    ):
        st.write("### Performance Attribution (Brinson–Fachler)")

        # Asset-level attribution
        if attribution_png_path is not None:
            st.image(
                attribution_png_path,
                caption="Allocation, Selection, and Interaction Effects (Assets)",
            )
        if attribution_csv_path is not None:
            try:
                attr_df = pd.read_csv(attribution_csv_path)
                st.dataframe(attr_df)
            except Exception as e:
                st.caption(f"Could not load performance_attribution.csv: {e}")

        # Sector-level attribution (optional)
        if sector_attribution_png_path is not None or sector_attribution_csv_path is not None:
            st.write("#### Sector-level Attribution")
            if sector_attribution_png_path is not None:
                st.image(
                    sector_attribution_png_path,
                    caption=(
                        "Allocation, Selection, and Interaction Effects "
                        "(Sectors)"
                    ),
                )
            if sector_attribution_csv_path is not None:
                try:
                    sec_attr_df = pd.read_csv(sector_attribution_csv_path)
                    st.dataframe(sec_attr_df)
                except Exception as e:
                    st.caption(
                        f"Could not load performance_attribution_sector.csv: {e}"
                    )

    # 7) Extra charts
    if other_pngs:
        st.write("### Additional Charts")
        for f, full_path in other_pngs:
            st.image(full_path, caption=f)

    if gif_files:
        st.write("### Animated Charts")
        for f, full_path in gif_files:
            st.image(full_path, caption=f)

    # 8) Multi-factor regression tables
    mf_files = {
        "Fama-French 3-Factor (FF3)": "factor_regression_ff3.csv",
        "Carhart 4-Factor (Carhart4)": "factor_regression_carhart4.csv",
        "Fama-French 5-Factor (FF5)": "factor_regression_ff5.csv",
        "Quality & Low-Volatility": "factor_regression_quality_lowvol.csv",
    }

    first_mf = True
    for title, fname in mf_files.items():
        fpath = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(fpath):
            if first_mf:
                st.write("### Multi-Factor Regression Tables")
                first_mf = False
            st.write(f"**{title}**")
            try:
                df_mf = pd.read_csv(fpath)
                st.dataframe(df_mf)
            except Exception as e:
                st.write(f"*(Could not load {fname}: {e})*")

    # 9) CAPM PNGs
    if capm_pngs:
        st.write("### CAPM Scatter Plots")
        for f, full_path in capm_pngs:
            st.image(full_path, caption=f)

    # 10) CSV downloads
    if csv_files:
        st.write("### CSV Data Files")
        for f, full_path in csv_files:
            with open(full_path, "rb") as fh:
                data = fh.read()

            st.download_button(
                label=f"Download {f}",
                data=data,
                file_name=f,
                mime="text/csv",
                key=f"download_csv_{f}",
            )
