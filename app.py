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
st.title("📊 Portfolio Analyzer App")

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
    or "2024-01-01"
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

# =========================
# Active Portfolio
# =========================
st.subheader("📈 Active Portfolio")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### Tickers (one per line)")
    tickers_text = st.text_area(
        "Tickers (one per line)",
        value="\n".join(ap.get("tickers", ["AAPL", "MSFT", "GLD"])),
        height=200,
        label_visibility="collapsed",
    )

tickers = [t.strip().upper() for t in tickers_text.split("\n") if t.strip()]

with col2:
    # spacer so checkboxes visually align with text box
    st.markdown("<br><br>", unsafe_allow_html=True)

    equal_weights = st.checkbox(
        "Use Equal Weights",
        value=False,
    )
    use_dividends = st.checkbox(
        "Use Dividends",
        value=ap.get("use_dividends", False),
    )
    allow_shorts = st.checkbox(
        "Allow Shorting",
        value=ap.get("allow_shorts", False),
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

    stored_weights = ap.get("weights")
    if isinstance(stored_weights, dict):
        default_weights = [stored_weights.get(t, 0.0) for t in tickers]
    elif isinstance(stored_weights, list):
        default_weights = stored_weights
    else:
        default_weights = [1.0 / len(tickers)] * len(tickers) if tickers else []

    for idx, t in enumerate(tickers):
        col = weight_cols[idx % 3]

        if idx < len(default_weights):
            default_val = float(default_weights[idx])
        else:
            default_val = 1.0 / len(tickers) if tickers else 0.0

        w = col.number_input(
            f"{t} weight",
            min_value=-1.0,
            max_value=2.0,
            value=default_val,
            step=0.01,
            format="%.4f",
        )
        weights_list.append(float(w))

    if tickers and abs(sum(weights_list) - 1.0) > 1e-6:
        st.error("❌ Weights must sum to 1.0 unless equal-weights is selected. ❌")
        st.stop()

weights_dict = {t: w for t, w in zip(tickers, weights_list)}

# =========================
# Build config.json
# =========================
cfg["tickers"] = tickers
cfg["benchmark"] = benchmark

cfg["start"] = str(start_date)
cfg["end"] = str(end_date)

cfg["risk_free_rate"] = float(cfg.get("risk_free_rate", 0.04))

# Shorting + bounds logic
cfg["short_sales"] = bool(allow_shorts)
if allow_shorts:
    # Allow up to 150% long or short per asset (net exposure still 100% via sum(w)=1)
    cfg["max_allocation_bounds"] = [-1.5, 1.5]
else:
    # Long-only case; per-asset cap at 150% but sum(w)=1 keeps net 100% long
    cfg["max_allocation_bounds"] = [0.0, 1.5]

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

cfg["passive_benchmark"] = benchmark
cfg["passive"] = {"benchmark": benchmark}
cfg["initial_capital"] = float(initial_capital)
cfg["start_date"] = str(start_date)
cfg["end_date"] = str(end_date)
cfg["y_cp"] = float(y_cp)

if st.button("💾 Save Config 💾"):
    save_config(cfg)
    st.success("Config saved!")

# =========================
# Run Analysis
# =========================
st.subheader("🚀 Run Analysis")

run_clicked = st.button("🏃‍♀️ Run Portfolio Analysis 🏃‍♀️")

if run_clicked:
    st.session_state["run_status"] = "running"
    st.session_state["run_error"] = None

    # Clear outputs so each run is clean (no leftover CAPM plots, etc.)
    clear_output_dir()

    python_exec = sys.executable

    # 🔄 Show spinner while analysis runs
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
if os.path.exists(OUTPUT_DIR):
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
    reports = []         # report.pdf, report.md, etc.
    priority_pngs = {}   # specific PNGs, by name
    capm_pngs = []       # capm_*.png
    other_pngs = []      # any other pngs
    csv_files = []       # all csvs
    other_files = []     # anything else (non-report pdf/md/etc.)
    gif_files = []       # all gifs

    # Special rolling-risk outputs
    rolling_metrics_path = None
    rolling_corr_name = None
    rolling_corr_path = None

    priority_order = [
        "correlation_matrix.png",
        "efficient_frontier.png",
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
                # reserve for Rolling Risk Analytics section
                rolling_metrics_path = full_path
            elif lower.startswith("capm_"):
                capm_pngs.append((f, full_path))
            else:
                other_pngs.append((f, full_path))
        elif lower.endswith(".gif"):
            gif_files.append((f, full_path))
        elif lower.endswith(".csv"):
            csv_files.append((f, full_path))
        elif lower.endswith(".pdf") or lower.endswith(".md"):
            if "report" in lower:
                reports.append((f, full_path))
            else:
                other_files.append((f, full_path))
        else:
            other_files.append((f, full_path))

    # Helper to pull & remove a specific GIF by name
    def pop_gif_by_name(target_name: str):
        target_name = target_name.lower()
        for idx, (gf, gpath) in enumerate(gif_files):
            if gf.lower() == target_name:
                gif_files.pop(idx)
                return gf, gpath
        return None, None

    # Reserve rolling correlation GIF for Rolling Risk section
    for candidate in ["rolling_corr_heatmap.gif", "rolling_cor_heatmap.gif"]:
        gf, gpath = pop_gif_by_name(candidate)
        if gpath:
            rolling_corr_name, rolling_corr_path = gf, gpath
            break

    # ---------- 1) Reports right after ZIP ----------
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

    # ---------- 2) Holdings + CAPM tables ----------
    holdings_path = os.path.join(OUTPUT_DIR, "holdings_table.csv")
    capm_path = os.path.join(OUTPUT_DIR, "capm_results.csv")

    if os.path.exists(holdings_path):
        st.write("### Holdings Table")
        df_hold = pd.read_csv(holdings_path, index_col=0)
        st.dataframe(df_hold)

    if os.path.exists(capm_path):
        st.write("### CAPM Regression Results")
        df_capm = pd.read_csv(capm_path, index_col=0)
        st.dataframe(df_capm)

    # ---------- ORP weights from summary.json ----------
    orp_df = None
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
        except Exception:
            orp_df = None

    # ---------- 3) Key Charts in desired order ----------
    st.write("### Key Charts")
    for name in priority_order:
        if name in priority_pngs:
            st.image(priority_pngs[name], caption=name)
            # Insert ORP table right after efficient_frontier.png
            if name == "efficient_frontier.png" and orp_df is not None:
                st.write("### ORP (Optimal Risky Portfolio) Weights")
                st.dataframe(orp_df)

    # ---------- 4) Rolling Risk Analytics (right after Key Charts) ----------
    if rolling_corr_path is not None or rolling_metrics_path is not None:
        st.write("### Rolling Risk Analytics")
        if rolling_corr_path is not None:
            st.image(
                rolling_corr_path,
                caption=rolling_corr_name or "rolling_corr_heatmap.gif",
            )
        if rolling_metrics_path is not None:
            st.image(rolling_metrics_path, caption="rolling_metrics.png")

    # ---------- 5) Any other non-CAPM PNGs ----------
    if other_pngs:
        st.write("### Additional Charts")
        for f, full_path in other_pngs:
            st.image(full_path, caption=f)

    # (Optional) any remaining GIFs (if you ever add more animations)
    if gif_files:
        st.write("### Animated Charts")
        for f, full_path in gif_files:
            st.image(full_path, caption=f)

    # ---------- 6) CAPM PNGs (now before CSVs) ----------
    if capm_pngs:
        st.write("### CAPM Scatter Plots")
        for f, full_path in capm_pngs:
            st.image(full_path, caption=f)

    # ---------- 7) CSV downloads (last) ----------
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