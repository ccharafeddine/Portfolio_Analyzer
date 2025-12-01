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
    value=cfg.get("benchmark") or cfg.get("passive_benchmark", "^GSPC"),
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
    "Complete Portfolio: % allocated to ORP",
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
        st.error("❌ Weights must sum to 1.0 unless equal-weights is selected.")
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
cfg["max_allocation_bounds"] = cfg.get("max_allocation_bounds", [0.0, 1.0])
cfg["short_sales"] = bool(allow_shorts)
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

if st.button("💾 Save Config"):
    save_config(cfg)
    st.success("Config saved!")

# =========================
# Run Analysis
# =========================
st.subheader("🚀 Step 4 – Run Analysis")

run_clicked = st.button("Run Portfolio Analysis")

if run_clicked:
    st.session_state["run_status"] = "running"
    st.session_state["run_error"] = None

    python_exec = sys.executable

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
# Outputs & Stats
# =========================
if os.path.exists(OUTPUT_DIR):
    files = sorted(os.listdir(OUTPUT_DIR))
    st.write("### Output Files")

    # One-click ZIP download of everything in outputs/
    if files:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in files:
                full_path = os.path.join(OUTPUT_DIR, f)
                if os.path.isfile(full_path):
                    zf.write(full_path, arcname=f)
        zip_buffer.seek(0)

        st.download_button(
            label="⬇️ Download all outputs as ZIP",
            data=zip_buffer,
            file_name="outputs.zip",
            mime="application/zip",
            key="download_all_zip",
        )

    # Individual files: show PNGs inline, others as downloads
    for f in files:
        full_path = os.path.join(OUTPUT_DIR, f)

        if f.lower().endswith(".png"):
            st.image(full_path, caption=f)
            continue

        if f.lower().endswith(".pdf"):
            mime = "application/pdf"
        elif f.lower().endswith(".md"):
            mime = "text/markdown"
        elif f.lower().endswith(".csv"):
            mime = "text/csv"
        else:
            mime = "application/octet-stream"

        with open(full_path, "rb") as fh:
            data = fh.read()

        st.download_button(
            label=f"Download {f}",
            data=data,
            file_name=f,
            mime=mime,
            key=f"download_{f}",
        )

    # Optional stats tables (once we save them from main.py)
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
