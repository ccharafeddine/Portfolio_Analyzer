"""
generate_report.py

Builds a markdown and PDF report summarizing the portfolio analysis
results saved in the outputs/ folder.

This version is simplified and robust:
- Uses safe DatetimeIndex intersections instead of "&"
- Uses "ME" (month end) for resampling
- Cleans text before sending to FPDF so no unsupported Unicode crashes
- Always uses explicit positive widths in FPDF cells/multi_cells
"""

from __future__ import annotations

import json
import os
from math import sqrt
from typing import Dict, Optional

import numpy as np
import pandas as pd
from fpdf import FPDF


# ------------------ Helpers to load data ------------------ #


def _read_value_series(csv_path: str, preferred_cols: list[str]) -> pd.Series:
    """
    Read a value series from a CSV.

    The CSV is expected to have a Date column and one or more numeric columns.
    We try a list of preferred column names first; if none are present, we
    fall back to:
      * the only numeric column, if there is exactly one
      * otherwise the first column.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date")

    # Try preferred columns in order
    for col in preferred_cols:
        if col in df.columns:
            s = df[col].astype(float)
            return s.sort_index()

    # Fall back to a single numeric column, if there is one
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 1:
        s = df[numeric_cols[0]].astype(float)
        return s.sort_index()

    # Otherwise just use the first column
    if len(df.columns) == 0:
        raise ValueError(f"No columns found in {csv_path}")
    s = df[df.columns[0]].astype(float)
    return s.sort_index()


def _load_series(outdir: str) -> Dict[str, pd.Series]:
    """
    Load realized value series for Active, Passive, ORP, and Complete.

    Any series whose CSV is missing will simply be skipped; at least one of
    Active/Passive should exist for the report to be useful.
    """
    mapping = {
        "Active": ("active_portfolio_value.csv", ["Value", "ActiveValue"]),
        "Passive": ("passive_portfolio_value.csv", ["Value", "PassiveValue"]),
        "ORP": ("orp_value_realized.csv", ["ORP_Value", "Value"]),
        "Complete": ("complete_portfolio_value.csv", ["Complete_Value", "Value"]),
    }

    series: Dict[str, pd.Series] = {}

    for name, (fname, preferred) in mapping.items():
        path = os.path.join(outdir, fname)
        if not os.path.exists(path):
            continue
        try:
            s = _read_value_series(path, preferred)
            s = s.dropna()
            if not s.empty:
                series[name] = s
        except Exception as e:
            print(f"Warning: failed to load {name} series from {path}: {e}")

    if not series:
        raise RuntimeError(
            "No value series found in outputs/. Expected files like "
            "'active_portfolio_value.csv' and 'passive_portfolio_value.csv'."
        )

    # Align all series to the common date intersection
    common_index: Optional[pd.DatetimeIndex] = None
    for s in series.values():
        if common_index is None:
            common_index = s.index
        else:
            common_index = common_index.intersection(s.index)

    if common_index is not None and len(common_index) > 0:
        for k in list(series.keys()):
            series[k] = series[k].loc[common_index]

    return series


# ------------------ Stats helpers ------------------ #


def _annualized_stats(series: pd.Series, rf_annual: float) -> Dict[str, float]:
    """Compute annualized return, volatility, and Sharpe for a value series."""
    # Use month-end ("ME") to avoid pandas "M" deprecation warning
    monthly = series.resample("ME").last().pct_change().dropna()
    if monthly.empty:
        return {"return": np.nan, "vol": np.nan, "sharpe": np.nan}

    mean_m = monthly.mean()
    vol_m = monthly.std()
    ret_ann = (1 + mean_m) ** 12 - 1
    vol_ann = vol_m * sqrt(12)
    sharpe = (ret_ann - rf_annual) / vol_ann if vol_ann > 0 else np.nan
    return {"return": ret_ann, "vol": vol_ann, "sharpe": sharpe}


def _maybe_img(outdir: str, filename: str) -> Optional[str]:
    """Return full path to an image in outdir if it exists, else None."""
    path = os.path.join(outdir, filename)
    return path if os.path.exists(path) else None


# ------------------ Markdown report ------------------ #


def _build_markdown_report(
    outdir: str,
    series: Dict[str, pd.Series],
    stats: Dict[str, Dict[str, float]],
    rf_annual: float,
    config: dict,
    holdings_table_path: Optional[str],
) -> str:
    """Construct the markdown report text and write it to outputs/report.md."""
    lines: list[str] = []

    # Basic date range
    any_series = next(iter(series.values()))
    start_date = any_series.index.min().strftime("%Y-%m-%d")
    end_date = any_series.index.max().strftime("%Y-%m-%d")

    benchmark = config.get("passive", {}).get("benchmark", "^GSPC")
    title = config.get("title", "Portfolio Analysis Report")

    lines.append(f"# {title}\n")
    lines.append("## 1. Overview\n")
    lines.append(
        "This report summarizes the historical performance of the active "
        "portfolio you constructed, a passive benchmark, the optimal risky "
        "portfolio (ORP), and the complete portfolio (combining the ORP with "
        "a risk-free asset) over the sample period.\n"
    )
    lines.append(f"- **Start date:** {start_date}\n")
    lines.append(f"- **End date:**   {end_date}\n")
    lines.append(f"- **Benchmark:**  {benchmark}\n")
    lines.append(f"- **Assumed annual risk-free rate:** {rf_annual:.2%}\n")

    # 2. Initial allocation
    lines.append("\n## 2. Active Portfolio Holdings (Initial Allocation)\n")
    if holdings_table_path and os.path.exists(holdings_table_path):
        try:
            ht = pd.read_csv(holdings_table_path)
            # Make a compact markdown table
            display_cols = [
                c
                for c in [
                    "Ticker",
                    "TargetWeight",
                    "PurchasePrice",
                    "Shares",
                    "Invested",
                    "RealizedWeight",
                ]
                if c in ht.columns
            ]
            if display_cols:
                lines.append(ht[display_cols].to_markdown(index=False))
                lines.append("\n")
            else:
                lines.append(
                    "Holdings table CSV found, but columns did not match the "
                    "expected schema.\n"
                )
        except Exception as e:
            lines.append(f"Could not load holdings_table.csv: {e}\n")
    else:
        lines.append(
            "The detailed holdings table was not found; please check that "
            "`outputs/holdings_table.csv` was generated.\n"
        )

    # 3. Risk/Return summary
    lines.append("\n## 3. Risk-Return Summary (Annualized)\n")
    lines.append(
        "| Portfolio | Annual Return | Annual Volatility | Sharpe (vs rf) |\n"
        "|-----------|--------------:|------------------:|---------------:|\n"
    )

    for name in ["Active", "Passive", "ORP", "Complete"]:
        st = stats.get(name)
        if st is None:
            continue
        r = st["return"]
        v = st["vol"]
        sh = st["sharpe"]
        lines.append(
            f"| {name} | {r: .2%} | {v: .2%} | {sh: .2f} |\n"
        )

    lines.append(
        "\nHigher annual return and a larger Sharpe ratio indicate better "
        "risk-adjusted performance. Comparing the active and passive "
        "portfolios shows whether your security selection and timing added "
        "value relative to simply holding the benchmark.\n"
    )

    # 4. Growth charts and additional visuals
    lines.append("\n## 4. Growth of $1,000,000\n")
    img = _maybe_img(outdir, "active_vs_passive_growth.png")
    if img:
        lines.append(
            "![Growth: Active vs Passive](active_vs_passive_growth.png)\n"
        )
        lines.append(
            "The figure above shows how $1,000,000 would have grown over time "
            "in the active portfolio versus the passive benchmark.\n"
        )
    else:
        lines.append(
            "Growth plot for Active vs Passive was not found "
            "(expected `active_vs_passive_growth.png`).\n"
        )

    img_all = _maybe_img(outdir, "all_portfolios_growth.png")
    if img_all:
        lines.append("\n### 4.1 All Portfolios\n")
        lines.append(
            "![Growth: All Portfolios](all_portfolios_growth.png)\n"
        )
        lines.append(
            "This chart compares the dollar growth of the Active, Passive, "
            "ORP, and Complete portfolios on a common scale.\n"
        )

    # 5. Risk-return scatter
    lines.append("\n## 5. Risk-Return Plot\n")
    rr_img = _maybe_img(outdir, "active_vs_passive_risk_return.png")
    if rr_img:
        lines.append(
            "![Risk-Return: Active vs Passive]"
            "(active_vs_passive_risk_return.png)\n"
        )
        lines.append(
            "The scatter plot places each portfolio in annualized risk-return "
            "space, highlighting the trade-off between volatility and return.\n"
        )
    else:
        lines.append(
            "Risk-return scatter plot (`active_vs_passive_risk_return.png`) "
            "was not found.\n"
        )

    # 6. Correlations and diversification
    lines.append("\n## 6. Diversification and Correlations\n")
    corr_img = _maybe_img(outdir, "correlation_matrix.png")
    if corr_img:
        lines.append("![Correlation Matrix](correlation_matrix.png)\n")
        lines.append(
            "The correlation heatmap helps illustrate how strongly each pair "
            "of assets moves together. Lower correlations generally improve "
            "diversification benefits.\n"
        )
    else:
        lines.append(
            "Correlation matrix plot (`correlation_matrix.png`) was not found.\n"
        )

    # 7. Complete portfolio composition
    lines.append("\n## 7. Complete Portfolio Allocation\n")
    cp_img = _maybe_img(outdir, "complete_portfolio_pie.png")
    if cp_img:
        lines.append("![Complete Portfolio Allocation](complete_portfolio_pie.png)\n")
        lines.append(
            "This pie chart shows the weights of each component in the "
            "complete portfolio that combines the optimal risky portfolio "
            "with the risk-free asset.\n"
        )
    else:
        lines.append(
            "Complete portfolio pie chart (`complete_portfolio_pie.png`) "
            "was not found.\n"
        )

    # 8. Forward-looking scenario analysis
    lines.append("\n## 8. Forward-Looking Scenario Analysis\n")
    fwd_img = _maybe_img(outdir, "forward_scenarios.png")
    if fwd_img:
        lines.append("![Forward Scenarios](forward_scenarios.png)\n")
        lines.append(
            "The scenario analysis illustrates how the active portfolio could "
            "evolve under optimistic, base-case, and pessimistic return "
            "assumptions going forward.\n"
        )
    else:
        lines.append(
            "Forward scenario plot (`forward_scenarios.png`) was not found.\n"
        )

    # 9. Concluding remarks
    lines.append("\n## 9. Interpretation and Takeaways\n")
    lines.append(
        "- Compare the active portfolio's Sharpe ratio and growth path to the "
        "passive benchmark to assess whether active management added value.\n"
        "- Use the ORP and complete portfolio results to understand how much "
        "additional return is available by taking on more risk, and how "
        "allocating between the risky portfolio and the risk-free asset "
        "shifts the overall profile.\n"
        "- The correlation structure across holdings indicates where "
        "diversification could potentially be improved by adjusting weights "
        "or adding new assets.\n"
    )

    md_path = os.path.join(outdir, "report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Saved report to {md_path}")
    return md_path


# ------------------ PDF report ------------------ #


class _PDF(FPDF):
    """Small helper subclass in case we want to tweak defaults later."""
    pass


def _clean(text: str) -> str:
    """
    Clean text so it only uses characters supported by core FPDF fonts.

    Replaces common Unicode punctuation with ASCII equivalents.
    """
    if not isinstance(text, str):
        text = str(text)

    replacements = {
        "–": "-",   # en dash
        "—": "-",   # em dash
        "−": "-",   # minus sign
        "’": "'",   # curly apostrophe
        "‘": "'",   # left single quote
        "“": '"',   # left double quote
        "”": '"',   # right double quote
        "•": "-",   # bullet
        "…": "...", # ellipsis
        "\u00a0": " ",  # non-breaking space
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return text


def _build_pdf_report(
    outdir: str,
    series: Dict[str, pd.Series],
    stats: Dict[str, Dict[str, float]],
    rf_annual: float,
    orp_weights: Optional[pd.DataFrame],
    allow_short: bool,
) -> None:
    """
    Build a simple PDF report from the same information as the markdown.

    Layout is intentionally conservative so that fpdf2 does not run into
    horizontal-space issues when wrapping text.
    """
    pdf = _PDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Effective page width (total width minus margins)
    epw = pdf.w - pdf.l_margin - pdf.r_margin

    pdf.set_title(_clean("Portfolio Analysis Report"))
    pdf.set_author(_clean("Portfolio Analyzer"))

    # Title
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_x(pdf.l_margin)
    pdf.cell(epw, 10, _clean("Portfolio Analysis Report"), ln=1, align="C")
    pdf.ln(4)

    # Date range
    any_series = next(iter(series.values()))
    start_date = any_series.index.min().strftime("%Y-%m-%d")
    end_date = any_series.index.max().strftime("%Y-%m-%d")

    pdf.set_font("Helvetica", "", 11)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(
        epw,
        6,
        _clean(
            f"This report summarizes the performance of the active portfolio, "
            f"a passive benchmark, the optimal risky portfolio (ORP), and the "
            f"complete portfolio over the period {start_date} to {end_date}. "
            f"We assume an annual risk-free rate of {rf_annual:.2%}."
        ),
    )
    pdf.ln(4)

    # Risk/Return table (text only)
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_x(pdf.l_margin)
    pdf.cell(epw, 8, _clean("Annualized Risk-Return Summary"), ln=1)
    pdf.set_font("Helvetica", "", 11)

    for name in ["Active", "Passive", "ORP", "Complete"]:
        st = stats.get(name)
        if st is None:
            continue
        r = st["return"]
        v = st["vol"]
        sh = st["sharpe"]
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(
            epw,
            6,
            _clean(
                f"- {name}: Return {r:.2%}, Volatility {v:.2%}, Sharpe {sh:.2f}"
            ),
        )
    pdf.ln(4)

    # If ORP weights are available, summarise them briefly
    if orp_weights is not None and not orp_weights.empty:
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_x(pdf.l_margin)
        pdf.cell(epw, 8, _clean("ORP Weights"), ln=1)
        pdf.set_font("Helvetica", "", 10)

        try:
            w = orp_weights.copy()
            if "weight" in w.columns:
                w = w.sort_values("weight", ascending=False)
            # Cap at top 10
            w = w.head(10)
            for idx, row in w.iterrows():
                w_val = row["weight"] if "weight" in row else row.iloc[0]
                pdf.set_x(pdf.l_margin)
                pdf.multi_cell(epw, 5, _clean(f"- {idx}: {w_val:.2%}"))
        except Exception:
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(
                epw,
                5,
                _clean(
                    "ORP weights file found, but could not be parsed cleanly."
                ),
            )
        pdf.ln(4)

    # Helper to add image sections
    def add_image_section(title: str, filename: str, caption: str):
        img_path = os.path.join(outdir, filename)
        if not os.path.exists(img_path):
            return

        pdf.add_page()
        local_epw = pdf.w - pdf.l_margin - pdf.r_margin

        pdf.set_font("Helvetica", "B", 13)
        pdf.set_x(pdf.l_margin)
        pdf.cell(local_epw, 8, _clean(title), ln=1)
        pdf.ln(2)

        # Place image with a safe width (fits within margins)
        max_width = min(180, local_epw)
        pdf.set_x(pdf.l_margin)
        pdf.image(img_path, w=max_width)
        pdf.ln(4)

        pdf.set_font("Helvetica", "", 10)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(local_epw, 5, _clean(caption))

    add_image_section(
        "Growth of $1,000,000: Active vs Passive",
        "active_vs_passive_growth.png",
        "Growth of an initial $1,000,000 investment in the active portfolio "
        "versus the passive benchmark over the sample period.",
    )

    add_image_section(
        "Risk-Return Comparison: Active vs Passive",
        "active_vs_passive_risk_return.png",
        "Annualized risk-return trade-off for the active and passive "
        "portfolios.",
    )

    add_image_section(
        "Correlation Matrix",
        "correlation_matrix.png",
        "Correlation structure across portfolio holdings. Lower correlations "
        "indicate better diversification potential.",
    )

    add_image_section(
        "Complete Portfolio Allocation",
        "complete_portfolio_pie.png",
        "Weights of each component in the complete portfolio, combining the "
        "optimal risky portfolio with the risk-free asset.",
    )

    add_image_section(
        "Forward-Looking Scenarios",
        "forward_scenarios.png",
        "Projected paths of the active portfolio under pessimistic, base-case, "
        "and optimistic assumptions.",
    )

    pdf_path = os.path.join(outdir, "report.pdf")
    pdf.output(pdf_path)
    print(f"Saved PDF report to {pdf_path}")


# ------------------ Public entry point ------------------ #


def generate_report(outdir: str, config_path: str = "config.json") -> None:
    """Main entry point: build markdown + PDF reports."""
    os.makedirs(outdir, exist_ok=True)

    series = _load_series(outdir)

    # Load config if present
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            cfg = json.load(f)
    else:
        cfg = {}

    rf = cfg.get("risk_free_rate", 0.02)

    stats = {name: _annualized_stats(s, rf) for name, s in series.items()}

    holdings_table_path = os.path.join(outdir, "holdings_table.csv")
    if not os.path.exists(holdings_table_path):
        holdings_table_path = None

    # Build markdown
    _build_markdown_report(outdir, series, stats, rf, cfg, holdings_table_path)

    # ORP weights (if available)
    orp_weights_path = os.path.join(outdir, "orp_weights.csv")
    orp_weights: Optional[pd.DataFrame]
    if os.path.exists(orp_weights_path):
        try:
            orp_weights = pd.read_csv(orp_weights_path, index_col=0)
        except Exception:
            orp_weights = None
    else:
        orp_weights = None

    allow_short = cfg.get("constraints", {}).get("short_sales", False)

    try:
        _build_pdf_report(outdir, series, stats, rf, orp_weights, allow_short)
    except Exception as e:
        print(f"Failed to generate PDF: {e}")
