# ------------------------------------------------------------
# generate_report.py
# Full module with:
# - Smart file loading
# - Embedded charts
# - Summary table
# - Risk bucket classification
# - Interpretations
# - Holdings & benchmark in summary
# - Backtest start/end dates
# - Ordered narrative matching Streamlit outputs
# - Universal support for any portfolio
# - Optional Black-Litterman section (when outputs exist)
# ------------------------------------------------------------

import os
import json
from datetime import datetime
import math

import pandas as pd

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    Table,
    TableStyle,
)
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch


OUTPUT_DIR = "outputs"

# ============================================================
# Helper Smart Loaders
# ============================================================


def _find_matching_file(outdir, prefix, exts=(".csv", ".json", "")):
    """Find a file in outdir whose name starts with prefix.
    Supports exact matches, prefix matches, and extensionless files.
    Returns full path or None.
    """
    # First: try exact names with preferred extensions
    for ext in exts:
        candidate = os.path.join(outdir, prefix + ext)
        if os.path.exists(candidate):
            return candidate

    # Second: fuzzy match: any file that starts with prefix
    for fname in os.listdir(outdir):
        if fname.startswith(prefix):
            return os.path.join(outdir, fname)

    return None


def load_json_smart(outdir, prefix):
    """Load JSON even if extension is missing or different."""
    path = _find_matching_file(outdir, prefix, exts=(".json", ""))
    if path is None:
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def load_csv_smart(outdir, prefix):
    """Load CSV even if extension is missing or has variations."""
    path = _find_matching_file(outdir, prefix, exts=(".csv", ""))
    if path is None:
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


# ============================================================
# Markdown helper
# ============================================================


def write_md_section(md_lines, title, body):
    md_lines.append(f"## {title}\n\n")
    md_lines.append(body.strip() + "\n\n")


# ============================================================
# Formatting helpers
# ============================================================


def fmt_pct_or_num(x, force_pct=False):
    if x is None:
        return "N/A"
    try:
        val = float(x)
    except Exception:
        return str(x)

    if not math.isfinite(val):
        return "N/A"

    if force_pct or abs(val) < 2.0:
        return f"{val * 100:.2f}%"
    return f"{val:.2f}"


# ------------------------------------------------------------
# Interpretation Engine + Risk Buckets + Conclusions
# ------------------------------------------------------------


def interpret_performance(summary_json):
    if not summary_json:
        return "Performance data was not available."

    rp = summary_json.get("portfolio_return")
    rb = summary_json.get("benchmark_return")

    try:
        rp = float(rp)
        rb = float(rb)
    except Exception:
        return "Return data was not available."

    if not (math.isfinite(rp) and math.isfinite(rb)):
        return "Return data was not available."

    diff = rp - rb

    if diff > 0:
        return (
            "The portfolio outperformed the benchmark over the analysis period. "
            "This means the chosen holdings and weightings delivered stronger results "
            "than a passive exposure to the benchmark. The excess performance reflects either "
            "effective stock selection, meaningful growth exposure, or advantageous positioning "
            "during favorable market trends."
        )
    elif diff < 0:
        return (
            "The portfolio underperformed relative to the benchmark. This suggests the holdings "
            "or weightings lagged the broader market. The results may reflect exposure to slower-moving "
            "sectors, underperformance among key positions, or defensive allocation choices."
        )
    else:
        return (
            "The portfolio delivered performance closely aligned with the benchmark—neither significantly "
            "outperforming nor lagging. This suggests a broadly market-consistent return pattern."
        )


def interpret_risk(summary_json):
    if not summary_json:
        return "Risk measures were not available."

    sharpe = summary_json.get("portfolio_sharpe")

    if sharpe is None:
        return "Risk statistics were not available."

    if sharpe > 1.0:
        return (
            "The portfolio achieved strong risk-adjusted returns. A Sharpe ratio above 1 indicates "
            "that the portfolio was well compensated for the volatility taken on."
        )
    elif sharpe > 0.5:
        return (
            "The portfolio delivered reasonable risk-adjusted results. The relationship between volatility "
            "and return was acceptable and broadly efficient for the strategy employed."
        )
    else:
        return (
            "Risk-adjusted returns were modest. The volatility taken on did not translate into proportionally "
            "strong returns. This may indicate concentration risk, elevated market volatility, or positions "
            "that did not behave as expected."
        )


def interpret_drawdown(dd_df):
    if dd_df is None or dd_df.empty:
        return "Drawdown statistics were not available."

    col = None
    for candidate in ["MaxDrawdown", "max_drawdown"]:
        if candidate in dd_df.columns:
            col = candidate
            break

    if col is None:
        return "Drawdown statistics were not available."

    try:
        max_dd = float(dd_df[col].iloc[0])
    except Exception:
        return "Drawdown statistics were not available."

    if not math.isfinite(max_dd):
        return "Drawdown statistics were not available."

    if max_dd < -0.25:
        return (
            "The portfolio experienced a deep peak-to-trough decline, typical of high-volatility or growth-tilted "
            "strategies. This level of drawdown often reflects periods of market stress or concentrated exposures."
        )
    elif max_dd < -0.15:
        return (
            "The portfolio experienced a moderate drawdown. This is normal for portfolios with a tilt toward "
            "growth assets or higher-return strategies and remained within a range most long-term investors could tolerate."
        )
    else:
        return (
            "Drawdowns remained relatively limited, suggesting that the portfolio maintained stability even during "
            "periods of broader market volatility."
        )


def interpret_attribution(df):
    if df is None or df.empty or "Total" not in df.columns:
        return "Attribution results were not available."

    totals = df["Total"]
    idx_top = totals.idxmax()
    idx_bottom = totals.idxmin()

    label_col = None
    for c in ["Bucket", "Asset", "Name"]:
        if c in df.columns:
            label_col = c
            break

    if label_col is not None:
        top_label = df.loc[idx_top, label_col]
        bottom_label = df.loc[idx_bottom, label_col]
    else:
        top_label = str(idx_top)
        bottom_label = str(idx_bottom)

    if totals.loc[idx_top] > 0:
        return (
            f"The portfolio’s relative performance was driven primarily by **{top_label}**, "
            "which provided the strongest positive contribution to active return. This reflects either "
            "successful overweighting or strong selection insight. "
            f"By contrast, **{bottom_label}** was the weakest contributor."
        )

    return (
        "Attribution results indicate that no single position provided a dominant positive contribution, "
        "and relative performance was spread broadly across holdings."
    )


def interpret_sector_attribution(df):
    if df is None or df.empty or "Total" not in df.columns:
        return "Sector-level attribution was not available."

    totals = df["Total"]
    idx_top = totals.idxmax()
    idx_bottom = totals.idxmin()

    sector_col = "Sector" if "Sector" in df.columns else None

    if sector_col is not None:
        top_label = df.loc[idx_top, sector_col]
        bottom_label = df.loc[idx_bottom, sector_col]
    else:
        top_label = str(idx_top)
        bottom_label = str(idx_bottom)

    explanation = (
        f"Sector attribution shows that **{top_label}** contributed most positively to active return. "
        "This typically reflects strong performance combined with meaningful exposure to that sector. "
    )

    if totals.loc[idx_bottom] < 0:
        explanation += (
            f"Meanwhile, **{bottom_label}** detracted from performance relative to the benchmark, "
            "likely due to weak returns or overweighting."
        )
    else:
        explanation += (
            "Other sectors exhibited neutral or modest contributions without materially detracting."
        )

    return explanation


def interpret_factors(df):
    """Explain multi-factor regression results in plain language."""
    if df is None or df.empty:
        return (
            "Multi-factor regressions were run to measure how the holdings load onto common style factors "
            "such as market, size, value, momentum, quality, and low-volatility. "
            "The regressions help distinguish broad style tilts from idiosyncratic (stock-specific) effects."
        )

    cols = df.columns
    beta_cols = [c for c in cols if c.endswith("_coef") and c.lower() != "const_coef"]

    lines = []
    for b in beta_cols:
        try:
            val = float(df[b].abs().mean())
        except Exception:
            continue

        factor = b.replace("_coef", "").upper()
        if val > 0.3:
            lines.append(
                f"- The portfolio shows a **meaningful tilt** toward the **{factor}** factor on average."
            )

    if not lines:
        return (
            "Factor analysis indicates that returns were driven primarily by stock-specific behavior "
            "rather than strong systematic factor tilts. The estimated factor loadings are generally modest."
        )

    intro = (
        "Multi-factor regressions help decompose returns into systematic style exposures. "
        "Based on the estimated factor loadings across holdings:"
    )
    return intro + "\n" + "\n".join(lines)


def interpret_black_litterman(bl_weights_df, bl_views_df, bl_tickers=None):
    """Explain the Black-Litterman results in the same narrative style.

    bl_tickers is an optional ordered list of tickers corresponding to the
    rows of bl_weights_df. When provided, it is used to label weights so the
    narrative and tables reference actual names instead of row indices.
    """
    if (
        (bl_weights_df is None or bl_weights_df.empty)
        and (bl_views_df is None or bl_views_df.empty)
    ):
        return (
            "The Black-Litterman model blends market-implied equilibrium returns "
            "with explicit investor views to produce an alternative, view-consistent "
            "set of expected returns and portfolio weights. In this run, no "
            "Black-Litterman outputs were detected in the outputs directory."
        )

    text = (
        "The Black-Litterman model combines market-cap–implied equilibrium returns "
        "with the investor views specified in the configuration. The result is a "
        "posterior set of expected returns that is consistent with both the market "
        "and the views, and a corresponding Black-Litterman portfolio optimized "
        "on that posterior return vector.\n\n"
    )

    if bl_weights_df is not None and not bl_weights_df.empty:
        try:
            df = bl_weights_df.copy()

            # Attach tickers if not already present
            ticker_col = None
            for c in df.columns:
                cl = c.lower()
                if cl in ("ticker", "asset", "name", "symbol"):
                    ticker_col = c
                    break
            if ticker_col is None and bl_tickers is not None:
                labels = [str(t) for t in bl_tickers]
                if len(labels) >= len(df):
                    df.insert(0, "Ticker", labels[: len(df)])
                    ticker_col = "Ticker"

            # Find weight column
            weight_col = None
            for c in df.columns:
                cl = c.lower()
                if cl in ("weight", "weights", "bl_weight", "bl_weights"):
                    weight_col = c
                    break
            if weight_col is None:
                num_cols = df.select_dtypes(include="number").columns
                if len(num_cols) == 1:
                    weight_col = num_cols[0]

            if weight_col is not None:
                s = df[weight_col].astype(float)

                # Label the series
                if ticker_col is not None:
                    s.index = df[ticker_col].astype(str)
                elif df.index.name is not None or df.index.dtype == "object":
                    s.index = df.index.astype(str)

                s = s[s.abs() > 1e-6].sort_values(ascending=False)
                top = s.head(5)
                if not top.empty:
                    bullets = ", ".join(f"{idx} ({val:.1%})" for idx, val in top.items())
                    text += (
                        "The resulting Black-Litterman portfolio tilts toward a "
                        "subset of names that express those views. Key weights include: "
                        f"{bullets}.\n\n"
                    )
        except Exception:
            # Fail silently; narrative above still stands
            pass

    if bl_views_df is not None and not bl_views_df.empty:
        text += (
            "The view table summarises the assumptions used in the model—such as "
            "absolute or relative return views and their associated confidence "
            "levels. Together with the benchmark covariance matrix, these define "
            "how far the posterior returns can deviate from equilibrium.\n"
        )
    else:
        text += (
            "The detailed view table was not available in the outputs directory; "
            "consult the configuration for the specific investor views used in "
            "this run.\n"
        )

    return text


# ============================================================
# Risk Bucket Classification
# ============================================================


def compute_risk_bucket(summary_json, dd_df):
    """Assign a risk bucket even if some metrics are missing."""
    vol = None
    if summary_json:
        vol = summary_json.get("portfolio_volatility")

    try:
        vol = float(vol)
    except Exception:
        vol = 0.0
    if not math.isfinite(vol):
        vol = 0.0

    max_dd = 0.0
    if dd_df is not None and not dd_df.empty:
        col = None
        for candidate in ["MaxDrawdown", "max_drawdown"]:
            if candidate in dd_df.columns:
                col = candidate
                break
        if col is not None:
            try:
                max_dd = float(dd_df[col].iloc[0])
            except Exception:
                max_dd = 0.0

    if vol < 0.10 and max_dd > -0.15:
        bucket = "Conservative"
        expl = (
            "The portfolio fits a **Conservative** risk bucket, prioritizing stability "
            "and limited drawdowns."
        )
    elif vol < 0.15 and max_dd > -0.25:
        bucket = "Moderate"
        expl = (
            "The portfolio fits a **Moderate** risk bucket, balancing growth potential "
            "with measured risk-taking."
        )
    elif vol < 0.25 and max_dd > -0.35:
        bucket = "Growth"
        expl = (
            "The portfolio aligns with a **Growth** profile, taking moderate volatility "
            "to target higher long-term returns."
        )
    elif vol < 0.35:
        bucket = "Aggressive"
        expl = (
            "The portfolio fits an **Aggressive** profile, accepting meaningful volatility "
            "in exchange for higher return potential."
        )
    else:
        bucket = "Very Aggressive"
        expl = (
            "The portfolio is **Very Aggressive**, with high sensitivity to market movement "
            "and deep drawdown potential."
        )

    return bucket, expl


# ============================================================
# Conclusions & Recommendations
# ============================================================


def interpret_conclusions(summary_json, dd_df, asset_attr, sector_attr, factor_df):
    performance_text = interpret_performance(summary_json)
    risk_text = interpret_risk(summary_json)
    bucket_label, _ = compute_risk_bucket(summary_json, dd_df)

    extras = []

    if summary_json:
        rp = summary_json.get("portfolio_return")
        rb = summary_json.get("benchmark_return")
        try:
            rp_f = float(rp)
            rb_f = float(rb)
        except Exception:
            rp_f = rb_f = None

        if rp_f is not None and rb_f is not None:
            if rp_f - rb_f > 0:
                extras.append(
                    "Recent outperformance suggests that the current structure is adding value, "
                    "though ongoing monitoring is important to avoid over-concentration."
                )
            elif rp_f - rb_f < 0:
                extras.append(
                    "Underperformance relative to the benchmark indicates it may be worthwhile "
                    "to reassess position sizing or sector exposures."
                )

    if asset_attr is not None and not asset_attr.empty and "Total" in asset_attr.columns:
        totals = asset_attr["Total"]
        idx_top = totals.idxmax()

        label = None
        for c in ["Bucket", "Asset", "Name"]:
            if c in asset_attr.columns:
                label = asset_attr.loc[idx_top, c]
                break
        name = label if label is not None else str(idx_top)

        extras.append(
            f"Attribution identifies **{name}** as a major driver of results. "
            "Its weighting and role within the portfolio should be reviewed periodically."
        )

    if factor_df is not None and not factor_df.empty:
        extras.append(
            "Factor analysis helps clarify whether returns are sourced from broad market style exposures "
            "or unique security-specific drivers."
        )

    extras_text = " ".join(extras) if extras else ""

    return (
        f"The portfolio currently aligns with a **{bucket_label}** risk profile. "
        f"{performance_text} {risk_text} {extras_text} "
        "Future allocation decisions should reflect the investor’s long-term objectives and risk tolerance."
    )


# ============================================================
# Summary Table Builder
# ============================================================


def build_summary_rows(
    summary,
    dd_df,
    asset_attr,
    sector_attr,
    risk_bucket_label,
    benchmark_ticker=None,
    start_date=None,
    end_date=None,
):
    rows = []

    if benchmark_ticker:
        rows.append(["Benchmark", str(benchmark_ticker)])

    if start_date:
        rows.append(["Backtest Start Date", str(start_date)])
    if end_date:
        rows.append(["Backtest End Date", str(end_date)])

    if summary:
        pr = summary.get("portfolio_return")
        br = summary.get("benchmark_return")
        alpha = summary.get("alpha")

        rows.append(["Portfolio return", fmt_pct_or_num(pr, True)])
        rows.append(["Benchmark return", fmt_pct_or_num(br, True)])

        if alpha is not None:
            rows.append(["Alpha (annualized)", fmt_pct_or_num(alpha, True)])

        vol = summary.get("portfolio_volatility")
        sharpe = summary.get("portfolio_sharpe")
        beta = summary.get("beta")
        bench_sharpe = summary.get("benchmark_sharpe")
        gain_to_pain = summary.get("portfolio_gain_to_pain")

        rows.append(["Volatility (annualized)", fmt_pct_or_num(vol, True)])
        if sharpe is not None:
            rows.append(["Portfolio Sharpe ratio", f"{sharpe:.2f}"])
        if bench_sharpe is not None:
            rows.append(["Benchmark Sharpe ratio", f"{float(bench_sharpe):.2f}"])
        if gain_to_pain is not None:
            rows.append(
                ["Gain-to-Pain ratio (portfolio)", f"{float(gain_to_pain):.2f}"]
            )
        if beta is not None:
            rows.append(["Beta", f"{beta:.2f}"])

    if dd_df is not None and not dd_df.empty:
        dd_col = None
        for candidate in ["MaxDrawdown", "max_drawdown"]:
            if candidate in dd_df.columns:
                dd_col = candidate
                break

        if dd_col is not None:
            rows.append(
                ["Max drawdown", fmt_pct_or_num(dd_df[dd_col].iloc[0], True)]
            )
        if "VaR_95" in dd_df.columns:
            rows.append(["95% VaR", fmt_pct_or_num(dd_df["VaR_95"].iloc[0], True)])
        if "CVaR_95" in dd_df.columns:
            rows.append(["95% CVaR", fmt_pct_or_num(dd_df["CVaR_95"].iloc[0], True)])

    if asset_attr is not None and not asset_attr.empty and "Total" in asset_attr.columns:
        totals = asset_attr["Total"]
        idx_top = totals.idxmax()

        label = None
        for c in ["Bucket", "Asset", "Name"]:
            if c in asset_attr.columns:
                label = asset_attr.loc[idx_top, c]
                break
        rows.append(
            [
                "Top contributing asset",
                label if label is not None else str(idx_top),
            ]
        )

    if sector_attr is not None and not sector_attr.empty and "Total" in sector_attr.columns:
        totals_s = sector_attr["Total"]
        idx_top_s = totals_s.idxmax()

        if "Sector" in sector_attr.columns:
            sec_label = sector_attr.loc[idx_top_s, "Sector"]
        else:
            sec_label = str(idx_top_s)
        rows.append(["Top contributing sector", sec_label])

    rows.append(["Risk bucket", risk_bucket_label])

    return rows


# ------------------------------------------------------------
# Image classifier + PDF embedding + section builders
# ------------------------------------------------------------


def classify_images(outdir):
    categories = {
        "performance": [],
        "drawdown": [],
        "rolling": [],
        "frontier": [],
        "correlation": [],
        "attribution_asset": [],
        "attribution_sector": [],
        "capm": [],
        "factor": [],
        "scenarios": [],
        "black_litterman": [],
        "other": [],
    }

    for root, dirs, files in os.walk(outdir):
        for fname in files:
            full = os.path.join(root, fname)
            name, ext = os.path.splitext(fname)
            key = name.lower()

            if ext.lower() not in (".png", ".jpg", ".jpeg", ""):
                continue

            if "performance_attribution_sector" in key:
                categories["attribution_sector"].append(full)
            elif "performance_attribution" in key:
                categories["attribution_asset"].append(full)
            elif "drawdown" in key or "loss_histogram" in key:
                categories["drawdown"].append(full)
            elif key.startswith("rolling") or "rolling_" in key:
                categories["rolling"].append(full)
            elif "black_litterman" in key:
                categories["black_litterman"].append(full)
            elif "efficient_frontier" in key or key == "cal":
                categories["frontier"].append(full)
            elif "corr" in key and ("matrix" in key or "heatmap" in key):
                categories["correlation"].append(full)
            elif key.startswith("capm"):
                categories["capm"].append(full)
            elif "factor" in key:
                categories["factor"].append(full)
            elif "forward_scenarios" in key:
                categories["scenarios"].append(full)
            elif (
                "growth" in key
                or "portfolio_value" in key
                or "complete_portfolio" in key
                or "active_minus_passive" in key
                or "orp" in key
            ):
                categories["performance"].append(full)
            else:
                categories["other"].append(full)

    return categories


def add_image_flowable(story, path, caption, style_caption, max_width=6.5 * inch):
    try:
        img = Image(path)
        img._restrictSize(max_width, 4 * inch)
        story.append(img)
        if caption:
            story.append(Paragraph(caption, style_caption))
        story.append(Spacer(1, 12))
    except Exception:
        return


def add_section(
    story,
    title,
    text,
    image_keys,
    image_dict,
    style_section,
    style_body,
    style_caption,
):
    story.append(Paragraph(title, style_section))
    story.append(Spacer(1, 6))

    story.append(Paragraph(text, style_body))
    story.append(Spacer(1, 10))

    for key in image_keys:
        paths = image_dict.get(key, [])
        for p in paths:
            base = os.path.basename(p)
            caption = f"{title} — {base}"
            add_image_flowable(story, p, caption, style_caption)

    story.append(Spacer(1, 18))


def add_key_charts_section(
    story,
    text,
    image_dict,
    style_section,
    style_body,
    style_caption,
):
    """Custom section for Key Charts with a fixed, professional order."""
    story.append(Paragraph("Key Charts: Portfolio Growth & Allocation", style_section))
    story.append(Spacer(1, 6))

    story.append(Paragraph(text, style_body))
    story.append(Spacer(1, 10))

    desired_order = [
        "growth_active_vs_passive",
        "active_minus_passive_cumulative",
        "correlation_matrix",
        "efficient_frontier",
        "cal",
        "orp_real_vs_expected",
        "complete_portfolio_pie",
        "growth_all_portfolios",
        "forward_scenarios",
    ]

    # Flatten all image paths into a single list
    all_paths = []
    for paths in image_dict.values():
        all_paths.extend(paths)

    for name in desired_order:
        for p in all_paths:
            base = os.path.splitext(os.path.basename(p))[0].lower()
            if base == name:
                caption = (
                    "Key Charts: Portfolio Growth & Allocation — "
                    + os.path.basename(p)
                )
                add_image_flowable(story, p, caption, style_caption)
                break

    story.append(Spacer(1, 18))


# ============================================================
# Markdown summary builder (story-ordered)
# ============================================================


def build_markdown_report(
    md_lines,
    summary_rows,
    risk_bucket_expl,
    performance_text,
    risk_text,
    drawdown_text,
    factor_text,
    attribution_text,
    sector_attr_text,
    conclusions_text,
    benchmark_label=None,
    holdings_df=None,
    bl_enabled=False,
    bl_views_df=None,
    bl_weights_df=None,
    bl_tickers=None,
):
    md_lines.append("## Summary\n\n")
    if summary_rows:
        md_lines.append("| Metric | Value |\n")
        md_lines.append("| --- | --- |\n")
        for metric, value in summary_rows:
            md_lines.append(f"| {metric} | {value} |\n")
        md_lines.append("\n")
    else:
        md_lines.append("Summary statistics were not available.\n\n")

    if benchmark_label:
        md_lines.append(f"**Benchmark:** `{benchmark_label}`\n\n")

    if holdings_df is not None and not holdings_df.empty:
        md_lines.append("**Initial Holdings:**\n\n")
        cols = [
            c
            for c in [
                "Ticker",
                "TargetWeight",
                "RealizedWeight",
                "Shares",
                "PurchasePrice",
            ]
            if c in holdings_df.columns
        ]
        if not cols:
            cols = list(holdings_df.columns)[:5]

        md_lines.append("| " + " | ".join(cols) + " |\n")
        md_lines.append("| " + " | ".join(["---"] * len(cols)) + " |\n")
        for _, row in holdings_df[cols].head(20).iterrows():
            cells = []
            for c in cols:
                v = row[c]
                if isinstance(v, float):
                    if "Weight" in c:
                        cells.append(f"{v:.4f}")
                    elif c == "PurchasePrice":
                        cells.append(f"{v:.2f}")
                    elif c == "Shares":
                        cells.append(f"{v:.2f}")
                    else:
                        cells.append(f"{v:.4f}")
                else:
                    cells.append(str(v))
            md_lines.append("| " + " | ".join(cells) + " |\n")
        md_lines.append("\n")

    write_md_section(md_lines, "Risk Bucket", risk_bucket_expl)

    write_md_section(
        md_lines,
        "Key Charts: Portfolio Growth & Allocation",
        (
            "This section summarizes how the portfolio and benchmark evolved over time, "
            "along with the composition of the Optimal Risky Portfolio (ORP) and the "
            "complete portfolio. " + performance_text
        ),
    )

    write_md_section(md_lines, "Drawdown & Tail Risk", drawdown_text)

    write_md_section(
        md_lines,
        "Rolling Risk Analytics",
        (
            "Rolling analytics highlight how volatility and correlations evolved over time, "
            "helping identify regime shifts or changes in portfolio behavior. " + risk_text
        ),
    )

    write_md_section(
        md_lines,
        "Performance Attribution (Assets)",
        attribution_text,
    )

    write_md_section(
        md_lines,
        "Performance Attribution (Sectors)",
        sector_attr_text,
    )

    write_md_section(
        md_lines,
        "Multi-Factor Regression & Style Exposures",
        factor_text,
    )

    write_md_section(
        md_lines,
        "CAPM Regression",
        (
            "CAPM regressions relate each asset’s excess return to the market’s excess return. "
            "Alpha represents return unexplained by the market, while beta measures sensitivity "
            "to broad market moves."
        ),
    )

    write_md_section(
        md_lines,
        "Efficient Frontier & Optimal Risky Portfolio",
        (
            "The efficient frontier illustrates the best achievable combinations of risk and return. "
            "The Optimal Risky Portfolio (ORP) lies at the maximum Sharpe point and reflects the most "
            "efficient risk-taking position."
        ),
    )

    # ---- Black-Litterman section (only when enabled) ----
    if bl_enabled:
        bl_text = interpret_black_litterman(bl_weights_df, bl_views_df, bl_tickers=bl_tickers)
        write_md_section(
            md_lines,
            "Black-Litterman Model (Optional)",
            bl_text,
        )

        # Allocation table
        if bl_weights_df is not None and not bl_weights_df.empty:
            md_lines.append("**Black-Litterman Allocation:**\n\n")
            df = bl_weights_df.copy()

            ticker_col = None
            for c in df.columns:
                if c.lower() in ("ticker", "asset", "name", "symbol"):
                    ticker_col = c
                    break
            if ticker_col is None and bl_tickers is not None:
                labels = [str(t) for t in bl_tickers]
                if len(labels) >= len(df):
                    df.insert(0, "Ticker", labels[: len(df)])
                    ticker_col = "Ticker"

            weight_col = None
            for c in df.columns:
                if c.lower() in ("weight", "weights", "bl_weight", "bl_weights"):
                    weight_col = c
                    break
            if weight_col is None:
                num_cols = df.select_dtypes(include="number").columns
                if len(num_cols) >= 1:
                    weight_col = num_cols[0]

            cols = []
            if ticker_col is not None:
                cols.append(ticker_col)
            if weight_col is not None:
                cols.append(weight_col)

            # Add one more numeric column if available and distinct
            for c in df.columns:
                if c in cols:
                    continue
                if df[c].dtype.kind in "if" and len(cols) < 3:
                    cols.append(c)
                    break

            if cols:
                md_lines.append("| " + " | ".join(cols) + " |\n")
                md_lines.append("| " + " | ".join(["---"] * len(cols)) + " |\n")
                for _, row in df[cols].head(20).iterrows():
                    cells = []
                    for c in cols:
                        v = row[c]
                        if c == weight_col and isinstance(v, (float, int)):
                            cells.append(f"{float(v):.4%}")
                        else:
                            cells.append(str(v))
                    md_lines.append("| " + " | ".join(cells) + " |\n")
                md_lines.append("\n")

        if bl_views_df is not None and not bl_views_df.empty:
            md_lines.append("**Black-Litterman Views Used in This Run:**\n\n")
            cols = list(bl_views_df.columns)[:6]
            if cols:
                md_lines.append("| " + " | ".join(cols) + " |\n")
                md_lines.append("| " + " | ".join(["---"] * len(cols)) + " |\n")
                for _, row in bl_views_df[cols].head(15).iterrows():
                    cells = [str(row[c]) for c in cols]
                    md_lines.append("| " + " | ".join(cells) + " |\n")
                md_lines.append("\n")

    write_md_section(
        md_lines,
        "Forward-Looking Scenarios",
        (
            "Simulated scenarios help illustrate a realistic range of future outcomes based on "
            "historical return distributions. These results help guide planning and expectation setting."
        ),
    )

    write_md_section(
        md_lines,
        "Conclusions & Recommendations",
        conclusions_text,
    )

    md_lines.append("## Appendix\n\n")
    md_lines.append(
        "Additional charts, diagnostics, and supporting visuals are included in the appendix.\n\n"
    )

    return md_lines


# ------------------------------------------------------------
# Final assembly: Markdown + PDF generation
# ------------------------------------------------------------


def generate_report(outdir=OUTPUT_DIR, config_path="config.json"):
    print("[report] Generating full report...")

    benchmark_label = None
    holdings_df = None
    start_date_cfg = None
    end_date_cfg = None
    cfg = None
    try:
        if config_path and os.path.exists(config_path):
            with open(config_path, "r") as f:
                cfg = json.load(f)
    except Exception as e:
        print(f"[report] Warning: could not load config '{config_path}': {e}")

    if cfg:
        benchmark_label = cfg.get("benchmark") or cfg.get("passive_benchmark")
        start_date_cfg = cfg.get("start") or cfg.get("start_date")
        end_date_cfg = cfg.get("end") or cfg.get("end_date")

    holdings_df = load_csv_smart(outdir, "holdings_table")
    if (holdings_df is None or holdings_df.empty) and cfg:
        w_dict = cfg.get("active_portfolio", {}).get("weights", {})
        if isinstance(w_dict, dict) and w_dict:
            holdings_df = pd.DataFrame(
                {"Ticker": list(w_dict.keys()), "TargetWeight": list(w_dict.values())}
            )

    summary = load_json_smart(outdir, "summary")
    summary_stats = load_csv_smart(outdir, "summary_stats")
    dd_df = load_csv_smart(outdir, "drawdown_tail_metrics")
    capm_df = load_csv_smart(outdir, "capm_results")

    factor_df = load_csv_smart(outdir, "factor_regression_ff5")
    if factor_df is None:
        factor_df = load_csv_smart(outdir, "factor_regression_ff3")

    asset_attr = load_csv_smart(outdir, "performance_attribution")
    sector_attr = load_csv_smart(outdir, "performance_attribution_sector")

    # --- Black-Litterman outputs (optional) ---
    bl_weights_df = load_csv_smart(outdir, "black_litterman_weights")
    bl_views_df = load_csv_smart(outdir, "black_litterman_views")
    bl_frontier_path = _find_matching_file(
        outdir, "black_litterman_efficient_frontier", exts=(".png", ".jpg", ".jpeg", "")
    )
    bl_enabled = False
    if bl_weights_df is not None and not bl_weights_df.empty:
        bl_enabled = True
    if bl_views_df is not None and not bl_views_df.empty:
        bl_enabled = True
    if bl_frontier_path is not None:
        bl_enabled = True

    # derive BL tickers from config if possible
    bl_tickers = None
    if cfg:
        ap = cfg.get("active_portfolio") or {}
        if isinstance(ap, dict):
            bl_tickers = ap.get("tickers")
        if not bl_tickers:
            bl_tickers = cfg.get("tickers")
        if isinstance(bl_tickers, dict):
            bl_tickers = list(bl_tickers.keys())

    risk_bucket_label, risk_bucket_expl = compute_risk_bucket(summary, dd_df)
    performance_text = interpret_performance(summary)
    risk_text = interpret_risk(summary)
    drawdown_text = interpret_drawdown(dd_df)
    factor_text = interpret_factors(factor_df)
    attribution_text = interpret_attribution(asset_attr)
    sector_attr_text = interpret_sector_attribution(sector_attr)
    conclusions_text = interpret_conclusions(
        summary, dd_df, asset_attr, sector_attr, factor_df
    )

    summary_rows = build_summary_rows(
        summary,
        dd_df,
        asset_attr,
        sector_attr,
        risk_bucket_label,
        benchmark_ticker=benchmark_label,
        start_date=start_date_cfg,
        end_date=end_date_cfg,
    )

    # ----------------- Markdown -----------------
    md_lines = []
    md_lines.append("# Portfolio Analysis Report\n\n")
    md_lines.append(
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    )

    md_lines = build_markdown_report(
        md_lines,
        summary_rows,
        risk_bucket_expl,
        performance_text,
        risk_text,
        drawdown_text,
        factor_text,
        attribution_text,
        sector_attr_text,
        conclusions_text,
        benchmark_label=benchmark_label,
        holdings_df=holdings_df,
        bl_enabled=bl_enabled,
        bl_views_df=bl_views_df,
        bl_weights_df=bl_weights_df,
        bl_tickers=bl_tickers,
    )

    md_path = os.path.join(outdir, "report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"[report] Markdown written to: {md_path}")

    # ----------------- PDF -----------------
    pdf_path = os.path.join(outdir, "report.pdf")

    styles = getSampleStyleSheet()
    style_body = styles["BodyText"]
    style_title = styles["Heading1"]
    style_section = styles["Heading2"]

    style_caption = ParagraphStyle(
        "Caption",
        parent=style_body,
        fontSize=8,
        textColor=colors.grey,
        spaceBefore=2,
        spaceAfter=6,
    )

    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    story = []

    story.append(Paragraph("Portfolio Analysis Report", style_title))
    story.append(
        Paragraph(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            style_body,
        )
    )
    story.append(Spacer(1, 20))

    story.append(Paragraph("Summary", style_section))
    story.append(Spacer(1, 6))

    if summary_rows:
        header = [["Metric", "Value"]]
        table = Table(header + summary_rows, colWidths=[2.5 * inch, 3.5 * inch])
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ]
            )
        )
        story.append(table)
    else:
        story.append(Paragraph("Summary statistics were not available.", style_body))

    story.append(Spacer(1, 12))

    if benchmark_label or (holdings_df is not None and not holdings_df.empty):
        story.append(Paragraph("Holdings & Benchmark", style_section))
        story.append(Spacer(1, 6))

        if benchmark_label:
            story.append(
                Paragraph(f"Benchmark: <b>{benchmark_label}</b>", style_body)
            )
            story.append(Spacer(1, 6))

        if start_date_cfg or end_date_cfg:
            date_text_parts = []
            if start_date_cfg:
                date_text_parts.append(f"Start: {start_date_cfg}")
            if end_date_cfg:
                date_text_parts.append(f"End: {end_date_cfg}")
            story.append(
                Paragraph(
                    "Backtest period: " + ", ".join(date_text_parts), style_body
                )
            )
            story.append(Spacer(1, 6))

        if holdings_df is not None and not holdings_df.empty:
            cols = [
                c
                for c in [
                    "Ticker",
                    "TargetWeight",
                    "RealizedWeight",
                    "Shares",
                    "PurchasePrice",
                ]
                if c in holdings_df.columns
            ]
            if not cols:
                cols = list(holdings_df.columns)[:5]

            header_cols = cols
            data_rows = []
            for _, row in holdings_df[cols].head(20).iterrows():
                row_vals = []
                for c in cols:
                    v = row[c]
                    if isinstance(v, float):
                        if "Weight" in c:
                            row_vals.append(f"{v:.4f}")
                        elif c == "PurchasePrice":
                            row_vals.append(f"{v:.2f}")
                        elif c == "Shares":
                            row_vals.append(f"{v:.2f}")
                        else:
                            row_vals.append(f"{v:.4f}")
                    else:
                        row_vals.append(str(v))
                data_rows.append(row_vals)

            if data_rows:
                h_table = Table([header_cols] + data_rows)
                h_table.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                        ]
                    )
                )
                story.append(h_table)

        story.append(Spacer(1, 20))

    story.append(Paragraph("Risk Bucket", style_section))
    story.append(Paragraph(risk_bucket_expl, style_body))
    story.append(Spacer(1, 20))

    img_cats = classify_images(outdir)

    key_charts_text = (
        "These charts show how the portfolio and benchmark evolved over time, "
        "the relationship between risk and return on the efficient frontier, and "
        "how the Optimal Risky Portfolio (ORP) and complete portfolio allocate capital. "
        + performance_text
    )
    add_key_charts_section(
        story,
        key_charts_text,
        img_cats,
        style_section,
        style_body,
        style_caption,
    )

    add_section(
        story,
        "Drawdown & Tail Risk",
        drawdown_text,
        ["drawdown"],
        img_cats,
        style_section,
        style_body,
        style_caption,
    )

    rolling_text = (
        "Rolling analytics highlight how volatility and correlations evolved over time, "
        "helping identify regime shifts or changes in portfolio behavior. " + risk_text
    )
    add_section(
        story,
        "Rolling Risk Analytics",
        rolling_text,
        ["rolling"],
        img_cats,
        style_section,
        style_body,
        style_caption,
    )

    add_section(
        story,
        "Performance Attribution (Assets)",
        attribution_text,
        ["attribution_asset"],
        img_cats,
        style_section,
        style_body,
        style_caption,
    )

    add_section(
        story,
        "Performance Attribution (Sectors)",
        sector_attr_text,
        ["attribution_sector"],
        img_cats,
        style_section,
        style_body,
        style_caption,
    )

    add_section(
        story,
        "Multi-Factor Regression & Style Exposures",
        factor_text,
        [],
        img_cats,
        style_section,
        style_body,
        style_caption,
    )

    if capm_df is not None:
        capm_text = (
            "CAPM analysis relates each asset’s excess return to the market’s excess return. "
            "Beta measures sensitivity to broad market moves, while alpha represents return "
            "unexplained by the market. The scatter plots illustrate this relationship for each holding."
        )
        add_section(
            story,
            "CAPM Regression & Scatter Plots",
            capm_text,
            ["capm"],
            img_cats,
            style_section,
            style_body,
            style_caption,
        )

    # ---- Black-Litterman section in PDF (only when enabled) ----
    if bl_enabled:
        bl_text = interpret_black_litterman(bl_weights_df, bl_views_df, bl_tickers=bl_tickers)
        add_section(
            story,
            "Black-Litterman Model (Optional)",
            bl_text,
            ["black_litterman"],
            img_cats,
            style_section,
            style_body,
            style_caption,
        )

        # Allocation table
        if bl_weights_df is not None and not bl_weights_df.empty:
            story.append(Paragraph("Black-Litterman Allocation", style_section))
            story.append(Spacer(1, 6))
            df = bl_weights_df.copy()

            ticker_col = None
            for c in df.columns:
                if c.lower() in ("ticker", "asset", "name", "symbol"):
                    ticker_col = c
                    break
            if ticker_col is None and bl_tickers is not None:
                labels = [str(t) for t in bl_tickers]
                if len(labels) >= len(df):
                    df.insert(0, "Ticker", labels[: len(df)])
                    ticker_col = "Ticker"

            weight_col = None
            for c in df.columns:
                if c.lower() in ("weight", "weights", "bl_weight", "bl_weights"):
                    weight_col = c
                    break
            if weight_col is None:
                num_cols = df.select_dtypes(include="number").columns
                if len(num_cols) >= 1:
                    weight_col = num_cols[0]

            cols = []
            if ticker_col is not None:
                cols.append(ticker_col)
            if weight_col is not None:
                cols.append(weight_col)

            # Add one more numeric column if available and distinct
            for c in df.columns:
                if c in cols:
                    continue
                if df[c].dtype.kind in "if" and len(cols) < 3:
                    cols.append(c)
                    break

            data_rows = []
            if cols:
                for _, row in df[cols].head(20).iterrows():
                    row_vals = []
                    for c in cols:
                        v = row[c]
                        if c == weight_col and isinstance(v, (float, int)):
                            row_vals.append(f"{float(v):.4%}")
                        else:
                            row_vals.append(str(v))
                    data_rows.append(row_vals)

            if data_rows:
                tbl = Table([cols] + data_rows)
                tbl.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                        ]
                    )
                )
                story.append(tbl)
                story.append(Spacer(1, 18))

        if bl_views_df is not None and not bl_views_df.empty:
            story.append(Paragraph("Black-Litterman Views", style_section))
            story.append(Spacer(1, 6))
            cols = list(bl_views_df.columns)[:5]
            data_rows = []
            for _, row in bl_views_df[cols].head(15).iterrows():
                data_rows.append([str(row[c]) for c in cols])
            if data_rows:
                tbl = Table([cols] + data_rows)
                tbl.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                        ]
                    )
                )
                story.append(tbl)
                story.append(Spacer(1, 18))

    story.append(Paragraph("Conclusions & Recommendations", style_section))
    story.append(Spacer(1, 6))
    story.append(Paragraph(conclusions_text, style_body))
    story.append(Spacer(1, 20))

    story.append(Paragraph("Appendix: Additional Charts & Diagnostics", style_section))
    story.append(Spacer(1, 10))
    story.append(
        Paragraph(
            "This appendix includes additional charts and diagnostics that support the analysis "
            "presented in the main report.",
            style_body,
        )
    )
    story.append(Spacer(1, 12))

    for p in img_cats.get("other", []):
        add_image_flowable(
            story,
            p,
            "Additional Diagnostic Chart",
            style_caption,
        )

    doc.build(story)
    print(f"[report] PDF written to: {pdf_path}")
    print("[report] Done.")
