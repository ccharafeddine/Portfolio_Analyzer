import os
import json
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from plotting import (
    plot_drawdown_curves,
    plot_loss_histogram_with_var,
)


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _normalize_to_million(series: pd.Series) -> pd.Series:
    """
    Scale any value series so its first non-NaN value = $1,000,000.

    If the series is entirely NaN or starts at 0, return a NaN series.
    """
    s = series.dropna()
    if s.empty:
        return pd.Series(np.nan, index=series.index)
    base = float(s.iloc[0])
    if base == 0:
        return pd.Series(np.nan, index=series.index)
    return series / base * 1_000_000.0


def _load_value_series(path: str) -> Optional[pd.Series]:
    """
    Load a single portfolio value series from a CSV with a Date index.

    Returns None if the path does not exist or no numeric column is found.
    """
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    num_cols = df.select_dtypes(include=["number"]).columns
    if len(num_cols) == 0:
        return None
    s = df[num_cols[0]].astype(float).sort_index()
    return s


# ------------------------------------------------------------
# Main performance plots
# ------------------------------------------------------------
def run_performance_plots(outdir: str = "outputs", rf_annual: float = 0.04) -> None:
    """
    Create the core performance plots:

      * growth_active_vs_passive.png
      * active_minus_passive_cumulative.png
      * orp_real_vs_expected.png           (if ORP can be constructed)
      * growth_all_portfolios.png          (Active / Passive / ORP / Complete)
    """
    os.makedirs(outdir, exist_ok=True)

    # --------------------------------------------------------
    # Active & Passive value series
    # --------------------------------------------------------
    active = _load_value_series(os.path.join(outdir, "active_portfolio_value.csv"))
    passive = _load_value_series(os.path.join(outdir, "passive_portfolio_value.csv"))

    if active is None or passive is None:
        print("[performance_plots] Missing active or passive series; skipping plots.")
        return

    df_ap = pd.DataFrame({"Active": active, "Passive": passive}).dropna(how="any")

    # --------------------------------------------------------
    # Plot 1: Growth of $1M, Active vs Passive
    # --------------------------------------------------------
    growth_ap = df_ap.apply(_normalize_to_million)

    plt.figure(figsize=(12, 7))
    plt.plot(growth_ap.index, growth_ap["Active"], label="Active")
    plt.plot(growth_ap.index, growth_ap["Passive"], label="Passive")
    ax = plt.gca()
    ax.yaxis.set_major_formatter(
        mtick.FuncFormatter(lambda x, pos: f"${x/1e6:.1f}M")
    )
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value (Millions)")
    plt.title("Growth of $1,000,000: Active vs Passive")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "growth_active_vs_passive.png"), dpi=300)
    plt.close()

    # --------------------------------------------------------
    # Plot 2: Active − Passive cumulative outperformance
    # --------------------------------------------------------
    diff = df_ap["Active"] - df_ap["Passive"]
    diff_norm = diff - float(diff.iloc[0])

    plt.figure(figsize=(12, 7))
    plt.plot(diff_norm.index, diff_norm)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(
        mtick.FuncFormatter(lambda x, pos: f"${x/1e6:.1f}M")
    )
    plt.xlabel("Date")
    plt.ylabel("Cumulative Outperformance (Millions)")
    plt.title("Active − Passive: Cumulative Outperformance")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "active_minus_passive_cumulative.png"), dpi=300)
    plt.close()

    # --------------------------------------------------------
    # ORP & Complete series from clean prices + summary.json
    # --------------------------------------------------------
    prices_path = os.path.join(outdir, "clean_prices.csv")
    if not os.path.exists(prices_path):
        print("[performance_plots] clean_prices.csv not found; skipping ORP/Complete.")
        orp_values = None
        complete_values = None
    else:
        prices = pd.read_csv(
            prices_path, parse_dates=["Date"], index_col="Date"
        ).sort_index()
        prices = prices.ffill()

        summary_path = os.path.join(outdir, "summary.json")
        if not os.path.exists(summary_path):
            print("[performance_plots] summary.json not found; skipping ORP/Complete.")
            orp_values = None
            complete_values = None
        else:
            with open(summary_path, "r") as f:
                summary = json.load(f)

            w_dict = summary.get("max_sharpe_weights", {})
            if not w_dict:
                print(
                    "[performance_plots] No max_sharpe_weights in summary; "
                    "skipping ORP/Complete."
                )
                orp_values = None
                complete_values = None
            else:
                w = pd.Series(w_dict, dtype=float)
                # Drop tickers with exactly zero weight
                w = w[w != 0.0]

                available = [t for t in w.index if t in prices.columns]
                if not available:
                    print(
                        "[performance_plots] ORP tickers not present in clean_prices; "
                        "skipping ORP/Complete."
                    )
                    orp_values = None
                    complete_values = None
                else:
                    w = w.loc[available]
                    prices_orp = prices[available].dropna(how="all")

                    # Align to Active/Passive dates to keep everything consistent
                    common_index = df_ap.index.intersection(prices_orp.index)
                    if common_index.empty:
                        print(
                            "[performance_plots] No overlapping dates for ORP and "
                            "Active/Passive; skipping ORP/Complete."
                        )
                        orp_values = None
                        complete_values = None
                    else:
                        prices_orp = prices_orp.loc[common_index]

                        # --- Realized ORP values from daily prices ---
                        alloc = w * 1_000_000.0
                        initial_prices = prices_orp.iloc[0]
                        shares = alloc / initial_prices
                        orp_values = (prices_orp * shares).sum(axis=1)
                        orp_values.name = "ORP"

                        # --- Expected ORP path (monthly, then upsampled) ---
                        monthly_orp_prices = (
                            prices_orp.resample("M").last().pct_change().dropna()
                        )
                        if monthly_orp_prices.empty:
                            expected_orp_daily = None
                        else:
                            r_orp_monthly = monthly_orp_prices.dot(w)
                            expected_orp_monthly = 1_000_000.0 * (
                                1.0 + r_orp_monthly
                            ).cumprod()
                            # Upsample to daily dates using forward-fill
                            expected_orp_daily = expected_orp_monthly.reindex(
                                orp_values.index, method="ffill"
                            )

                        # --- Risk-free (cash) series for the Complete portfolio ---
                        rf_daily = (1.0 + rf_annual) ** (1.0 / 252.0) - 1.0
                        rf_path = 1_000_000.0 * (1.0 + rf_daily) ** np.arange(
                            len(orp_values)
                        )
                        rf_series = pd.Series(rf_path, index=orp_values.index)

                        y_cp = float(summary.get("y_cp", 1.0))
                        complete_values = y_cp * orp_values + (1.0 - y_cp) * rf_series
                        complete_values.name = "Complete"

                        # Save realized ORP and Complete series
                        pd.DataFrame({"ORP_Value": orp_values}).to_csv(
                            os.path.join(outdir, "orp_value_realized.csv"),
                            index_label="Date",
                        )
                        pd.DataFrame({"Complete_Value": complete_values}).to_csv(
                            os.path.join(outdir, "complete_portfolio_value.csv"),
                            index_label="Date",
                        )

                        # --- Plot ORP Real vs Expected if we have both ---
                        if (
                            expected_orp_daily is not None
                            and not expected_orp_daily.empty
                        ):
                            plt.figure(figsize=(10, 6))
                            plt.plot(
                                orp_values.index,
                                orp_values,
                                label="ORP (realized)",
                            )
                            plt.plot(
                                expected_orp_daily.index,
                                expected_orp_daily,
                                linestyle="--",
                                label="Expected ORP (monthly-smoothed)",
                            )
                            ax = plt.gca()
                            ax.yaxis.set_major_formatter(
                                mtick.FuncFormatter(
                                    lambda x, pos: f"${x/1e6:.1f}M"
                                )
                            )
                            plt.xlabel("Date")
                            plt.ylabel("Portfolio Value (Millions)")
                            plt.title(
                                "Optimal Risky Portfolio: Realized vs Expected"
                            )
                            plt.legend()
                            plt.grid(True, alpha=0.3)
                            plt.tight_layout()
                            plt.savefig(
                                os.path.join(outdir, "orp_real_vs_expected.png"),
                                dpi=300,
                            )
                            plt.close()
                        else:
                            print(
                                "[performance_plots] Could not build expected ORP "
                                "path; skipping orp_real_vs_expected plot."
                            )

    # --------------------------------------------------------
    # Plot: Growth of all portfolios (Active / Passive / ORP / Complete)
    # --------------------------------------------------------
    series_dict = {
        "Active": active,
        "Passive": passive,
    }

    # Only add ORP/Complete if they exist
    if "orp_values" in locals() and isinstance(orp_values, pd.Series):
        series_dict["ORP"] = orp_values
    if "complete_values" in locals() and isinstance(complete_values, pd.Series):
        series_dict["Complete"] = complete_values

    combined_index = None
    for s in series_dict.values():
        if combined_index is None:
            combined_index = s.index
        else:
            combined_index = combined_index.union(s.index)

    combined_df = pd.DataFrame(
        {name: s.reindex(combined_index).ffill() for name, s in series_dict.items()}
    )

    growth_all = combined_df.apply(_normalize_to_million)

    plt.figure(figsize=(14, 8))
    for name, col in growth_all.items():
        if col.dropna().empty:
            continue
        plt.plot(growth_all.index, col, label=name)

    ax = plt.gca()
    ax.yaxis.set_major_formatter(
        mtick.FuncFormatter(lambda x, pos: f"${x/1e6:.1f}M")
    )
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value (Millions)")
    plt.title("Growth of $1,000,000: Active vs Passive vs ORP vs Complete")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "growth_all_portfolios.png"), dpi=300)
    plt.close()

    print("[performance_plots] Saved performance plots in", outdir)


# ------------------------------------------------------------
# Drawdown & tail-risk plots (used by main.py)
# ------------------------------------------------------------
def run_drawdown_tail_plots(
    outdir: str = "outputs",
    var_level: float = 0.95,
    focus_portfolio: str = "Active",
) -> None:
    """
    Build drawdown & tail-risk charts from the CSVs written by
    compute_all_portfolio_drawdown_and_tail_risk.

    Produces:
      * drawdown_curves.png
      * loss_histogram_<focus_portfolio>.png
    """
    os.makedirs(outdir, exist_ok=True)

    dd_series_path = os.path.join(outdir, "drawdown_series.csv")
    metrics_path = os.path.join(outdir, "drawdown_tail_metrics.csv")

    if not os.path.exists(dd_series_path):
        print("[drawdown_tail_plots] drawdown_series.csv not found; skipping.")
        return

    try:
        dd_df = pd.read_csv(
            dd_series_path, parse_dates=["Date"], index_col="Date"
        )
    except Exception as e:
        print(f"[drawdown_tail_plots] Failed to read drawdown_series.csv: {e}")
        return

    # 1) Drawdown curves for all portfolios
    try:
        plot_drawdown_curves(
            dd_df,
            os.path.join(outdir, "drawdown_curves.png"),
        )
    except Exception as e:
        print(f"[drawdown_tail_plots] Failed to plot drawdown curves: {e}")

    # 2) Tail-loss histogram for focus_portfolio (if we can)
    #    Need both daily returns and VaR/CVaR from metrics.
    if not os.path.exists(metrics_path):
        print("[drawdown_tail_plots] drawdown_tail_metrics.csv not found; skipping histogram.")
        return

    try:
        metrics_df = pd.read_csv(metrics_path)
    except Exception as e:
        print(f"[drawdown_tail_plots] Failed to read drawdown_tail_metrics.csv: {e}")
        return

    row = metrics_df.loc[metrics_df["Portfolio"] == focus_portfolio]
    if row.empty:
        print(
            f"[drawdown_tail_plots] No metrics row for portfolio '{focus_portfolio}'; "
            "skipping histogram."
        )
        return

    suffix = str(int(round(var_level * 100)))
    var_col = f"VaR_{suffix}"
    cvar_col = f"CVaR_{suffix}"

    if var_col not in row.columns or cvar_col not in row.columns:
        print(
            f"[drawdown_tail_plots] Columns {var_col}/{cvar_col} not found in metrics; "
            "skipping histogram."
        )
        return

    var_value = float(row.iloc[0][var_col])
    cvar_value = float(row.iloc[0][cvar_col])

    # Map focus_portfolio to its value CSV
    value_file_map = {
        "Active": "active_portfolio_value.csv",
        "Passive": "passive_portfolio_value.csv",
        "ORP": "orp_value_realized.csv",
        "Complete": "complete_portfolio_value.csv",
    }
    value_fname = value_file_map.get(focus_portfolio)
    if value_fname is None:
        print(
            f"[drawdown_tail_plots] No value-file mapping for '{focus_portfolio}'; "
            "skipping histogram."
        )
        return

    value_path = os.path.join(outdir, value_fname)
    if not os.path.exists(value_path):
        print(
            f"[drawdown_tail_plots] Value file {value_fname} not found; "
            "skipping histogram."
        )
        return

    try:
        vals = pd.read_csv(
            value_path, parse_dates=["Date"], index_col="Date"
        )
        num_cols = vals.select_dtypes(include=["number"]).columns
        if len(num_cols) == 0:
            print(
                f"[drawdown_tail_plots] No numeric columns in {value_fname}; "
                "skipping histogram."
            )
            return
        value_series = vals[num_cols[0]].astype(float).sort_index()
        rets = value_series.pct_change().dropna()
    except Exception as e:
        print(f"[drawdown_tail_plots] Failed to derive returns from {value_fname}: {e}")
        return

    out_name = f"loss_histogram_{focus_portfolio.lower()}.png"
    try:
        plot_loss_histogram_with_var(
            rets,
            var_level=var_level,
            var_value=var_value,
            cvar_value=cvar_value,
            fname=os.path.join(outdir, out_name),
            title=f"{focus_portfolio} daily returns with VaR/CVaR ({int(var_level*100)}%)",
        )
    except Exception as e:
        print(f"[drawdown_tail_plots] Failed to plot loss histogram: {e}")
