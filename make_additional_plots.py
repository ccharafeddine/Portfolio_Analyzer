import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def run_additional_plots(
    outdir: str = "outputs",
    orp_weights: dict | None = None,
    y_cp: float = 0.8,
) -> None:
    """
    Generate additional plots:

    1) Correlation heatmap of monthly returns using clean_prices.csv
    2) Complete Portfolio allocation pie chart, using ORP weights and y_cp.
       If ORP includes shorts, the pie uses absolute exposures and labels
       short positions with "(short)" so all wedge sizes are non-negative.
    """

    prices_path = os.path.join(outdir, "clean_prices.csv")
    if not os.path.exists(prices_path):
        print("[additional_plots] clean_prices.csv not found; skipping.")
        return

    prices = pd.read_csv(prices_path, parse_dates=["Date"], index_col="Date")

    # ------------------------------------------------------------------
    # 1) Correlation heatmap of monthly returns
    # ------------------------------------------------------------------
    try:
        monthly = (
            prices.resample("ME")
            .last()
            .pct_change(fill_method=None)  # avoid deprecated pad behaviour
            .dropna(how="all")
        )
    except Exception as e:
        print(f"[additional_plots] Could not compute monthly returns: {e}")
        monthly = pd.DataFrame()

    if not monthly.empty:
        corr = monthly.corr()

        plt.figure(figsize=(12, 10))
        sns.heatmap(
            corr,
            annot=True,
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            linewidths=0.5,
            linecolor="black",
        )
        plt.title("Correlation Matrix of Monthly Returns")
        plt.tight_layout()
        corr_path = os.path.join(outdir, "correlation_matrix.png")
        plt.savefig(corr_path, dpi=300)
        plt.close()
        print(f"Saved {corr_path}")
    else:
        print("[additional_plots] Monthly returns empty; skipping correlation heatmap.")

    # ------------------------------------------------------------------
    # 2) Complete Portfolio pie chart (if ORP weights provided)
    # ------------------------------------------------------------------
    if orp_weights is None:
        print("[additional_plots] No ORP weights provided; skipping CP pie chart.")
        return

    if not isinstance(orp_weights, dict):
        try:
            orp_weights = dict(orp_weights)
        except Exception as e:
            print(f"[additional_plots] Could not interpret orp_weights: {e}")
            return

    if len(orp_weights) == 0:
        print("[additional_plots] Empty ORP weights; skipping CP pie chart.")
        return

    total_orp = float(sum(orp_weights.values()))
    if abs(total_orp) < 1e-10:
        print("[additional_plots] ORP weights sum ~ 0; skipping CP pie chart.")
        return

    # Normalise ORP to sum to 1
    orp_norm = {ticker: w / total_orp for ticker, w in orp_weights.items()}

    # Mix ORP with risk-free: y_cp in risky, (1-y_cp) in risk-free
    cp_weights: dict[str, float] = {
        ticker: float(y_cp) * float(w) for ticker, w in orp_norm.items()
    }
    cp_weights["Risk-Free"] = float(1.0 - float(y_cp))

    # Build plotting weights as absolute exposures and mark shorts
    plot_weights: dict[str, float] = {}
    for ticker, w in cp_weights.items():
        if abs(w) <= 1e-6:
            continue
        if w < 0:
            plot_weights[f"{ticker} (short)"] = -w
        else:
            plot_weights[ticker] = w

    total_plot = float(sum(plot_weights.values()))
    if total_plot <= 0:
        print("[additional_plots] No positive exposures for CP pie; skipping.")
        return

    # Normalise for pie chart
    plot_weights = {t: w / total_plot for t, w in plot_weights.items()}

    # --- NEW: drop effectively-zero slices so we don't show 0.0% wedges ---
    # Anything below 0.05% of the portfolio is removed from the pie.
    min_slice = 0.0005  # 0.05%
    plot_weights = {t: w for t, w in plot_weights.items() if w >= min_slice}

    if not plot_weights:
        print(
            "[additional_plots] All CP exposures below min_slice; "
            "skipping pie chart."
        )
        return

    # Renormalize after dropping tiny slices
    total_plot = float(sum(plot_weights.values()))
    plot_weights = {t: w / total_plot for t, w in plot_weights.items()}

    labels = list(plot_weights.keys())
    sizes = list(plot_weights.values())

    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
    plt.title(f"Complete Portfolio Allocation (y = {y_cp:.2f}, abs exposure)")
    plt.tight_layout()
    cp_path = os.path.join(outdir, "complete_portfolio_pie.png")
    plt.savefig(cp_path, dpi=300)
    plt.close()
    print(f"Saved {cp_path}")
