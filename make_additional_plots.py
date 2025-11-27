import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def run_additional_plots(
    outdir: str = "outputs",
    orp_weights: dict | None = None,
    y_cp: float = 0.8
):
    """
    Generate:
      1) Correlation heatmap of monthly returns
      2) Complete Portfolio pie chart (if ORP weights provided)

    Parameters
    ----------
    outdir : str
        Directory where outputs (clean_prices.csv, plots) live.
    orp_weights : dict | None
        Dict of {ticker: weight} for the ORP (max-Sharpe portfolio).
        If None, only the correlation matrix is plotted.
    y_cp : float
        Fraction of total wealth allocated to ORP in the Complete Portfolio.
        The remainder (1 - y_cp) goes to the risk-free asset.
    """
    prices_path = os.path.join(outdir, "clean_prices.csv")
    prices = pd.read_csv(prices_path, parse_dates=["Date"], index_col="Date")

    # ---- 1) Correlation heatmap ----
    monthly = prices.resample("ME").last().pct_change().dropna()
    corr = monthly.corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        linecolor="black"
    )
    plt.title("Correlation Matrix of Monthly Returns")
    plt.tight_layout()
    corr_path = os.path.join(outdir, "correlation_matrix.png")
    plt.savefig(corr_path, dpi=300)
    plt.close()
    print(f"Saved {corr_path}")

    # ---- 2) Complete Portfolio pie chart ----
    if orp_weights is not None:
        # Scale ORP weights by y_cp to get CP risky weights
        cp_weights = {ticker: y_cp * w for ticker, w in orp_weights.items()}

        # Add risk-free slice
        rf_weight = 1.0 - y_cp
        cp_weights["Risk-Free"] = rf_weight

        # Filter out tiny slices so the pie is readable 
        min_slice = 0.005  # 0.5%
        filtered = {k: v for k, v in cp_weights.items() if v >= min_slice}

        # Renormalize so slices sum to 1, in case we dropped tiny ones
        total = sum(filtered.values())
        if total <= 0:
            print("Complete Portfolio weights sum to 0 — skipping CP pie chart.")
            return
        norm_weights = {k: v / total for k, v in filtered.items()}

        labels = list(norm_weights.keys())
        sizes = list(norm_weights.values())

        plt.figure(figsize=(8, 8))
        plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
        plt.title(f"Complete Portfolio Allocation (y = {y_cp:.2f})")
        plt.tight_layout()
        cp_path = os.path.join(outdir, "complete_portfolio_pie.png")
        plt.savefig(cp_path, dpi=300)
        plt.close()
        print(f"Saved {cp_path}")
