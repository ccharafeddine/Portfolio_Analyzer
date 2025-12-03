import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os  # to infer asset name from filename


def plot_efficient_frontier(R: np.ndarray, V: np.ndarray, fname: str):
    plt.figure()
    plt.plot(V, R, marker="o", linestyle="-")
    plt.xlabel("Annualized Volatility")
    plt.ylabel("Annualized Return")
    plt.title("Efficient Frontier")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fname, dpi=160)
    plt.close()


def plot_cal(
    rf: float,
    orp_return: float,
    orp_vol: float,
    fname: str,
    frontier_R: np.ndarray | None = None,
    frontier_V: np.ndarray | None = None,
):
    plt.figure()

    # Capital Allocation Line
    max_vol = 0.4
    if orp_vol > 0:
        slope = (orp_return - rf) / orp_vol
        x_vals = np.linspace(0, max_vol, 100)
        y_vals = rf + slope * x_vals
        plt.plot(x_vals, y_vals, label="CAL", color="C1")

    if frontier_R is not None and frontier_V is not None:
        plt.scatter(frontier_V, frontier_R, s=15, alpha=0.7, label="Efficient Frontier")

    plt.scatter([orp_vol], [orp_return], color="red", label="ORP", zorder=5)

    plt.xlabel("Volatility (σ)")
    plt.ylabel("Expected Return")
    plt.title("Capital Allocation Line & ORP")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fname, dpi=160)
    plt.close()


def plot_capm_scatter(
    asset_rets_m: pd.Series,
    market_rets_m: pd.Series,
    alpha: float,
    beta: float,
    fname: str,
):
    """Plot CAPM scatter with regression line."""
    ex_m = market_rets_m.values
    ex_i = asset_rets_m.values

    plt.figure()
    plt.scatter(ex_m, ex_i, alpha=0.5, label="Monthly excess returns")

    x_line = np.linspace(ex_m.min(), ex_m.max(), 100)
    y_line = alpha + beta * x_line
    plt.plot(x_line, y_line, color="red", label="CAPM line")

    plt.xlabel("Market Excess Return")
    plt.ylabel("Asset Excess Return")

    # Try to derive a simple asset label from the fname (capm_{TICKER}.png)
    asset_label = None
    try:
        base = os.path.basename(fname)
        if base.lower().startswith("capm_") and base.lower().endswith(".png"):
            asset_label = base[5:-4]  # strip 'capm_' and '.png'
    except Exception:
        asset_label = None

    if asset_label:
        plt.title(f"{asset_label} — CAPM Regression")
    else:
        plt.title("CAPM Regression")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fname, dpi=160)
    plt.close()


def plot_drawdown_curves(drawdowns: pd.DataFrame, fname: str) -> None:
    """Plot portfolio drawdown curves on a single chart.

    Parameters
    ----------
    drawdowns : DataFrame
        DataFrame of drawdown series (in decimals), columns per portfolio.
    fname : str
        Output PNG filename.
    """
    if drawdowns is None or drawdowns.empty:
        return

    plt.figure(figsize=(12, 6))
    for col in drawdowns.columns:
        plt.plot(drawdowns.index, drawdowns[col], label=col, alpha=0.9)

    plt.axhline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    plt.ylabel("Drawdown")
    plt.xlabel("Date")
    plt.title("Portfolio Drawdowns Over Time")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.close()


def plot_loss_histogram_with_var(
    returns: pd.Series,
    var_level: float,
    var_value: float,
    cvar_value: float,
    fname: str,
    title: str | None = None,
) -> None:
    """Plot a return histogram with VaR / CVaR marked on the left tail.

    Parameters
    ----------
    returns : Series
        One-period simple returns (e.g. daily) for a portfolio.
    var_level : float
        Confidence level, e.g. 0.95.
    var_value : float
        Historical VaR at that level (left-tail quantile of returns).
    cvar_value : float
        Historical CVaR (expected shortfall) at that level.
    fname : str
        Output PNG filename.
    title : str, optional
        Plot title; if None, a default is used.
    """
    r = pd.Series(returns).dropna().astype(float)
    if r.empty:
        return

    plt.figure(figsize=(10, 6))
    plt.hist(r, bins=50, alpha=0.75, edgecolor="black")
    plt.axvline(var_value, color="red", linestyle="--", linewidth=2, label=f"VaR {int(var_level*100)}%")
    plt.axvline(cvar_value, color="orange", linestyle="--", linewidth=2, label=f"CVaR {int(var_level*100)}%")

    plt.xlabel("Daily Return")
    plt.ylabel("Frequency")
    if title is None:
        title = f"Return Distribution with VaR/CVaR (alpha={var_level:.2f})"
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.close()
