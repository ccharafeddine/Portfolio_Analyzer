
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

def plot_cal(sigma_rp: float, mu_rp: float, rf: float, fname: str):
    xs = np.linspace(0, max(0.05, sigma_rp * 1.5), 50)
    slope = (mu_rp - rf) / sigma_rp if sigma_rp > 0 else 0.0
    ys = rf + slope * xs
    plt.figure()
    plt.plot(xs, ys, label="CAL")
    plt.scatter([sigma_rp], [mu_rp], label="Optimal Risky Portfolio", zorder=5)
    plt.axhline(rf, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("Volatility")
    plt.ylabel("Expected Return")
    plt.title("Capital Allocation Line (CAL)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fname, dpi=160)
    plt.close()

def plot_capm_scatter(asset_ex: pd.Series, market_ex: pd.Series, alpha: float, beta: float, fname: str):
    plt.figure()
    plt.scatter(market_ex, asset_ex, s=18, alpha=0.7, label="Monthly Excess Returns")
    x = np.linspace(market_ex.min(), market_ex.max(), 100)
    y = alpha + beta * x
    plt.plot(x, y, label=f"CAPM Fit (alpha={alpha:.4f}, beta={beta:.2f})")
    plt.xlabel("Market Excess Return (Monthly)")
    plt.ylabel("Asset Excess Return (Monthly)")
    plt.title("CAPM Regression")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fname, dpi=160)
    plt.close()
