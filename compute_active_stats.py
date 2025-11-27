import pandas as pd
import numpy as np
from math import sqrt

RF_ANNUAL = 0.04   # 4% risk-free rate

# Load active portfolio values
active = pd.read_csv("outputs/active_portfolio_value.csv", parse_dates=["Date"], index_col="Date")

# Rename the column (easier to refer to)
active = active.rename(columns={active.columns[0]: "Active"})

# Compute monthly returns
monthly = active["Active"].resample("M").last().pct_change().dropna()

# Annualized return
ret_m = monthly.mean()
ret_a = (1 + ret_m)**12 - 1

# Annualized volatility
vol_m = monthly.std()
vol_a = vol_m * np.sqrt(12)

# Sharpe ratio
excess_ret = ret_a - RF_ANNUAL
sharpe = excess_ret / vol_a if vol_a != 0 else np.nan

print("\n=== Active Portfolio Risk/Return Metrics ===")
print(f"Annualized Return: {ret_a:.4f} ({ret_a*100:.2f}%)")
print(f"Annualized Volatility: {vol_a:.4f} ({vol_a*100:.2f}%)")
print(f"Sharpe Ratio: {sharpe:.4f}")
