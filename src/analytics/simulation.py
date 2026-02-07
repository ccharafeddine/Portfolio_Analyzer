"""
Monte Carlo simulation for portfolio forecasting.

Two methods:
1. Parametric (GBM): assume normal returns with estimated μ and σ
2. Bootstrap: resample from historical daily returns (preserves fat tails)

Self-contained: replaces old simulate_forecasts.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data import transforms as T


@dataclass
class SimulationResult:
    """Result of a Monte Carlo simulation run."""

    name: str
    method: str
    paths: np.ndarray  # shape (n_paths, horizon_days)
    terminal_values: np.ndarray  # shape (n_paths,)
    percentiles: dict[str, float]  # P5, P10, P25, P50, P75, P90, P95
    expected_value: float
    prob_loss: float  # probability of ending below starting value
    starting_value: float
    horizon_days: int


def parametric_simulation(
    current_value: float,
    daily_returns: pd.Series,
    horizon_days: int = 252 * 3,
    n_paths: int = 1000,
    seed: int | None = 42,
) -> SimulationResult:
    """
    GBM-based Monte Carlo: assume daily returns ~ N(μ, σ).

    Parameters
    ----------
    current_value : starting portfolio value
    daily_returns : historical daily returns to estimate μ and σ
    horizon_days : simulation horizon in trading days
    n_paths : number of simulation paths
    seed : random seed (None for non-deterministic)
    """
    rng = np.random.default_rng(seed)
    r = daily_returns.dropna().values

    mu = float(r.mean())
    sigma = float(r.std())

    # Generate random daily returns
    random_returns = rng.normal(mu, sigma, size=(n_paths, horizon_days))

    # Cumulative growth paths
    growth = np.cumprod(1 + random_returns, axis=1)
    paths = current_value * growth

    terminal = paths[:, -1]

    return SimulationResult(
        name="Parametric (GBM)",
        method="parametric",
        paths=paths,
        terminal_values=terminal,
        percentiles=_compute_percentiles(terminal),
        expected_value=float(terminal.mean()),
        prob_loss=float(np.mean(terminal < current_value)),
        starting_value=current_value,
        horizon_days=horizon_days,
    )


def bootstrap_simulation(
    current_value: float,
    daily_returns: pd.Series,
    horizon_days: int = 252 * 3,
    n_paths: int = 1000,
    block_size: int = 21,
    seed: int | None = 42,
) -> SimulationResult:
    """
    Block bootstrap Monte Carlo: resample blocks of historical returns.

    Preserves autocorrelation and fat tails better than parametric.

    Parameters
    ----------
    current_value : starting portfolio value
    daily_returns : historical daily returns to resample from
    horizon_days : simulation horizon in trading days
    n_paths : number of simulation paths
    block_size : size of contiguous blocks to resample (default 21 ~ 1 month)
    seed : random seed
    """
    rng = np.random.default_rng(seed)
    r = daily_returns.dropna().values
    n = len(r)

    if n < block_size:
        block_size = max(1, n // 2)

    n_blocks = int(np.ceil(horizon_days / block_size))
    max_start = n - block_size

    paths = np.zeros((n_paths, horizon_days))

    for i in range(n_paths):
        # Pick random block start indices
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        # Stitch blocks together
        sampled = np.concatenate([r[s : s + block_size] for s in starts])
        sampled = sampled[:horizon_days]
        paths[i] = current_value * np.cumprod(1 + sampled)

    terminal = paths[:, -1]

    return SimulationResult(
        name="Bootstrap",
        method="bootstrap",
        paths=paths,
        terminal_values=terminal,
        percentiles=_compute_percentiles(terminal),
        expected_value=float(terminal.mean()),
        prob_loss=float(np.mean(terminal < current_value)),
        starting_value=current_value,
        horizon_days=horizon_days,
    )


def _compute_percentiles(terminal: np.ndarray) -> dict[str, float]:
    """Compute standard percentiles from terminal values."""
    pcts = [5, 10, 25, 50, 75, 90, 95]
    values = np.percentile(terminal, pcts)
    return {f"P{p}": float(v) for p, v in zip(pcts, values)}


def simulation_summary_df(results: list[SimulationResult]) -> pd.DataFrame:
    """Create a comparison DataFrame from multiple simulation results."""
    rows = []
    for r in results:
        row = {
            "Method": r.name,
            "Expected Value": f"${r.expected_value:,.0f}",
            "Prob(Loss)": f"{r.prob_loss:.1%}",
        }
        for k, v in r.percentiles.items():
            row[k] = f"${v:,.0f}"
        rows.append(row)
    return pd.DataFrame(rows)


def run_all_simulations(
    current_value: float,
    daily_returns: pd.Series,
    horizon_days: int = 252 * 3,
    n_paths: int = 1000,
    seed: int | None = 42,
) -> list[SimulationResult]:
    """Run both parametric and bootstrap simulations."""
    return [
        parametric_simulation(
            current_value, daily_returns, horizon_days, n_paths, seed
        ),
        bootstrap_simulation(
            current_value, daily_returns, horizon_days, n_paths,
            block_size=21, seed=seed,
        ),
    ]
