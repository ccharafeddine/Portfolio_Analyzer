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


# ──────────────────────────────────────────────────────────────
# Retirement / withdrawal planning (Phase 4)
# ──────────────────────────────────────────────────────────────


@dataclass
class PlanResult:
    """Cashflow-aware projection for retirement / goal planning."""

    horizon_years: int
    starting_value: float
    paths: np.ndarray  # (n_paths, horizon_days)
    horizon_days: int
    percentiles: dict[str, float]
    median_terminal: float
    success_prob: float      # P(not depleted), or P(>= goal) if a goal is set
    depletion_prob: float    # P(ran out of money before the horizon)
    goal_amount: float
    goal_prob: float         # P(terminal >= goal); 0 if no goal
    annual_contribution: float
    annual_withdrawal: float
    safe_withdrawal_rate: float | None = None


def _cashflow_paths(
    current_value: float,
    r: np.ndarray,
    horizon_days: int,
    n_paths: int,
    annual_contribution: float,
    annual_withdrawal: float,
    inflation: float,
    rng,
    block_size: int = 21,
) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap return paths with inflation-adjusted daily net cash flows.
    Returns (paths, depleted_mask)."""
    n = len(r)
    if n < block_size:
        block_size = max(1, n // 2)
    n_blocks = int(np.ceil(horizon_days / block_size))
    max_start = max(0, n - block_size)

    # Return matrix (one bootstrapped path per row).
    R = np.zeros((n_paths, horizon_days))
    for i in range(n_paths):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        sampled = np.concatenate([r[s : s + block_size] for s in starts])[:horizon_days]
        R[i, : len(sampled)] = sampled

    # Inflation-adjusted daily net contribution (negative = net withdrawal).
    year = np.arange(horizon_days) // 252
    infl = (1.0 + inflation) ** year
    net_daily = ((annual_contribution - annual_withdrawal) / 252.0) * infl

    paths = np.empty((n_paths, horizon_days))
    v = np.full(n_paths, float(current_value))
    depleted = np.zeros(n_paths, dtype=bool)
    for t in range(horizon_days):
        v = v * (1.0 + R[:, t]) + net_daily[t]
        np.maximum(v, 0.0, out=v)
        depleted |= v <= 0.0
        paths[:, t] = v
    return paths, depleted


def _recenter(r: np.ndarray, expected_annual: float | None) -> np.ndarray:
    """Shift historical daily returns to a target expected annual return, keeping
    their volatility and shape. Avoids extrapolating a short hot streak for
    decades. ``None`` leaves the historical mean untouched."""
    if expected_annual is None:
        return r
    daily_target = (1.0 + expected_annual) ** (1.0 / 252.0) - 1.0
    return r - r.mean() + daily_target


def run_retirement_plan(
    current_value: float,
    daily_returns: pd.Series,
    horizon_years: int = 30,
    annual_contribution: float = 0.0,
    annual_withdrawal: float = 0.0,
    inflation: float = 0.025,
    goal_amount: float = 0.0,
    expected_return: float | None = 0.07,
    n_paths: int = 1000,
    seed: int | None = 42,
    compute_swr: bool = True,
) -> PlanResult:
    """Project the portfolio forward with contributions/withdrawals and estimate
    the probability of success (not running out, or reaching a goal).

    ``expected_return`` recenters the bootstrapped returns to a long-run annual
    assumption (default 7%); pass ``None`` to use the raw historical mean.
    """
    r = _recenter(daily_returns.dropna().values, expected_return)
    horizon_days = int(horizon_years * 252)
    rng = np.random.default_rng(seed)

    paths, depleted = _cashflow_paths(
        current_value, r, horizon_days, n_paths,
        annual_contribution, annual_withdrawal, inflation, rng,
    )
    terminal = paths[:, -1]
    depletion_prob = float(depleted.mean())
    goal_prob = float((terminal >= goal_amount).mean()) if goal_amount > 0 else 0.0
    success_prob = goal_prob if goal_amount > 0 else (1.0 - depletion_prob)

    swr = None
    if compute_swr:
        swr = safe_withdrawal_rate(
            current_value, daily_returns, horizon_years, inflation,
            expected_return=expected_return, seed=seed,
        )

    return PlanResult(
        horizon_years=horizon_years,
        starting_value=current_value,
        paths=paths,
        horizon_days=horizon_days,
        percentiles=_compute_percentiles(terminal),
        median_terminal=float(np.percentile(terminal, 50)),
        success_prob=success_prob,
        depletion_prob=depletion_prob,
        goal_amount=goal_amount,
        goal_prob=goal_prob,
        annual_contribution=annual_contribution,
        annual_withdrawal=annual_withdrawal,
        safe_withdrawal_rate=swr,
    )


def safe_withdrawal_rate(
    current_value: float,
    daily_returns: pd.Series,
    horizon_years: int = 30,
    inflation: float = 0.025,
    target_success: float = 0.90,
    expected_return: float | None = 0.07,
    n_paths: int = 400,
    seed: int | None = 42,
) -> float:
    """Largest inflation-adjusted annual withdrawal (as a % of the starting value)
    that keeps the depletion probability at or below ``1 - target_success``."""
    r = _recenter(daily_returns.dropna().values, expected_return)
    horizon_days = int(horizon_years * 252)

    def success_at(rate: float) -> float:
        rng = np.random.default_rng(seed)
        _, depleted = _cashflow_paths(
            current_value, r, horizon_days, n_paths,
            0.0, rate * current_value, inflation, rng,
        )
        return 1.0 - float(depleted.mean())

    lo, hi = 0.0, 0.15
    for _ in range(16):  # binary search
        mid = (lo + hi) / 2.0
        if success_at(mid) >= target_success:
            lo = mid
        else:
            hi = mid
    return lo
