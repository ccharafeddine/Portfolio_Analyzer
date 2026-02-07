"""
Analysis pipeline — structured orchestrator replacing main.py.

Key differences from the old approach:
- No subprocess.run — Streamlit calls this directly
- Typed results via dataclasses, not scattered CSV reads
- Each step is independently callable and testable
- Progress callback for real-time UI feedback
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from src.config.models import PortfolioConfig
from src.data.fetcher import fetch_prices
from src.data import transforms as T


# ──────────────────────────────────────────────────────────────
# Result containers
# ──────────────────────────────────────────────────────────────


@dataclass
class PortfolioSeries:
    """A named portfolio value time series with computed stats."""

    name: str
    values: pd.Series
    daily_returns: pd.Series = field(init=False)
    ann_return: float = field(init=False)
    ann_vol: float = field(init=False)
    sharpe: float = field(init=False)
    max_dd: float = field(init=False)

    def __post_init__(self):
        self.daily_returns = self.values.pct_change().dropna()
        self.ann_return = T.annualize_return(self.daily_returns)
        self.ann_vol = T.annualize_vol(self.daily_returns)
        self.max_dd = T.max_drawdown(self.values)
        # Sharpe computed with rf=0 here; pipeline overrides with actual rf
        self.sharpe = np.nan

    def compute_sharpe(self, rf_annual: float) -> None:
        self.sharpe = T.sharpe_ratio(self.daily_returns, rf_annual=rf_annual)


@dataclass
class OptimizationResult:
    """Optimal portfolio from mean-variance or BL optimization."""

    weights: pd.Series
    expected_return: float
    expected_vol: float
    sharpe: float
    frontier_returns: np.ndarray
    frontier_vols: np.ndarray


@dataclass
class CAPMResult:
    """CAPM regression result for one asset."""

    ticker: str
    alpha: float
    beta: float
    t_alpha: float
    t_beta: float
    r_squared: float


@dataclass
class AnalysisResults:
    """
    Complete results from a pipeline run.

    This is what the UI consumes — no file reads needed.
    """

    config: PortfolioConfig

    # Price data
    prices: pd.DataFrame = field(default_factory=pd.DataFrame)
    monthly_returns: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Portfolio series
    active: Optional[PortfolioSeries] = None
    passive: Optional[PortfolioSeries] = None
    orp: Optional[PortfolioSeries] = None
    complete: Optional[PortfolioSeries] = None

    # Optimization
    orp_optimization: Optional[OptimizationResult] = None
    bl_optimization: Optional[OptimizationResult] = None
    hrp_weights: Optional[pd.Series] = None

    # CAPM
    capm_results: list[CAPMResult] = field(default_factory=list)

    # Risk metrics
    correlation_matrix: Optional[pd.DataFrame] = None
    drawdown_metrics: Optional[pd.DataFrame] = None

    # Holdings
    holdings: Optional[pd.DataFrame] = None

    # Attribution
    asset_attribution: Optional[pd.DataFrame] = None
    sector_attribution: Optional[pd.DataFrame] = None

    # Factor regressions
    factor_results: dict[str, pd.DataFrame] = field(default_factory=dict)

    # Monte Carlo
    forecast_paths: dict[str, np.ndarray] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────────


ProgressCallback = Callable[[str, float], None]


class AnalysisPipeline:
    """
    Orchestrates the full portfolio analysis.

    Usage:
        config = PortfolioConfig.load("config.json")
        pipeline = AnalysisPipeline(config)
        results = pipeline.run(progress_callback=my_callback)
    """

    def __init__(
        self,
        config: PortfolioConfig,
        output_dir: str = "outputs",
    ):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = AnalysisResults(config=config)

    def run(
        self,
        progress: Optional[ProgressCallback] = None,
    ) -> AnalysisResults:
        """
        Execute the full analysis pipeline.

        Parameters
        ----------
        progress : callable, optional
            Called with (step_label: str, fraction: float) at each step.
            Fraction goes from 0.0 to 1.0.
        """
        steps: list[tuple[str, Callable]] = [
            ("Fetching price data", self._fetch_data),
            ("Building active portfolio", self._build_active),
            ("Building passive portfolio", self._build_passive),
            ("Optimizing (ORP & frontier)", self._optimize),
            ("Running CAPM regressions", self._run_capm),
            ("Computing risk metrics", self._compute_risk),
            ("Building complete portfolio", self._build_complete),
            ("Running attribution", self._run_attribution),
            ("Running factor models", self._run_factors),
            ("Monte Carlo simulation", self._simulate),
            ("Saving outputs", self._save_outputs),
        ]

        total = len(steps)
        for i, (label, step_fn) in enumerate(steps):
            if progress:
                progress(label, i / total)
            try:
                step_fn()
            except Exception as e:
                print(f"[pipeline] Step '{label}' failed: {e}")
                # Non-fatal: continue pipeline so we get partial results

        if progress:
            progress("Complete", 1.0)

        return self.results

    # ── Step implementations ──

    def _fetch_data(self) -> None:
        """Download prices for all tickers + benchmark."""
        all_tickers = sorted(
            set(self.config.tickers + [self.config.benchmark])
        )

        prices = fetch_prices(
            tickers=all_tickers,
            start=self.config.start_date,
            end=self.config.end_date,
        )

        self.results.prices = prices
        self.results.monthly_returns = T.monthly_returns(prices)

        # Correlation matrix of monthly returns (assets only, no benchmark)
        asset_cols = [t for t in self.config.tickers if t in self.results.monthly_returns.columns]
        if asset_cols:
            self.results.correlation_matrix = (
                self.results.monthly_returns[asset_cols].corr()
            )

        # Save for downstream modules that still read CSVs
        prices.to_csv(self.output_dir / "clean_prices.csv")
        self.results.monthly_returns.to_csv(self.output_dir / "monthly_returns.csv")

    def _build_active(self) -> None:
        """Build active portfolio from config weights."""
        prices = self.results.prices
        weights = self.config.weights
        capital = self.config.capital

        # Find tickers present in price data
        available = [t for t in weights if t in prices.columns]
        if not available:
            raise ValueError("None of the weighted tickers have price data.")

        # Find first common trading date
        start_ts = pd.Timestamp(self.config.start_date)
        subset = prices.loc[prices.index >= start_ts, available].dropna(how="any")
        if subset.empty:
            raise ValueError("No common trading date for all tickers after start_date.")

        purchase_date = subset.index[0]
        purchase_prices = subset.loc[purchase_date]

        # Compute shares
        holdings_rows = []
        shares = {}
        total_invested = 0.0

        for ticker in available:
            w = weights[ticker]
            allocation = capital * w
            n_shares = np.floor(allocation / purchase_prices[ticker])
            invested = n_shares * purchase_prices[ticker]

            holdings_rows.append({
                "Ticker": ticker,
                "TargetWeight": w,
                "PurchasePrice": float(purchase_prices[ticker]),
                "Shares": float(n_shares),
                "Invested": float(invested),
            })
            shares[ticker] = n_shares
            total_invested += invested

        holdings_df = pd.DataFrame(holdings_rows)
        holdings_df["RealizedWeight"] = holdings_df["Invested"] / total_invested
        self.results.holdings = holdings_df

        # Portfolio value series
        prices_after = prices.loc[prices.index >= purchase_date, available]
        port_value = sum(
            prices_after[t] * n for t, n in shares.items()
        )
        port_value = port_value.dropna()
        port_value.name = "Active"

        self.results.active = PortfolioSeries("Active", port_value)
        self.results.active.compute_sharpe(self.config.risk_free_rate)

        # Save CSVs for legacy compatibility
        holdings_df.to_csv(self.output_dir / "holdings_table.csv", index=False)
        port_value.to_frame("PortfolioValue").to_csv(
            self.output_dir / "active_portfolio_value.csv"
        )

    def _build_passive(self) -> None:
        """Build passive buy-and-hold benchmark portfolio."""
        prices = self.results.prices
        benchmark = self.config.benchmark

        if benchmark not in prices.columns:
            raise ValueError(f"Benchmark {benchmark} not in price data.")

        start_ts = pd.Timestamp(self.config.start_date)
        bench = prices[benchmark].loc[prices.index >= start_ts].dropna()
        if bench.empty:
            raise ValueError(f"No benchmark data after {self.config.start_date}")

        purchase_price = float(bench.iloc[0])
        units = self.config.capital / purchase_price
        passive_value = bench * units
        passive_value.name = "Passive"

        self.results.passive = PortfolioSeries("Passive", passive_value)
        self.results.passive.compute_sharpe(self.config.risk_free_rate)

        passive_value.to_frame("Passive").to_csv(
            self.output_dir / "passive_portfolio_value.csv"
        )

    def _optimize(self) -> None:
        """Run max-Sharpe optimization and trace the efficient frontier."""
        # Import from existing analytics (reuse the math, not the IO)
        from analytics import max_sharpe, efficient_frontier, sharpe_ratio

        rets_m = self.results.monthly_returns
        benchmark = self.config.benchmark

        # Asset returns only (exclude benchmark)
        asset_cols = [
            t for t in self.config.tickers
            if t in rets_m.columns and t != benchmark
        ]
        asset_rets = rets_m[asset_cols].dropna(how="all")
        bench_rets = rets_m[benchmark].dropna() if benchmark in rets_m.columns else None

        # Align
        if bench_rets is not None:
            aligned = asset_rets.join(bench_rets, how="inner")
            asset_rets = aligned[asset_cols]

        cov_m = asset_rets.cov()
        mu_a = (1.0 + asset_rets.mean()) ** 12 - 1.0  # annualized

        # Max-Sharpe
        bounds = self.config.allocation_bounds
        res = max_sharpe(
            mu_a.values, cov_m.values,
            rf=self.config.risk_free_rate,
            bounds=bounds,
            short_sales=self.config.short_sales,
        )
        weights = pd.Series(res.x, index=asset_cols, name="weight").round(6)

        cov_a = cov_m.values * 12.0
        port_mean = float(weights @ mu_a.values)
        port_vol = float(np.sqrt(weights.values @ cov_a @ weights.values))
        port_sharpe = sharpe_ratio(
            weights.values, mu_a.values, cov_m.values,
            self.config.risk_free_rate,
        )

        # Efficient frontier
        W, R, V = efficient_frontier(
            mu_a.values, cov_m.values,
            self.config.frontier_points,
            bounds, self.config.short_sales,
        )

        self.results.orp_optimization = OptimizationResult(
            weights=weights,
            expected_return=port_mean,
            expected_vol=port_vol,
            sharpe=port_sharpe,
            frontier_returns=R,
            frontier_vols=V,
        )

        # Build realized ORP value series
        if self.results.active is not None:
            active_start = self.results.active.values.index[0]
            prices_orp = self.results.prices[asset_cols].loc[active_start:].dropna(how="all")
            if not prices_orp.empty:
                nonzero = weights[weights.abs() > 1e-8]
                available_orp = [t for t in nonzero.index if t in prices_orp.columns]
                if available_orp:
                    alloc = nonzero[available_orp] * self.config.capital
                    initial = prices_orp[available_orp].iloc[0]
                    shares_orp = alloc / initial
                    orp_values = (prices_orp[available_orp] * shares_orp).sum(axis=1)
                    orp_values.name = "ORP"

                    self.results.orp = PortfolioSeries("ORP", orp_values)
                    self.results.orp.compute_sharpe(self.config.risk_free_rate)

                    orp_values.to_frame("ORP_Value").to_csv(
                        self.output_dir / "orp_value_realized.csv"
                    )

    def _run_capm(self) -> None:
        """CAPM regression for each asset."""
        from analytics import capm_regression

        rets_m = self.results.monthly_returns
        benchmark = self.config.benchmark
        rf_m = (1 + self.config.risk_free_rate) ** (1 / 12) - 1

        if benchmark not in rets_m.columns:
            return

        bench_rets = rets_m[benchmark].dropna()
        results = []

        for t in self.config.tickers:
            if t not in rets_m.columns or t == benchmark:
                continue
            df = pd.concat(
                [rets_m[t], bench_rets], axis=1, keys=[t, "mkt"]
            ).dropna()
            if df.empty:
                continue

            r = capm_regression(df[t], df["mkt"], rf_m)
            results.append(CAPMResult(
                ticker=t,
                alpha=r["alpha"],
                beta=r["beta"],
                t_alpha=r["t_alpha"],
                t_beta=r["t_beta"],
                r_squared=r["r2"],
            ))

        self.results.capm_results = results

        # Save CSV
        if results:
            rows = [
                {
                    "Asset": r.ticker, "alpha": r.alpha, "beta": r.beta,
                    "t_alpha": r.t_alpha, "t_beta": r.t_beta, "r2": r.r_squared,
                }
                for r in results
            ]
            pd.DataFrame(rows).to_csv(
                self.output_dir / "capm_results.csv", index=False
            )

    def _compute_risk(self) -> None:
        """Drawdown, VaR, CVaR for all portfolios."""
        portfolios = {
            "Active": self.results.active,
            "Passive": self.results.passive,
            "ORP": self.results.orp,
        }
        rows = []
        for name, ps in portfolios.items():
            if ps is None:
                continue
            var95, cvar95 = T.var_cvar(ps.daily_returns, 0.95)
            var99, cvar99 = T.var_cvar(ps.daily_returns, 0.99)
            rows.append({
                "Portfolio": name,
                "MaxDrawdown": ps.max_dd,
                "VaR_95": var95,
                "CVaR_95": cvar95,
                "VaR_99": var99,
                "CVaR_99": cvar99,
            })

        if rows:
            self.results.drawdown_metrics = pd.DataFrame(rows)
            self.results.drawdown_metrics.to_csv(
                self.output_dir / "drawdown_tail_metrics.csv", index=False
            )

    def _build_complete(self) -> None:
        """Build complete portfolio (ORP + risk-free mix)."""
        if self.results.orp is None:
            return

        y = self.config.complete_portfolio.y
        rf_daily = (1 + self.config.risk_free_rate) ** (1 / 252) - 1

        orp_vals = self.results.orp.values
        n_days = len(orp_vals)
        rf_path = self.config.capital * (1 + rf_daily) ** np.arange(n_days)
        rf_series = pd.Series(rf_path, index=orp_vals.index)

        complete_vals = y * orp_vals + (1 - y) * rf_series
        complete_vals.name = "Complete"

        self.results.complete = PortfolioSeries("Complete", complete_vals)
        self.results.complete.compute_sharpe(self.config.risk_free_rate)

        complete_vals.to_frame("Complete_Value").to_csv(
            self.output_dir / "complete_portfolio_value.csv"
        )

    def _run_attribution(self) -> None:
        """Brinson-Fachler attribution (delegate to existing module)."""
        try:
            from performance_attribution import run_performance_attribution
            run_performance_attribution(
                outdir=str(self.output_dir),
                config_path=str(self.output_dir.parent / "config.json"),
            )
            # Load results back
            attr_path = self.output_dir / "performance_attribution.csv"
            if attr_path.exists():
                self.results.asset_attribution = pd.read_csv(attr_path)
            sec_path = self.output_dir / "performance_attribution_sector.csv"
            if sec_path.exists():
                self.results.sector_attribution = pd.read_csv(sec_path)
        except Exception as e:
            print(f"[pipeline] Attribution failed: {e}")

    def _run_factors(self) -> None:
        """Multi-factor regressions (delegate to existing modules)."""
        try:
            from factor_loader import load_factors
            from multi_factor_regression import run_all_factor_models

            rets_m = self.results.monthly_returns
            benchmark = self.config.benchmark
            asset_cols = [
                t for t in self.config.tickers
                if t in rets_m.columns and t != benchmark
            ]
            asset_rets = rets_m[asset_cols].dropna(how="all")

            factors_dict = {}
            for model in ["ff3", "carhart4", "ff5", "quality_lowvol"]:
                try:
                    fdf = load_factors(
                        model,
                        start=self.config.start_str,
                        end=self.config.end_str,
                    )
                    if fdf is not None and not fdf.empty:
                        factors_dict[model] = fdf
                except Exception as e:
                    print(f"[pipeline] Factor load {model} failed: {e}")

            if factors_dict:
                run_all_factor_models(asset_rets, factors_dict, str(self.output_dir))
        except Exception as e:
            print(f"[pipeline] Factor regressions failed: {e}")

    def _simulate(self) -> None:
        """Monte Carlo forward simulation (delegate to existing module)."""
        try:
            from simulate_forecasts import run_martingale_forecasts
            run_martingale_forecasts(
                outdir=str(self.output_dir),
                horizon_days=252 * 3,
                n_paths=500,
            )
        except Exception as e:
            print(f"[pipeline] Simulation failed: {e}")

    def _save_outputs(self) -> None:
        """Save summary.json and any remaining outputs."""
        orp_opt = self.results.orp_optimization

        summary = {
            "risk_free_rate_annual": self.config.risk_free_rate,
            "portfolio_return": (
                self.results.active.ann_return if self.results.active else None
            ),
            "benchmark_return": (
                self.results.passive.ann_return if self.results.passive else None
            ),
            "portfolio_volatility": (
                self.results.active.ann_vol if self.results.active else None
            ),
            "portfolio_sharpe": (
                self.results.active.sharpe if self.results.active else None
            ),
            "y_cp": self.config.complete_portfolio.y,
        }

        if orp_opt:
            summary["max_sharpe_weights"] = orp_opt.weights.to_dict()
            summary["max_sharpe_portfolio"] = {
                "ann_expected_return": orp_opt.expected_return,
                "ann_volatility": orp_opt.expected_vol,
                "sharpe_ratio": orp_opt.sharpe,
            }

        if self.results.capm_results:
            summary["alpha"] = float(
                np.mean([r.alpha for r in self.results.capm_results])
            )
            summary["beta"] = float(
                np.mean([r.beta for r in self.results.capm_results])
            )

        with open(self.output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
