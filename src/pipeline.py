"""
Analysis pipeline v2 — fully self-contained.

No imports from old codebase. Uses src.analytics.* for all computation.
Adds stress testing, bootstrap Monte Carlo, and risk contribution.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from src.config.models import PortfolioConfig
from src.data.fetcher import fetch_prices
from src.data import transforms as T
from src.analytics.optimization import (
    max_sharpe,
    efficient_frontier,
    sharpe_ratio,
    portfolio_return,
    portfolio_volatility,
)
from src.analytics.regression import (
    capm_regression,
    run_capm_all,
    RegressionResult,
)
from src.analytics.attribution import (
    simple_attribution_from_holdings,
    time_series_attribution,
)
from src.analytics.risk import (
    run_stress_tests,
    stress_results_to_df,
    marginal_risk_contribution,
    risk_contribution_pct,
    tail_metrics,
    rolling_correlation,
    StressResult,
)
from src.analytics.hrp import hrp_weights, hrp_linkage_matrix
from src.analytics.rebalance import (
    drift_from_target,
    rebalanced_backtest,
    compute_turnover,
    trade_recommendations,
)
from src.analytics.exposure import (
    get_sector_weights,
    get_factor_tilts,
)
from src.analytics.income import (
    compute_income_summary,
    portfolio_income_metrics,
    cumulative_income_series,
)
from src.analytics.simulation import (
    run_all_simulations,
    SimulationResult,
    simulation_summary_df,
)
from src.analytics.backtest import build_backtest, BacktestResult


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
        self.sharpe = np.nan

    def compute_sharpe(self, rf_annual: float) -> None:
        self.sharpe = T.sharpe_ratio(self.daily_returns, rf_annual=rf_annual)


@dataclass
class OptimizationResult:
    """Optimal portfolio from mean-variance optimization."""

    weights: pd.Series
    expected_return: float
    expected_vol: float
    sharpe: float
    frontier_returns: np.ndarray
    frontier_vols: np.ndarray


@dataclass
class AnalysisResults:
    """Complete results from a pipeline run."""

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

    # CAPM
    capm_results: list[RegressionResult] = field(default_factory=list)

    # Risk metrics
    correlation_matrix: Optional[pd.DataFrame] = None
    drawdown_metrics: Optional[pd.DataFrame] = None
    tail_risk: Optional[dict] = None
    risk_contribution: Optional[pd.Series] = None

    # Stress testing
    stress_results: list[StressResult] = field(default_factory=list)
    stress_df: Optional[pd.DataFrame] = None

    # Holdings
    holdings: Optional[pd.DataFrame] = None

    # Backtest engine output (inception handling, costs, trade log, coverage)
    backtest: Optional[BacktestResult] = None

    # Attribution
    asset_attribution: Optional[pd.DataFrame] = None
    sector_attribution: Optional[pd.DataFrame] = None
    # Per-period contribution of each holding to active return (time series)
    ts_attribution: Optional[pd.DataFrame] = None

    # HRP
    hrp: Optional[PortfolioSeries] = None
    hrp_weights: Optional[pd.Series] = None
    hrp_linkage: Optional[np.ndarray] = None

    # Rebalancing
    rebalanced: Optional[PortfolioSeries] = None
    weight_drift: Optional[pd.DataFrame] = None
    turnover_table: Optional[pd.DataFrame] = None
    # Buy/sell trades to rebalance current drifted weights to a target
    trade_recos: Optional[pd.DataFrame] = None          # -> your target weights
    trade_recos_orp: Optional[pd.DataFrame] = None      # -> the optimal (ORP) weights

    # Correlation regime
    correlation_regime: Optional[pd.DataFrame] = None

    # Sector & factor exposure
    sector_weights: Optional[pd.DataFrame] = None
    factor_tilts: Optional[pd.DataFrame] = None
    # Real Fama-French factor loadings: {model_name: DataFrame(Asset, *_coef, R2)}
    factor_models: dict = field(default_factory=dict)
    # Interactive what-if scenario model: {value, drivers:[{name,ticker,group,beta}], presets}
    scenario_model: Optional[dict] = None

    # Income
    income_summary: Optional[pd.DataFrame] = None
    income_metrics: Optional[dict] = None
    cumulative_income: Optional[pd.Series] = None

    # Performance measurement (benchmark-relative suite)
    performance_summary: Optional[pd.DataFrame] = None
    capture_metrics: Optional[dict] = None
    rolling_alpha_beta: Optional[pd.DataFrame] = None

    # Tax analysis
    tax_metrics: Optional[dict] = None
    tax_detail: Optional[pd.DataFrame] = None

    # Monte Carlo
    simulations: list[SimulationResult] = field(default_factory=list)
    simulation_summary: Optional[pd.DataFrame] = None

    # Retirement / withdrawal planning
    plan_result: Optional[object] = None

    # Interpretations (generated by report engine)
    interpretations: Optional[dict] = None


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
        # Effective benchmark column/label — becomes a synthetic blended series when
        # config.benchmark_weights is set (see _fetch_data).
        self._bench_label = config.benchmark

    def run(
        self,
        progress: Optional[ProgressCallback] = None,
    ) -> AnalysisResults:
        """Execute the full analysis pipeline."""
        steps: list[tuple[str, Callable]] = [
            ("Fetching price data", self._fetch_data),
            ("Building active portfolio", self._build_active),
            ("Computing rebalance analysis", self._compute_rebalance),
            ("Computing income analytics", self._compute_income),
            ("Building passive portfolio", self._build_passive),
            ("Optimizing (ORP & frontier)", self._optimize),
            ("Running CAPM regressions", self._run_capm),
            ("Measuring performance", self._measure_performance),
            ("Analyzing taxes", self._compute_tax),
            ("Computing risk metrics", self._compute_risk),
            ("Recommending trades", self._recommend_trades),
            ("Running stress tests", self._run_stress_tests),
            ("Building scenario model", self._build_scenario_model),
            ("Building complete portfolio", self._build_complete),
            ("Running attribution", self._run_attribution),
            ("Computing exposures", self._compute_exposure),
            ("Fama-French factor models", self._run_factor_models),
            ("Monte Carlo simulation", self._simulate),
            ("Retirement planning", self._run_plan),
            ("Generating reports", self._generate_reports),
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
                import traceback
                traceback.print_exc()

        if progress:
            progress("Complete", 1.0)

        return self.results

    # ── Step implementations ──

    def _fetch_data(self) -> None:
        blend = self.config.benchmark_weights or {}
        bench_tickers = list(blend) if blend else [self.config.benchmark]
        all_tickers = sorted(set(self.config.tickers + bench_tickers))

        prices = fetch_prices(
            tickers=all_tickers,
            start=self.config.start_date,
            end=self.config.end_date,
        )

        # Blended benchmark → synthesize a value series and inject it as a column so
        # every downstream benchmark reference (via self._bench_label) works unchanged.
        if blend:
            from src.analytics.performance import blended_benchmark

            series = blended_benchmark(prices, blend, self.config.capital)
            label = self.config.benchmark or "Blended Benchmark"
            if label in prices.columns:  # avoid clobbering a real ticker column
                label = f"{label} (blend)"
            prices[label] = series.reindex(prices.index)
            self._bench_label = label
        else:
            self._bench_label = self.config.benchmark

        self.results.prices = prices
        self.results.monthly_returns = T.monthly_returns(prices)

        asset_cols = [
            t for t in self.config.tickers
            if t in self.results.monthly_returns.columns
        ]
        if asset_cols:
            self.results.correlation_matrix = (
                self.results.monthly_returns[asset_cols].corr()
            )

    def _build_active(self) -> None:
        """Build the active portfolio via the backtest engine, which handles
        assets with different inception dates (rescale / cash) plus optional
        calendar rebalancing and transaction costs."""
        bt = self.config.backtest
        # capital is the total account value; invest the risky portion (capital -
        # cash) in the stocks, then fold the cash sleeve back in below.
        invested = self.config.capital - self.config.cash
        result = build_backtest(
            prices=self.results.prices,
            weights=self.config.weights,
            capital=invested,
            rf_annual=self.config.risk_free_rate,
            inception_mode=bt.inception_mode,
            rebalance_frequency=bt.rebalance_frequency,
            cost_bps=bt.transaction_cost_bps,
            start=self.config.start_date,
        )

        self.results.backtest = result
        self.results.holdings = result.initial_holdings

        port_value = result.values
        if port_value.empty:
            raise ValueError("Backtest produced no portfolio values.")
        port_value.name = "Active"

        # Fold a cash balance in as a risk-free sleeve so the active portfolio's
        # return/vol/drawdown reflect the honest cash drag (cash held alongside
        # the stocks, growing at the risk-free rate).
        active_value = T.blend_cash(
            port_value, self.config.cash, self.config.risk_free_rate
        )
        active_value.name = "Active"

        self.results.active = PortfolioSeries("Active", active_value)
        self.results.active.compute_sharpe(self.config.risk_free_rate)

    def _measure_performance(self) -> None:
        """Benchmark-relative performance suite (capture, batting, IR, rolling
        alpha/beta, dual-window summary)."""
        if self.results.active is None or self.results.passive is None:
            return
        from src.analytics import performance as perf

        av = self.results.active.values
        pv = self.results.passive.values
        eff = self.results.backtest.effective_start if self.results.backtest else None
        self.results.performance_summary = perf.performance_summary(
            av, pv, self.config.risk_free_rate, eff
        )
        self.results.capture_metrics = perf.capture_metrics(av, pv)
        self.results.rolling_alpha_beta = perf.rolling_alpha_beta(
            av, pv, self.config.risk_free_rate, window=12
        )

    def _compute_tax(self) -> None:
        """Unrealized gains/losses, harvest candidates, and estimated tax on
        realized gains from rebalancing (simple average-cost tier)."""
        if not self.config.tax.enabled or self.results.backtest is None:
            return
        from src.analytics.tax import build_tax_analysis

        metrics, detail = build_tax_analysis(
            self.results.backtest, self.results.prices,
            self.config.cost_basis, self.config.tax,
        )
        self.results.tax_metrics = metrics
        self.results.tax_detail = detail

    def _run_plan(self) -> None:
        """Retirement/withdrawal projection (only when enabled in config)."""
        plan = self.config.plan
        if not plan.enabled or self.results.active is None:
            return
        from src.analytics.simulation import run_retirement_plan

        current_value = float(self.results.active.values.iloc[-1])
        self.results.plan_result = run_retirement_plan(
            current_value=current_value,
            daily_returns=self.results.active.daily_returns,
            horizon_years=plan.horizon_years,
            annual_contribution=plan.annual_contribution,
            annual_withdrawal=plan.annual_withdrawal,
            inflation=plan.inflation,
            goal_amount=plan.goal_amount,
            expected_return=plan.expected_return,
        )

    def _build_passive(self) -> None:
        prices = self.results.prices
        benchmark = self._bench_label

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

    def _apply_black_litterman(self, mu_a, cov_a, asset_cols, rets_m):
        """Return BL posterior annual returns (pd.Series over asset_cols)."""
        from src.analytics.optimization import (
            BL_CONFIDENCE_SCALE,
            black_litterman_posterior,
        )

        idx = {t: i for i, t in enumerate(asset_cols)}
        n = len(asset_cols)

        # Prior "market" = the user's target weights (fallback: equal weight).
        w = np.array([float(self.config.weights.get(t, 0.0)) for t in asset_cols])
        if w.sum() <= 0:
            w = np.ones(n)
        w = w / w.sum()

        # Risk aversion implied by the benchmark; clamp to a sane range.
        delta = 2.5
        bench = self._bench_label
        if bench in rets_m.columns:
            br = rets_m[bench].dropna()
            if len(br) > 1:
                ann_ret = (1.0 + br.mean()) ** 12 - 1.0
                ann_var = float(br.std() ** 2) * 12.0
                if ann_var > 1e-9:
                    d = (ann_ret - self.config.risk_free_rate) / ann_var
                    if 0.5 <= d <= 15.0:
                        delta = float(d)

        views = []
        for v in self.config.black_litterman.views:
            c = BL_CONFIDENCE_SCALE.get(v.confidence, 1.0)
            if v.type == "absolute" and v.asset in idx:
                p = np.zeros(n)
                p[idx[v.asset]] = 1.0
                views.append((p, float(v.q), c))
            elif v.type == "relative" and v.asset_long in idx and v.asset_short in idx:
                p = np.zeros(n)
                p[idx[v.asset_long]] = 1.0
                p[idx[v.asset_short]] = -1.0
                views.append((p, float(v.q), c))
        if not views:
            return mu_a

        post = black_litterman_posterior(
            cov_a, w, views, self.config.risk_free_rate,
            tau=self.config.black_litterman.tau, delta=delta,
        )
        return pd.Series(post, index=asset_cols, name="mu_bl")

    def _optimize(self) -> None:
        if not self.config.include_orp:
            return

        rets_m = self.results.monthly_returns
        benchmark = self._bench_label

        asset_cols = [
            t for t in self.config.tickers
            if t in rets_m.columns and t != benchmark
        ]
        if len(asset_cols) < 2:
            return

        asset_rets = rets_m[asset_cols].dropna(how="all")

        if benchmark in rets_m.columns:
            aligned = asset_rets.join(rets_m[benchmark], how="inner")
            asset_rets = aligned[asset_cols]

        cov_m = asset_rets.cov()
        mu_a = (1.0 + asset_rets.mean()) ** 12 - 1.0
        cov_a = cov_m.values * 12.0

        # Black-Litterman: blend the user's views into the expected returns.
        bl = self.config.black_litterman
        if bl.enabled and bl.views:
            mu_a = self._apply_black_litterman(mu_a, cov_a, asset_cols, rets_m)

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

        # Risk contribution for ORP
        self.results.risk_contribution = risk_contribution_pct(weights, cov_a)

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

        # HRP weights
        try:
            hrp_w = hrp_weights(asset_rets)
            self.results.hrp_weights = hrp_w
            self.results.hrp_linkage = hrp_linkage_matrix(asset_rets)

            # Build realized HRP value series
            if self.results.active is not None:
                active_start = self.results.active.values.index[0]
                prices_hrp = self.results.prices[asset_cols].loc[active_start:].dropna(how="all")
                if not prices_hrp.empty:
                    nonzero_hrp = hrp_w[hrp_w.abs() > 1e-8]
                    available_hrp = [t for t in nonzero_hrp.index if t in prices_hrp.columns]
                    if available_hrp:
                        alloc_hrp = nonzero_hrp[available_hrp] * self.config.capital
                        initial_hrp = prices_hrp[available_hrp].iloc[0]
                        shares_hrp = alloc_hrp / initial_hrp
                        hrp_values = (prices_hrp[available_hrp] * shares_hrp).sum(axis=1)
                        hrp_values.name = "HRP"

                        self.results.hrp = PortfolioSeries("HRP", hrp_values)
                        self.results.hrp.compute_sharpe(self.config.risk_free_rate)
        except Exception as e:
            print(f"[pipeline] HRP failed: {e}")

    def _run_capm(self) -> None:
        rets_m = self.results.monthly_returns
        self.results.capm_results = run_capm_all(
            rets_m,
            self.config.tickers,
            self._bench_label,
            self.config.risk_free_rate,
        )

    def _compute_risk(self) -> None:
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

        if self.results.active:
            self.results.tail_risk = tail_metrics(self.results.active.daily_returns)

        # Correlation regime detection
        try:
            asset_cols = [
                t for t in self.config.tickers
                if t in self.results.prices.columns
            ]
            if len(asset_cols) >= 2:
                daily_rets = self.results.prices[asset_cols].pct_change().dropna()
                if len(daily_rets) > 63:
                    self.results.correlation_regime = rolling_correlation(
                        daily_rets, window=63
                    )
        except Exception as e:
            print(f"[pipeline] Correlation regime failed: {e}")

    def _run_stress_tests(self) -> None:
        if self.results.active is None or self.results.passive is None:
            return

        self.results.stress_results = run_stress_tests(
            self.results.active.values,
            self.results.passive.values,
        )
        self.results.stress_df = stress_results_to_df(self.results.stress_results)

    def _build_complete(self) -> None:
        if self.results.orp is None or not self.config.include_complete:
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

    def _run_attribution(self) -> None:
        if self.results.holdings is None or self.results.monthly_returns.empty:
            return

        try:
            self.results.asset_attribution = simple_attribution_from_holdings(
                self.results.holdings,
                self.results.monthly_returns,
                self._bench_label,
            )
        except Exception as e:
            print(f"[pipeline] Attribution failed: {e}")

        try:
            self.results.ts_attribution = time_series_attribution(
                self.results.monthly_returns, self.config.weights, self._bench_label,
            )
        except Exception as e:
            print(f"[pipeline] Time-series attribution failed: {e}")

    def _compute_rebalance(self) -> None:
        if self.results.holdings is None or self.results.prices.empty:
            return

        try:
            weights = self.config.weights
            tickers = [t for t in weights if t in self.results.prices.columns]
            if not tickers:
                return

            purchase_date = self.results.active.values.index[0] if self.results.active else self.config.start_date

            self.results.weight_drift = drift_from_target(
                self.results.prices, weights, purchase_date,
            )

            rebal_values = rebalanced_backtest(
                self.results.prices.loc[self.results.prices.index >= pd.Timestamp(purchase_date)],
                weights, self.config.capital,
                frequency="quarterly",
            )
            if not rebal_values.empty:
                self.results.rebalanced = PortfolioSeries("Rebalanced", rebal_values)
                self.results.rebalanced.compute_sharpe(self.config.risk_free_rate)

            if self.results.weight_drift is not None and not self.results.weight_drift.empty:
                self.results.turnover_table = compute_turnover(
                    self.results.weight_drift, frequency="quarterly",
                )
        except Exception as e:
            print(f"[pipeline] Rebalance analysis failed: {e}")

    def _compute_income(self) -> None:
        if self.results.holdings is None:
            return

        try:
            self.results.income_summary = compute_income_summary(
                self.results.holdings,
                self.config.start_date,
                self.config.end_date,
            )

            total_invested = float(self.results.holdings["Invested"].sum())
            if self.results.income_summary is not None:
                self.results.income_metrics = portfolio_income_metrics(
                    self.results.income_summary, total_invested,
                )

            self.results.cumulative_income = cumulative_income_series(
                self.results.holdings,
                self.config.start_date,
                self.config.end_date,
            )
        except Exception as e:
            print(f"[pipeline] Income analytics failed: {e}")

    def _compute_exposure(self) -> None:
        if self.results.holdings is None:
            return

        try:
            self.results.sector_weights = get_sector_weights(self.results.holdings)
        except Exception as e:
            print(f"[pipeline] Sector weights failed: {e}")

        try:
            rets_m = self.results.monthly_returns
            benchmark = self._bench_label
            asset_cols = [
                t for t in self.config.tickers
                if t in rets_m.columns and t != benchmark
            ]
            if asset_cols and benchmark in rets_m.columns:
                self.results.factor_tilts = get_factor_tilts(
                    rets_m[asset_cols],
                    rets_m[benchmark],
                )
        except Exception as e:
            print(f"[pipeline] Factor tilts failed: {e}")

    def _recommend_trades(self) -> None:
        """Concrete buy/sell trades to rebalance to the target (and to the ORP)."""
        bt = self.results.backtest
        if bt is None or self.results.active is None:
            return
        wh = bt.weight_history
        if wh is None or wh.empty:
            return
        current = wh.iloc[-1].drop(labels=["Cash"], errors="ignore")
        total_value = float(self.results.active.values.iloc[-1])
        latest = self.results.prices.ffill().iloc[-1]

        self.results.trade_recos = trade_recommendations(
            current, self.config.weights, total_value, latest
        )
        orp = self.results.orp_optimization
        if orp is not None and orp.weights is not None and not orp.weights.empty:
            self.results.trade_recos_orp = trade_recommendations(
                current, orp.weights.to_dict(), total_value, latest
            )

    def _build_scenario_model(self) -> None:
        """Precompute betas for the interactive what-if scenario builder (best-effort)."""
        if self.results.active is None:
            return
        try:
            from src.analytics.scenario import MACRO_FACTORS, build_scenario_model
        except Exception:
            return
        try:
            macro = fetch_prices(
                [t for _, t in MACRO_FACTORS],
                self.config.start_date, self.config.end_date,
            )
        except Exception:
            macro = pd.DataFrame()
        value = float(self.results.active.values.iloc[-1])
        self.results.scenario_model = build_scenario_model(
            self.results.active.daily_returns, macro, self.config.weights, value
        )

    def _run_factor_models(self) -> None:
        """Real Fama-French loadings (best-effort; needs the Ken French library)."""
        prices = self.results.prices
        asset_cols = [t for t in self.config.tickers if t in prices.columns]
        if not asset_cols:
            return
        try:
            from src.analytics.factor_models import FACTOR_SETS, run_factor_model
            from src.data.factors import fetch_ff_factors
        except Exception:
            return

        daily = prices[asset_cols].pct_change().dropna(how="all")
        port = self.results.active.daily_returns if self.results.active else None

        for model, cols in FACTOR_SETS.items():
            try:
                ff = fetch_ff_factors(self.config.start_date, self.config.end_date, model)
                if ff is None or ff.empty:
                    continue
                df = run_factor_model(daily, ff, cols, port_returns=port)
                if not df.empty:
                    self.results.factor_models[model] = df
            except Exception as e:
                print(f"[pipeline] Factor model '{model}' failed: {e}")

    def _simulate(self) -> None:
        if self.results.active is None:
            return

        current_val = float(self.results.active.values.iloc[-1])
        daily_rets = self.results.active.daily_returns

        self.results.simulations = run_all_simulations(
            current_value=current_val,
            daily_returns=daily_rets,
            horizon_days=252 * 3,
            n_paths=500,
            seed=42,
        )
        self.results.simulation_summary = simulation_summary_df(
            self.results.simulations
        )

    def _generate_reports(self) -> None:
        try:
            from src.reports.interpreter import generate_full_interpretation
            self.results.interpretations = generate_full_interpretation(self.results)
        except Exception as e:
            print(f"[pipeline] Report generation failed: {e}")

    def _save_outputs(self) -> None:
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

        if self.results.capture_metrics:
            summary["capture"] = self.results.capture_metrics
        if self.results.tax_metrics:
            summary["tax"] = self.results.tax_metrics
        plan = self.results.plan_result
        if plan is not None:
            summary["retirement_plan"] = {
                "horizon_years": plan.horizon_years,
                "success_probability": plan.success_prob,
                "depletion_probability": plan.depletion_prob,
                "median_terminal": plan.median_terminal,
                "safe_withdrawal_rate": plan.safe_withdrawal_rate,
                "goal_probability": plan.goal_prob if plan.goal_amount else None,
            }
        if self.results.backtest is not None:
            summary["transaction_costs_total"] = self.results.backtest.total_costs

        with open(self.output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

        # Holdings
        if self.results.holdings is not None:
            self.results.holdings.to_csv(
                self.output_dir / "holdings_table.csv", index=False
            )

        # Stress tests
        if self.results.stress_df is not None:
            self.results.stress_df.to_csv(
                self.output_dir / "stress_test_results.csv", index=False
            )

        # Clean prices
        if not self.results.prices.empty:
            self.results.prices.to_csv(self.output_dir / "clean_prices.csv")

        # Monthly returns
        if not self.results.monthly_returns.empty:
            self.results.monthly_returns.to_csv(
                self.output_dir / "monthly_returns.csv"
            )

        # CAPM results
        if self.results.capm_results:
            capm_rows = []
            for r in self.results.capm_results:
                capm_rows.append({
                    "Asset": r.ticker,
                    "Alpha": r.alpha,
                    "Beta": r.beta,
                    "t_alpha": r.t_alpha,
                    "t_beta": r.t_beta,
                    "R_squared": r.r_squared,
                })
            pd.DataFrame(capm_rows).to_csv(
                self.output_dir / "capm_results.csv", index=False
            )

        # Performance attribution
        if self.results.asset_attribution is not None:
            self.results.asset_attribution.to_csv(
                self.output_dir / "performance_attribution.csv", index=False
            )

        # Simulation summary
        if self.results.simulation_summary is not None:
            self.results.simulation_summary.to_csv(
                self.output_dir / "simulation_summary.csv", index=False
            )

        # Drawdown metrics
        if self.results.drawdown_metrics is not None:
            self.results.drawdown_metrics.to_csv(
                self.output_dir / "drawdown_metrics.csv", index=False
            )

        # Correlation matrix
        if self.results.correlation_matrix is not None:
            self.results.correlation_matrix.to_csv(
                self.output_dir / "correlation_matrix.csv"
            )

        # Income summary
        if self.results.income_summary is not None:
            self.results.income_summary.to_csv(
                self.output_dir / "income_summary.csv", index=False
            )

        # Factor tilts
        if self.results.factor_tilts is not None:
            self.results.factor_tilts.to_csv(
                self.output_dir / "factor_tilts.csv", index=False
            )

        # Sector weights
        if self.results.sector_weights is not None:
            self.results.sector_weights.to_csv(
                self.output_dir / "sector_weights.csv", index=False
            )

        # Turnover
        if self.results.turnover_table is not None:
            self.results.turnover_table.to_csv(
                self.output_dir / "turnover.csv", index=False
            )

        # ── Extended analytics exports ──

        # Daily value series for every portfolio variant.
        value_cols = {}
        for ps in (
            self.results.active, self.results.passive, self.results.orp,
            self.results.hrp, self.results.rebalanced, self.results.complete,
        ):
            if ps is not None:
                value_cols[ps.name] = ps.values
        if value_cols:
            pd.DataFrame(value_cols).to_csv(self.output_dir / "portfolio_values.csv")

        # Performance measurement suite.
        if self.results.performance_summary is not None:
            self.results.performance_summary.to_csv(
                self.output_dir / "performance_summary.csv", index=False
            )
        if self.results.capture_metrics:
            pd.DataFrame([self.results.capture_metrics]).to_csv(
                self.output_dir / "capture_metrics.csv", index=False
            )
        if (
            self.results.rolling_alpha_beta is not None
            and not self.results.rolling_alpha_beta.empty
        ):
            self.results.rolling_alpha_beta.to_csv(
                self.output_dir / "rolling_alpha_beta.csv"
            )

        # Sector attribution.
        if self.results.sector_attribution is not None:
            self.results.sector_attribution.to_csv(
                self.output_dir / "sector_attribution.csv", index=False
            )

        # HRP weights.
        if self.results.hrp_weights is not None:
            self.results.hrp_weights.rename("HRP_Weight").to_csv(
                self.output_dir / "hrp_weights.csv", header=True
            )

        # Tax detail.
        if self.results.tax_detail is not None and not self.results.tax_detail.empty:
            self.results.tax_detail.to_csv(
                self.output_dir / "tax_detail.csv", index=False
            )

        # Backtest trade log.
        bt = self.results.backtest
        if bt is not None and bt.trades is not None and not bt.trades.empty:
            bt.trades.to_csv(self.output_dir / "trade_log.csv", index=False)

        # Retirement plan outcomes.
        plan = self.results.plan_result
        if plan is not None:
            pd.DataFrame([{
                "HorizonYears": plan.horizon_years,
                "StartingValue": plan.starting_value,
                "SuccessProbability": plan.success_prob,
                "DepletionProbability": plan.depletion_prob,
                "MedianTerminal": plan.median_terminal,
                "P5": plan.percentiles.get("P5"),
                "P50": plan.percentiles.get("P50"),
                "P95": plan.percentiles.get("P95"),
                "SafeWithdrawalRate": plan.safe_withdrawal_rate,
                "GoalAmount": plan.goal_amount,
                "GoalProbability": plan.goal_prob if plan.goal_amount else None,
            }]).to_csv(self.output_dir / "retirement_plan.csv", index=False)

        self._write_manifest()

    # Human-readable descriptions for the export manifest (README).
    _FILE_DESCRIPTIONS = {
        "summary.json": "Headline metrics: returns, Sharpe, optimal weights, alpha/beta, tax, plan.",
        "holdings_table.csv": "Opening positions: ticker, target/realized weight, shares, price, invested.",
        "portfolio_values.csv": "Daily dollar value of each portfolio variant.",
        "clean_prices.csv": "Cleaned adjusted close prices used for all calculations.",
        "monthly_returns.csv": "Monthly returns per asset and benchmark.",
        "performance_summary.csv": "Return/risk stats over max-available and common windows.",
        "capture_metrics.csv": "Up/down capture, capture ratio, batting average vs benchmark.",
        "rolling_alpha_beta.csv": "Rolling alpha and beta of the portfolio vs benchmark.",
        "drawdown_metrics.csv": "Max drawdown and VaR/CVaR (95%, 99%) per portfolio.",
        "correlation_matrix.csv": "Asset-by-asset return correlation matrix.",
        "capm_results.csv": "Per-asset CAPM alpha, beta, t-stats, and R-squared.",
        "performance_attribution.csv": "Brinson-Fachler attribution by asset.",
        "sector_attribution.csv": "Brinson-Fachler attribution by sector.",
        "sector_weights.csv": "Portfolio weight by GICS sector.",
        "factor_tilts.csv": "Style/factor tilts of the portfolio.",
        "income_summary.csv": "Dividend income, yield, and yield-on-cost by holding.",
        "stress_test_results.csv": "Portfolio vs benchmark returns across historical stress scenarios.",
        "simulation_summary.csv": "Monte Carlo outcomes by method (expected value, percentiles).",
        "retirement_plan.csv": "Retirement/withdrawal plan outcomes and safe withdrawal rate.",
        "tax_detail.csv": "Per-lot unrealized gains/losses and harvest candidates.",
        "trade_log.csv": "Every simulated trade from the backtest (date, ticker, shares, price, value).",
        "turnover.csv": "Portfolio turnover at each rebalance.",
        "hrp_weights.csv": "Hierarchical Risk Parity target weights.",
    }

    def _write_manifest(self) -> None:
        """Write a README describing every exported file (institutional data pack)."""
        try:
            files = sorted(
                f.name for f in self.output_dir.iterdir()
                if f.is_file() and f.name != "README.txt"
            )
            cfg = self.config
            lines = [
                "PORTFOLIO ANALYZER - DATA EXPORT",
                "=" * 60,
                f"Universe:   {', '.join(cfg.tickers)}",
                f"Benchmark:  {cfg.benchmark}",
                f"Period:     {cfg.start_date} to {cfg.end_date}",
                f"Capital:    ${cfg.capital:,.0f}",
                "",
                "FILES",
                "-" * 60,
            ]
            for name in files:
                desc = self._FILE_DESCRIPTIONS.get(name, "")
                lines.append(f"{name:<30} {desc}")
            lines += [
                "",
                "DISCLOSURE",
                "-" * 60,
                "For informational and educational purposes only. Not investment,",
                "tax, or financial advice. Backtested and simulated results have",
                "inherent limitations and do not guarantee future performance.",
                "Data may contain errors from third-party sources.",
            ]
            (self.output_dir / "README.txt").write_text(
                "\n".join(lines), encoding="utf-8"
            )
        except Exception:
            pass
