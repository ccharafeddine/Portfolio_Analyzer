"""
Test suite for Portfolio Analyzer v2.

Run with: pytest tests/ -v
"""

import json
import tempfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config.models import PortfolioConfig, BLView, BLConfig, CompletePortfolioConfig
from src.data.transforms import (
    annualize_return,
    annualize_vol,
    sharpe_ratio,
    max_drawdown,
    drawdown_series,
    var_cvar,
    gain_to_pain,
    daily_returns,
)
from src.analytics.optimization import (
    max_sharpe,
    efficient_frontier,
    portfolio_return,
    portfolio_volatility,
    sharpe_ratio as opt_sharpe,
    min_variance_portfolio,
)
from src.analytics.regression import capm_regression, multi_factor_regression
from src.analytics.risk import (
    marginal_risk_contribution,
    risk_contribution_pct,
    tail_metrics,
)
from src.analytics.simulation import (
    parametric_simulation,
    bootstrap_simulation,
)
from src.analytics.attribution import brinson_fachler


# ══════════════════════════════════════════════════════════════
# Config tests
# ══════════════════════════════════════════════════════════════


class TestPortfolioConfig:
    def test_valid_config_creates(self):
        cfg = PortfolioConfig(
            tickers=["AAPL", "MSFT"],
            weights={"AAPL": 0.6, "MSFT": 0.4},
            start_date="2020-01-01",
            end_date="2024-12-31",
        )
        assert cfg.tickers == ["AAPL", "MSFT"]
        assert cfg.allocation_bounds == (0.0, 1.0)

    def test_tickers_uppercased(self):
        cfg = PortfolioConfig(
            tickers=["aapl", " msft "],
            weights={"aapl": 0.6, "msft": 0.4},
            start_date="2020-01-01",
            end_date="2024-12-31",
        )
        assert cfg.tickers == ["AAPL", "MSFT"]
        assert "AAPL" in cfg.weights

    def test_weights_must_sum_to_one(self):
        with pytest.raises(ValueError, match="Weights sum to"):
            PortfolioConfig(
                tickers=["AAPL"],
                weights={"AAPL": 0.5},
                start_date="2020-01-01",
                end_date="2024-12-31",
            )

    def test_end_must_be_after_start(self):
        with pytest.raises(ValueError, match="end_date"):
            PortfolioConfig(
                tickers=["AAPL"],
                weights={"AAPL": 1.0},
                start_date="2024-12-31",
                end_date="2020-01-01",
            )

    def test_weights_reference_valid_tickers(self):
        with pytest.raises(ValueError, match="not in universe"):
            PortfolioConfig(
                tickers=["AAPL"],
                weights={"AAPL": 0.5, "GOOG": 0.5},
                start_date="2020-01-01",
                end_date="2024-12-31",
            )

    def test_save_and_load(self, tmp_path):
        cfg = PortfolioConfig(
            tickers=["AAPL", "MSFT"],
            weights={"AAPL": 0.6, "MSFT": 0.4},
            start_date="2020-01-01",
            end_date="2024-12-31",
            capital=500_000,
        )
        path = tmp_path / "config.json"
        cfg.save(path)

        loaded = PortfolioConfig.load(path)
        assert loaded.tickers == cfg.tickers
        assert loaded.capital == 500_000
        assert abs(loaded.weights["AAPL"] - 0.6) < 0.001

    def test_short_sales_bounds(self):
        cfg = PortfolioConfig(
            tickers=["AAPL", "MSFT"],
            weights={"AAPL": 0.6, "MSFT": 0.4},
            start_date="2020-01-01",
            end_date="2024-12-31",
            short_sales=True,
            max_weight_bound=1.5,
        )
        assert cfg.allocation_bounds == (-1.5, 1.5)

    def test_bl_validation(self):
        with pytest.raises(ValueError, match="not in ticker universe"):
            PortfolioConfig(
                tickers=["AAPL", "MSFT"],
                weights={"AAPL": 0.6, "MSFT": 0.4},
                start_date="2020-01-01",
                end_date="2024-12-31",
                black_litterman=BLConfig(
                    enabled=True,
                    views=[BLView(type="absolute", asset="GOOG", q=0.05)],
                ),
            )


# ══════════════════════════════════════════════════════════════
# Transform tests
# ══════════════════════════════════════════════════════════════


class TestTransforms:
    @pytest.fixture
    def sample_returns(self):
        np.random.seed(42)
        return pd.Series(np.random.normal(0.0005, 0.015, 252))

    def test_annualize_return(self, sample_returns):
        ar = annualize_return(sample_returns)
        assert isinstance(ar, float)
        assert not np.isnan(ar)
        # Should be roughly within reasonable bounds
        assert -1.0 < ar < 5.0

    def test_annualize_vol(self, sample_returns):
        av = annualize_vol(sample_returns)
        assert isinstance(av, float)
        assert av > 0

    def test_sharpe_ratio(self, sample_returns):
        s = sharpe_ratio(sample_returns, rf_annual=0.04, periods_per_year=252)
        assert isinstance(s, float)
        assert not np.isnan(s)

    def test_max_drawdown(self):
        values = pd.Series([100, 110, 90, 95, 80, 100])
        mdd = max_drawdown(values)
        # Max drawdown: from 110 to 80 = -27.27%
        assert mdd == pytest.approx(-0.2727, abs=0.01)

    def test_max_drawdown_no_drawdown(self):
        values = pd.Series([100, 110, 120, 130])
        mdd = max_drawdown(values)
        assert mdd == 0.0

    def test_var_cvar(self, sample_returns):
        var95, cvar95 = var_cvar(sample_returns, 0.95)
        assert var95 < 0  # VaR is a loss
        assert cvar95 <= var95  # CVaR is worse than VaR

    def test_gain_to_pain(self):
        rets = pd.Series([0.01, 0.02, -0.01, 0.015, -0.005])
        gtp = gain_to_pain(rets)
        assert gtp is not None
        assert gtp > 0

    def test_daily_returns(self):
        prices = pd.DataFrame({"A": [100, 110, 105, 115]})
        rets = daily_returns(prices)
        assert len(rets) == 3
        assert rets["A"].iloc[0] == pytest.approx(0.1)


# ══════════════════════════════════════════════════════════════
# Optimization tests
# ══════════════════════════════════════════════════════════════


class TestOptimization:
    @pytest.fixture
    def simple_assets(self):
        """Simple 3-asset universe with known properties."""
        mu = np.array([0.10, 0.15, 0.12])
        cov_monthly = np.array([
            [0.001, 0.0002, 0.0003],
            [0.0002, 0.002, 0.0004],
            [0.0003, 0.0004, 0.0015],
        ])
        return mu, cov_monthly

    def test_max_sharpe_weights_sum_to_one(self, simple_assets):
        mu, cov = simple_assets
        result = max_sharpe(mu, cov, rf=0.04)
        assert abs(result.x.sum() - 1.0) < 1e-6

    def test_max_sharpe_no_negative_weights(self, simple_assets):
        mu, cov = simple_assets
        result = max_sharpe(mu, cov, rf=0.04, short_sales=False)
        assert all(w >= -1e-8 for w in result.x)

    def test_efficient_frontier_shape(self, simple_assets):
        mu, cov = simple_assets
        W, R, V = efficient_frontier(mu, cov, n_points=20)
        assert W.shape == (20, 3)
        assert R.shape == (20,)
        assert V.shape == (20,)
        assert all(v > 0 for v in V)

    def test_portfolio_return_calculation(self, simple_assets):
        mu, _ = simple_assets
        w = np.array([0.5, 0.3, 0.2])
        pr = portfolio_return(w, mu)
        expected = 0.5 * 0.10 + 0.3 * 0.15 + 0.2 * 0.12
        assert pr == pytest.approx(expected)

    def test_min_variance(self, simple_assets):
        _, cov = simple_assets
        w = min_variance_portfolio(cov)
        assert abs(w.sum() - 1.0) < 1e-6
        assert all(w_i >= -1e-8 for w_i in w)


# ══════════════════════════════════════════════════════════════
# Regression tests
# ══════════════════════════════════════════════════════════════


class TestRegression:
    def test_capm_regression(self):
        np.random.seed(42)
        n = 60
        market = pd.Series(np.random.normal(0.01, 0.04, n))
        alpha_true = 0.002
        beta_true = 1.2
        asset = alpha_true + beta_true * market + np.random.normal(0, 0.02, n)
        asset = pd.Series(asset)

        result = capm_regression(asset, market, rf_periodic=0.003)
        assert result.beta == pytest.approx(beta_true, abs=0.3)
        assert result.r_squared > 0.5
        assert result.n_obs == n

    def test_capm_empty_returns(self):
        result = capm_regression(
            pd.Series(dtype=float),
            pd.Series(dtype=float),
        )
        assert np.isnan(result.alpha)


# ══════════════════════════════════════════════════════════════
# Risk analytics tests
# ══════════════════════════════════════════════════════════════


class TestRisk:
    def test_risk_contribution_sums_to_100(self):
        w = np.array([0.4, 0.3, 0.3])
        cov = np.array([
            [0.04, 0.01, 0.005],
            [0.01, 0.06, 0.01],
            [0.005, 0.01, 0.03],
        ])
        rc = risk_contribution_pct(w, cov)
        assert rc.sum() == pytest.approx(100.0, abs=0.1)

    def test_tail_metrics_returns_all_keys(self):
        np.random.seed(42)
        rets = pd.Series(np.random.normal(0.001, 0.02, 500))
        tm = tail_metrics(rets)
        expected_keys = ["VaR_95", "CVaR_95", "Skewness", "Sortino", "Calmar"]
        for k in expected_keys:
            assert k in tm


# ══════════════════════════════════════════════════════════════
# Simulation tests
# ══════════════════════════════════════════════════════════════


class TestSimulation:
    @pytest.fixture
    def sample_daily_returns(self):
        np.random.seed(42)
        return pd.Series(np.random.normal(0.0004, 0.012, 500))

    def test_parametric_shape(self, sample_daily_returns):
        result = parametric_simulation(
            current_value=1_000_000,
            daily_returns=sample_daily_returns,
            horizon_days=252,
            n_paths=100,
            seed=42,
        )
        assert result.paths.shape == (100, 252)
        assert len(result.terminal_values) == 100
        assert result.starting_value == 1_000_000

    def test_bootstrap_shape(self, sample_daily_returns):
        result = bootstrap_simulation(
            current_value=1_000_000,
            daily_returns=sample_daily_returns,
            horizon_days=252,
            n_paths=100,
            seed=42,
        )
        assert result.paths.shape == (100, 252)
        assert result.prob_loss >= 0.0
        assert result.prob_loss <= 1.0

    def test_percentiles_ordered(self, sample_daily_returns):
        result = parametric_simulation(
            current_value=1_000_000,
            daily_returns=sample_daily_returns,
            horizon_days=252,
            n_paths=500,
            seed=42,
        )
        p = result.percentiles
        assert p["P5"] <= p["P25"] <= p["P50"] <= p["P75"] <= p["P95"]


# ══════════════════════════════════════════════════════════════
# Attribution tests
# ══════════════════════════════════════════════════════════════


class TestAttribution:
    def test_brinson_fachler_basic(self):
        active_w = pd.Series({"A": 0.6, "B": 0.4})
        bench_w = pd.Series({"A": 0.5, "B": 0.5})
        active_r = pd.Series({"A": 0.10, "B": 0.05})
        bench_r = pd.Series({"A": 0.08, "B": 0.06})

        df = brinson_fachler(active_w, bench_w, active_r, bench_r)
        assert "Allocation" in df.columns
        assert "Selection" in df.columns
        assert "Interaction" in df.columns
        assert len(df) == 2

    def test_attribution_total_equals_excess(self):
        active_w = pd.Series({"A": 0.7, "B": 0.3})
        bench_w = pd.Series({"A": 0.5, "B": 0.5})
        r = pd.Series({"A": 0.10, "B": 0.05})

        df = brinson_fachler(active_w, bench_w, r, r)
        # With same returns, total attribution = difference in weighted returns
        port_ret = (active_w * r).sum()
        bench_ret = (bench_w * r).sum()
        assert df["Total"].sum() == pytest.approx(port_ret - bench_ret, abs=1e-10)
