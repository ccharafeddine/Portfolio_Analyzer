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
    rolling_correlation,
    herfindahl_index,
    effective_n_bets,
    concentration_ratio,
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


# ======================================================================
# Concentration metrics tests
# ======================================================================


class TestConcentration:
    def test_herfindahl_equal_weights(self):
        w = np.array([0.25, 0.25, 0.25, 0.25])
        hhi = herfindahl_index(w)
        assert hhi == pytest.approx(0.25, abs=1e-6)

    def test_herfindahl_concentrated(self):
        w = np.array([1.0, 0.0, 0.0])
        hhi = herfindahl_index(w)
        assert hhi == pytest.approx(1.0, abs=1e-6)

    def test_effective_n_bets_equal(self):
        w = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        n = effective_n_bets(w)
        assert n == pytest.approx(5.0, abs=1e-6)

    def test_concentration_ratio_top3(self):
        w = np.array([0.5, 0.3, 0.1, 0.05, 0.05])
        cr = concentration_ratio(w, top_n=3)
        assert cr == pytest.approx(0.9, abs=1e-6)

    def test_concentration_with_series(self):
        w = pd.Series([0.5, 0.5], index=["A", "B"])
        hhi = herfindahl_index(w)
        assert hhi == pytest.approx(0.5, abs=1e-6)


# ======================================================================
# Correlation regime detection tests
# ======================================================================


class TestCorrelationRegime:
    def test_rolling_correlation_shape(self):
        np.random.seed(42)
        rets = pd.DataFrame(
            np.random.normal(0, 0.01, (200, 3)),
            columns=["A", "B", "C"],
        )
        result = rolling_correlation(rets, window=63)
        assert not result.empty
        assert "AvgCorrelation" in result.columns
        assert len(result) == 200 - 63

    def test_rolling_correlation_single_asset(self):
        np.random.seed(42)
        rets = pd.DataFrame(
            np.random.normal(0, 0.01, (100, 1)),
            columns=["A"],
        )
        result = rolling_correlation(rets, window=20)
        assert result.empty


# ======================================================================
# HRP tests
# ======================================================================


class TestHRP:
    @pytest.fixture
    def sample_returns(self):
        np.random.seed(42)
        return pd.DataFrame(
            np.random.normal(0, 0.01, (252, 4)),
            columns=["A", "B", "C", "D"],
        )

    def test_hrp_weights_sum_to_one(self, sample_returns):
        from src.analytics.hrp import hrp_weights
        w = hrp_weights(sample_returns)
        assert abs(w.sum() - 1.0) < 1e-6

    def test_hrp_weights_all_positive(self, sample_returns):
        from src.analytics.hrp import hrp_weights
        w = hrp_weights(sample_returns)
        assert all(w >= 0)

    def test_hrp_weights_correct_index(self, sample_returns):
        from src.analytics.hrp import hrp_weights
        w = hrp_weights(sample_returns)
        assert list(w.index) == ["A", "B", "C", "D"]

    def test_hrp_linkage_shape(self, sample_returns):
        from src.analytics.hrp import hrp_linkage_matrix
        link = hrp_linkage_matrix(sample_returns)
        # Linkage matrix has (n-1, 4) shape
        assert link.shape == (3, 4)

    def test_hrp_two_assets(self):
        from src.analytics.hrp import hrp_weights
        np.random.seed(42)
        rets = pd.DataFrame(
            np.random.normal(0, 0.01, (100, 2)),
            columns=["X", "Y"],
        )
        w = hrp_weights(rets)
        assert abs(w.sum() - 1.0) < 1e-6
        assert len(w) == 2


# ======================================================================
# Rebalancing tests
# ======================================================================


class TestRebalance:
    @pytest.fixture
    def simple_prices(self):
        dates = pd.date_range("2020-01-01", periods=252, freq="B")
        np.random.seed(42)
        prices = pd.DataFrame({
            "A": 100 * (1 + np.random.normal(0.001, 0.02, 252)).cumprod(),
            "B": 50 * (1 + np.random.normal(0.0005, 0.01, 252)).cumprod(),
        }, index=dates)
        return prices

    def test_drift_sums_to_one(self, simple_prices):
        from src.analytics.rebalance import drift_from_target
        drift = drift_from_target(simple_prices, {"A": 0.6, "B": 0.4}, simple_prices.index[0])
        row_sums = drift.sum(axis=1)
        assert all(abs(s - 1.0) < 1e-8 for s in row_sums)

    def test_drift_initial_matches_target(self, simple_prices):
        from src.analytics.rebalance import drift_from_target
        drift = drift_from_target(simple_prices, {"A": 0.6, "B": 0.4}, simple_prices.index[0])
        assert drift.iloc[0]["A"] == pytest.approx(0.6, abs=1e-6)
        assert drift.iloc[0]["B"] == pytest.approx(0.4, abs=1e-6)

    def test_rebalanced_backtest_length(self, simple_prices):
        from src.analytics.rebalance import rebalanced_backtest
        result = rebalanced_backtest(simple_prices, {"A": 0.6, "B": 0.4}, 100000)
        assert len(result) == len(simple_prices)

    def test_rebalanced_starts_at_capital(self, simple_prices):
        from src.analytics.rebalance import rebalanced_backtest
        result = rebalanced_backtest(simple_prices, {"A": 0.6, "B": 0.4}, 100000)
        assert result.iloc[0] == pytest.approx(100000, rel=0.01)

    def test_turnover_nonneg(self, simple_prices):
        from src.analytics.rebalance import drift_from_target, compute_turnover
        drift = drift_from_target(simple_prices, {"A": 0.6, "B": 0.4}, simple_prices.index[0])
        turnover = compute_turnover(drift, frequency="quarterly")
        if not turnover.empty:
            assert all(turnover["Turnover"] >= 0)


# ======================================================================
# Sector & Factor exposure tests
# ======================================================================


class TestExposure:
    def test_factor_tilts_columns(self):
        from src.analytics.exposure import get_factor_tilts
        np.random.seed(42)
        rets = pd.DataFrame(
            np.random.normal(0.005, 0.04, (60, 3)),
            columns=["A", "B", "C"],
        )
        mkt = pd.Series(np.random.normal(0.006, 0.04, 60))
        result = get_factor_tilts(rets, mkt)
        assert "Asset" in result.columns
        assert "Beta" in result.columns
        assert "Size" in result.columns
        assert "Momentum" in result.columns
        assert "Quality" in result.columns
        assert len(result) == 3

    def test_effective_n_sectors_equal(self):
        from src.analytics.exposure import effective_n_sectors
        df = pd.DataFrame({"Sector": ["Tech", "Health", "Finance"], "Weight": [1/3, 1/3, 1/3]})
        n = effective_n_sectors(df)
        assert n == pytest.approx(3.0, abs=0.01)

    def test_effective_n_sectors_concentrated(self):
        from src.analytics.exposure import effective_n_sectors
        df = pd.DataFrame({"Sector": ["Tech", "Health"], "Weight": [0.99, 0.01]})
        n = effective_n_sectors(df)
        assert n < 1.1  # Heavily concentrated

    def test_sector_weights_with_mock(self):
        from unittest.mock import patch, MagicMock
        from src.analytics.exposure import get_sector_weights
        holdings = pd.DataFrame({
            "Ticker": ["AAPL", "MSFT"],
            "RealizedWeight": [0.6, 0.4],
        })
        mock_info = {"sector": "Technology"}
        with patch("src.analytics.exposure.yf") as mock_yf:
            mock_ticker = MagicMock()
            mock_ticker.info = mock_info
            mock_yf.Ticker.return_value = mock_ticker
            result = get_sector_weights(holdings)
            assert not result.empty
            assert result["Weight"].sum() == pytest.approx(1.0, abs=0.01)


# ======================================================================
# Income tests
# ======================================================================


class TestIncome:
    def test_fetch_dividends_mock(self):
        from unittest.mock import patch, MagicMock
        from src.analytics.income import fetch_dividends
        mock_divs = pd.Series(
            [0.25, 0.25, 0.30],
            index=pd.to_datetime(["2021-03-15", "2021-06-15", "2021-09-15"]),
        )
        with patch("src.analytics.income.yf") as mock_yf:
            mock_ticker = MagicMock()
            mock_ticker.dividends = mock_divs
            mock_yf.Ticker.return_value = mock_ticker
            result = fetch_dividends("AAPL", "2021-01-01", "2021-12-31")
            assert len(result) == 3

    def test_income_summary_mock(self):
        from unittest.mock import patch, MagicMock
        from src.analytics.income import compute_income_summary
        holdings = pd.DataFrame({
            "Ticker": ["AAPL"],
            "Shares": [100.0],
            "PurchasePrice": [150.0],
        })
        mock_divs = pd.Series(
            [0.22, 0.23],
            index=pd.to_datetime(["2021-06-01", "2021-12-01"]),
        )
        with patch("src.analytics.income.yf") as mock_yf:
            mock_ticker = MagicMock()
            mock_ticker.dividends = mock_divs
            mock_ticker.info = {"regularMarketPrice": 170.0}
            mock_yf.Ticker.return_value = mock_ticker
            result = compute_income_summary(holdings, "2021-01-01", "2021-12-31")
            assert len(result) == 1
            assert result["AnnualIncome"].iloc[0] > 0

    def test_portfolio_income_metrics(self):
        from src.analytics.income import portfolio_income_metrics
        summary = pd.DataFrame({
            "Ticker": ["AAPL", "MSFT"],
            "AnnualIncome": [500.0, 300.0],
            "YieldOnCost": [0.02, 0.015],
        })
        metrics = portfolio_income_metrics(summary, 50000.0)
        assert metrics["total_annual_income"] == 800.0
        assert metrics["n_payers"] == 2
        assert metrics["portfolio_yield"] > 0

    def test_cumulative_income_mock(self):
        from unittest.mock import patch, MagicMock
        from src.analytics.income import cumulative_income_series
        holdings = pd.DataFrame({
            "Ticker": ["AAPL"],
            "Shares": [100.0],
            "PurchasePrice": [150.0],
        })
        mock_divs = pd.Series(
            [0.22, 0.23, 0.24],
            index=pd.to_datetime(["2021-03-01", "2021-06-01", "2021-09-01"]),
        )
        with patch("src.analytics.income.yf") as mock_yf:
            mock_ticker = MagicMock()
            mock_ticker.dividends = mock_divs
            mock_yf.Ticker.return_value = mock_ticker
            result = cumulative_income_series(holdings, "2021-01-01", "2021-12-31")
            assert not result.empty
            # Cumulative should be increasing
            assert result.iloc[-1] >= result.iloc[0]
