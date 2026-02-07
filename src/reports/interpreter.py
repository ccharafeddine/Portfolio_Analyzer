"""
Analysis interpretation engine for Portfolio Analyzer v2.

Generates plain-English financial commentary from AnalysisResults,
using actual numbers throughout. Each function returns a string
that can be embedded in HTML/PDF reports or shown inline in the UI.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.pipeline import AnalysisResults


def _pct(v: float, decimals: int = 2) -> str:
    """Format a float as a percentage string."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    return f"{v * 100:.{decimals}f}%"


def _dollar(v: float) -> str:
    """Format a float as a dollar string."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    if abs(v) >= 1e6:
        return f"${v / 1e6:,.2f}M"
    return f"${v:,.0f}"


def _num(v: float, decimals: int = 2) -> str:
    """Format a float to N decimal places."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    return f"{v:.{decimals}f}"


# ---------------------------------------------------------------
# Section interpreters
# ---------------------------------------------------------------


def interpret_performance(results: AnalysisResults) -> str:
    """Interpret portfolio performance vs benchmark."""
    parts = []

    if results.active is None or results.passive is None:
        return "Performance data not available for this run."

    act_ret = results.active.ann_return
    pas_ret = results.passive.ann_return
    excess = act_ret - pas_ret

    parts.append(
        f"The active portfolio returned {_pct(act_ret)} annualized, "
        f"while the benchmark ({results.config.benchmark}) returned {_pct(pas_ret)}."
    )

    if excess > 0:
        parts.append(
            f"The portfolio outperformed the benchmark by "
            f"{_pct(excess)} (excess return)."
        )
    elif excess < 0:
        parts.append(
            f"The portfolio underperformed the benchmark by "
            f"{_pct(abs(excess))}."
        )
    else:
        parts.append("The portfolio matched the benchmark return.")

    act_sharpe = results.active.sharpe
    pas_sharpe = results.passive.sharpe
    if not np.isnan(act_sharpe) and not np.isnan(pas_sharpe):
        parts.append(
            f"On a risk-adjusted basis, the active portfolio achieved a Sharpe ratio of "
            f"{_num(act_sharpe)}, compared to the benchmark's {_num(pas_sharpe)}."
        )

    act_vol = results.active.ann_vol
    pas_vol = results.passive.ann_vol
    parts.append(
        f"Annualized volatility stood at {_pct(act_vol)} for the portfolio "
        f"vs {_pct(pas_vol)} for the benchmark."
    )

    # ORP comparison
    if results.orp is not None:
        parts.append(
            f"The Optimal Risky Portfolio (ORP) returned {_pct(results.orp.ann_return)} "
            f"with a Sharpe ratio of {_num(results.orp.sharpe)}."
        )

    # HRP comparison
    if results.hrp is not None:
        parts.append(
            f"The Hierarchical Risk Parity (HRP) portfolio returned "
            f"{_pct(results.hrp.ann_return)} with volatility of {_pct(results.hrp.ann_vol)}."
        )

    return " ".join(parts)


def interpret_risk(results: AnalysisResults) -> str:
    """Interpret risk metrics and tail risk."""
    parts = []

    if results.active is None:
        return "Risk data not available."

    act_vol = results.active.ann_vol
    parts.append(
        f"The portfolio's annualized volatility was {_pct(act_vol)}."
    )

    if results.drawdown_metrics is not None and not results.drawdown_metrics.empty:
        row = results.drawdown_metrics[
            results.drawdown_metrics["Portfolio"] == "Active"
        ]
        if not row.empty:
            r = row.iloc[0]
            parts.append(
                f"Value at Risk (95% daily) was {_pct(r['VaR_95'])}, meaning on 95% of days "
                f"losses did not exceed this threshold. "
                f"Conditional VaR (expected shortfall) at 95% was {_pct(r['CVaR_95'])}, "
                f"representing the average loss on the worst 5% of days."
            )
            parts.append(
                f"At the 99% confidence level, VaR was {_pct(r['VaR_99'])} "
                f"and CVaR was {_pct(r['CVaR_99'])}."
            )

    if results.tail_risk:
        tr = results.tail_risk
        sortino = tr.get("Sortino", np.nan)
        calmar = tr.get("Calmar", np.nan)
        skew = tr.get("Skewness", np.nan)
        kurt = tr.get("Excess_Kurtosis", np.nan)

        if not np.isnan(sortino):
            parts.append(f"The Sortino ratio was {_num(sortino)}, measuring downside risk-adjusted return.")
        if not np.isnan(calmar):
            parts.append(f"The Calmar ratio was {_num(calmar)} (annualized return / max drawdown).")
        if not np.isnan(skew):
            direction = "negatively" if skew < 0 else "positively"
            parts.append(f"Return distribution was {direction} skewed ({_num(skew, 3)}).")
        if not np.isnan(kurt):
            tail_desc = "fat-tailed" if kurt > 0 else "thin-tailed"
            parts.append(f"Excess kurtosis of {_num(kurt, 3)} indicates a {tail_desc} distribution.")

    return " ".join(parts)


def interpret_drawdown(results: AnalysisResults) -> str:
    """Interpret drawdown characteristics."""
    parts = []

    if results.active is None:
        return "Drawdown data not available."

    max_dd = results.active.max_dd
    parts.append(
        f"The portfolio's maximum drawdown was {_pct(max_dd)}, "
        f"representing the largest peak-to-trough decline during the analysis period."
    )

    if results.passive is not None:
        bench_dd = results.passive.max_dd
        parts.append(
            f"By comparison, the benchmark experienced a max drawdown of {_pct(bench_dd)}."
        )
        if abs(max_dd) < abs(bench_dd):
            parts.append("The portfolio demonstrated better drawdown protection than the benchmark.")
        elif abs(max_dd) > abs(bench_dd):
            parts.append("The portfolio experienced deeper drawdowns than the benchmark.")

    if results.orp is not None:
        parts.append(
            f"The ORP's max drawdown was {_pct(results.orp.max_dd)}."
        )

    if results.tail_risk:
        worst = results.tail_risk.get("Worst_Day", np.nan)
        best = results.tail_risk.get("Best_Day", np.nan)
        if not np.isnan(worst) and not np.isnan(best):
            parts.append(
                f"The worst single day saw a {_pct(worst)} return, "
                f"while the best day returned {_pct(best)}."
            )

    return " ".join(parts)


def interpret_capm(results: AnalysisResults) -> str:
    """Interpret CAPM regression results."""
    if not results.capm_results:
        return "CAPM regression results not available."

    parts = []
    avg_alpha = float(np.mean([r.alpha for r in results.capm_results]))
    avg_beta = float(np.mean([r.beta for r in results.capm_results]))
    avg_r2 = float(np.mean([r.r_squared for r in results.capm_results]))

    parts.append(
        f"CAPM regressions were run for {len(results.capm_results)} assets "
        f"against {results.config.benchmark}."
    )
    parts.append(
        f"The average monthly alpha was {avg_alpha:.4f} "
        f"({avg_alpha * 12:.2%} annualized), with an average beta of {_num(avg_beta)}."
    )
    parts.append(
        f"Average R-squared was {_num(avg_r2, 3)}, indicating that "
        f"{avg_r2 * 100:.1f}% of return variance is explained by the market factor."
    )

    # Highlight significant alphas
    sig_alphas = [r for r in results.capm_results if abs(r.t_alpha) > 2.0]
    if sig_alphas:
        tickers = ", ".join(r.ticker for r in sig_alphas)
        parts.append(
            f"Statistically significant alpha (|t| > 2) was found for: {tickers}."
        )

    # High/low beta assets
    high_beta = [r for r in results.capm_results if r.beta > 1.2]
    low_beta = [r for r in results.capm_results if r.beta < 0.8]
    if high_beta:
        tickers = ", ".join(f"{r.ticker} ({_num(r.beta)})" for r in high_beta)
        parts.append(f"High-beta assets (>1.2): {tickers}.")
    if low_beta:
        tickers = ", ".join(f"{r.ticker} ({_num(r.beta)})" for r in low_beta)
        parts.append(f"Low-beta assets (<0.8): {tickers}.")

    return " ".join(parts)


def interpret_optimization(results: AnalysisResults) -> str:
    """Interpret optimization and portfolio construction results."""
    parts = []

    orp = results.orp_optimization
    if orp is None:
        return "Optimization results not available."

    parts.append(
        f"Mean-variance optimization identified an Optimal Risky Portfolio (ORP) "
        f"with an expected return of {_pct(orp.expected_return)} "
        f"and expected volatility of {_pct(orp.expected_vol)}, "
        f"yielding a Sharpe ratio of {_num(orp.sharpe, 3)}."
    )

    # Top holdings
    w_sorted = orp.weights.abs().sort_values(ascending=False)
    top3 = w_sorted.head(3)
    top_str = ", ".join(
        f"{t} ({_pct(float(orp.weights[t]))})" for t in top3.index
    )
    parts.append(f"The largest ORP positions are: {top_str}.")

    # Zero/near-zero positions
    near_zero = orp.weights[orp.weights.abs() < 0.01]
    if len(near_zero) > 0:
        parts.append(
            f"{len(near_zero)} asset(s) received near-zero allocation in the ORP."
        )

    # HRP comparison
    if results.hrp_weights is not None:
        hrp_top = results.hrp_weights.sort_values(ascending=False).head(3)
        hrp_str = ", ".join(
            f"{t} ({_pct(float(v))})" for t, v in hrp_top.items()
        )
        parts.append(f"HRP's largest positions are: {hrp_str}.")
        parts.append(
            "HRP uses hierarchical clustering to diversify across correlation-based clusters, "
            "often producing more balanced allocations than mean-variance optimization."
        )

    # Risk contribution
    if results.risk_contribution is not None:
        top_risk = results.risk_contribution.sort_values(ascending=False).head(2)
        risk_str = ", ".join(
            f"{t} ({_num(float(v), 1)}%)" for t, v in top_risk.items()
        )
        parts.append(f"The largest risk contributors in the ORP are: {risk_str}.")

    return " ".join(parts)


def interpret_stress_tests(results: AnalysisResults) -> str:
    """Interpret stress test results."""
    if results.stress_df is None or results.stress_df.empty:
        return "Stress test results not available."

    parts = []
    df = results.stress_df.copy()

    parts.append(
        f"The portfolio was stress-tested against {len(df)} historical scenarios."
    )

    # Parse percentages
    valid = df[df["Portfolio"] != "N/A"].copy()
    if valid.empty:
        return "No stress test scenarios had sufficient data for analysis."

    valid["Port_f"] = valid["Portfolio"].str.rstrip("%").astype(float) / 100
    valid["Bench_f"] = valid["Benchmark"].str.rstrip("%").astype(float) / 100

    worst = valid.loc[valid["Port_f"].idxmin()]
    parts.append(
        f"The worst scenario was '{worst['Scenario']}', where the portfolio "
        f"would have returned {_pct(worst['Port_f'])} vs the benchmark's "
        f"{_pct(worst['Bench_f'])}."
    )

    # Count outperformance
    outperform = (valid["Port_f"] > valid["Bench_f"]).sum()
    parts.append(
        f"The portfolio outperformed the benchmark in {outperform} of "
        f"{len(valid)} stress scenarios."
    )

    # Average impact
    avg_port = valid["Port_f"].mean()
    avg_bench = valid["Bench_f"].mean()
    parts.append(
        f"Average scenario impact: portfolio {_pct(avg_port)}, "
        f"benchmark {_pct(avg_bench)}."
    )

    return " ".join(parts)


def interpret_income(results: AnalysisResults) -> str:
    """Interpret dividend income analytics."""
    if results.income_metrics is None:
        return "Income analytics not available."

    im = results.income_metrics
    parts = []

    total = im.get("total_annual_income", 0)
    yld = im.get("portfolio_yield", 0)
    n_payers = im.get("n_payers", 0)

    parts.append(
        f"The portfolio generates an estimated ${total:,.2f} in annual dividend income, "
        f"representing a portfolio yield of {_pct(yld)}."
    )
    parts.append(
        f"{n_payers} of the holdings pay dividends."
    )

    avg_yoc = im.get("avg_yield_on_cost", 0)
    if avg_yoc > 0:
        parts.append(f"The average yield on cost is {_pct(avg_yoc)}.")

    if results.income_summary is not None and not results.income_summary.empty:
        top_payer = results.income_summary.sort_values(
            "AnnualIncome", ascending=False
        ).iloc[0]
        if top_payer["AnnualIncome"] > 0:
            parts.append(
                f"The largest income contributor is {top_payer['Ticker']} "
                f"at ${top_payer['AnnualIncome']:,.2f}/year."
            )

    return " ".join(parts)


def interpret_simulation(results: AnalysisResults) -> str:
    """Interpret Monte Carlo simulation results."""
    if not results.simulations:
        return "Monte Carlo simulation results not available."

    parts = []

    for sim in results.simulations:
        parts.append(
            f"{sim.name}: Expected portfolio value after "
            f"{sim.horizon_days // 252:.0f} years is "
            f"{_dollar(sim.expected_value)} (starting from "
            f"{_dollar(sim.starting_value)}). "
            f"The probability of loss is {_pct(sim.prob_loss)}. "
            f"The median outcome (P50) is {_dollar(sim.percentiles['P50'])}, "
            f"with a worst-case (P5) of {_dollar(sim.percentiles['P5'])} "
            f"and best-case (P95) of {_dollar(sim.percentiles['P95'])}."
        )

    if len(results.simulations) >= 2:
        s1, s2 = results.simulations[0], results.simulations[1]
        diff = s1.expected_value - s2.expected_value
        if abs(diff) > 100:
            higher = s1.name if diff > 0 else s2.name
            parts.append(
                f"The {higher} method projects a higher expected value "
                f"by {_dollar(abs(diff))}."
            )

    return " ".join(parts)


def interpret_correlation(results: AnalysisResults) -> str:
    """Interpret correlation and diversification."""
    parts = []

    if results.correlation_matrix is not None and not results.correlation_matrix.empty:
        corr = results.correlation_matrix
        n = len(corr)
        # Average off-diagonal correlation
        mask = ~np.eye(n, dtype=bool)
        avg_corr = float(corr.values[mask].mean())
        max_corr = float(corr.values[mask].max())
        min_corr = float(corr.values[mask].min())

        parts.append(
            f"The average pairwise correlation across {n} assets is {_num(avg_corr, 3)}."
        )
        parts.append(
            f"Correlations range from {_num(min_corr, 3)} to {_num(max_corr, 3)}."
        )

        if avg_corr < 0.3:
            parts.append("Low average correlation suggests good diversification benefits.")
        elif avg_corr < 0.6:
            parts.append("Moderate correlation indicates reasonable diversification.")
        else:
            parts.append(
                "High average correlation suggests limited diversification. "
                "Consider adding uncorrelated asset classes."
            )

    if results.correlation_regime is not None and not results.correlation_regime.empty:
        avg_series = results.correlation_regime["AvgCorrelation"]
        mean_val = float(avg_series.mean())
        std_val = float(avg_series.std())
        current = float(avg_series.iloc[-1])
        threshold = mean_val + std_val

        parts.append(
            f"Rolling 63-day average correlation is currently {_num(current, 3)} "
            f"(historical mean: {_num(mean_val, 3)}, std: {_num(std_val, 3)})."
        )

        if current > threshold:
            parts.append(
                "Current correlation is elevated (above mean + 1 std), "
                "indicating a high-correlation regime where diversification benefits are reduced."
            )
        else:
            parts.append("Correlation is within normal range.")

    if not parts:
        return "Correlation data not available."

    return " ".join(parts)


def generate_executive_summary(results: AnalysisResults) -> str:
    """Generate a concise executive summary covering key findings."""
    parts = []

    parts.append(
        f"This report analyzes a portfolio of {len(results.config.tickers)} assets "
        f"({', '.join(results.config.tickers)}) "
        f"benchmarked against {results.config.benchmark}, "
        f"with ${results.config.capital:,.0f} initial capital "
        f"from {results.config.start_date} to {results.config.end_date}."
    )

    if results.active and results.passive:
        act_ret = results.active.ann_return
        pas_ret = results.passive.ann_return
        excess = act_ret - pas_ret
        direction = "outperformed" if excess > 0 else "underperformed"
        parts.append(
            f"The portfolio returned {_pct(act_ret)} annualized and "
            f"{direction} the benchmark by {_pct(abs(excess))} "
            f"({_pct(excess)} excess return)."
        )
        parts.append(
            f"Risk-adjusted performance: Sharpe {_num(results.active.sharpe)}, "
            f"max drawdown {_pct(results.active.max_dd)}, "
            f"volatility {_pct(results.active.ann_vol)}."
        )

    if results.orp_optimization:
        parts.append(
            f"The optimized portfolio (ORP) targets {_pct(results.orp_optimization.expected_return)} return "
            f"at {_pct(results.orp_optimization.expected_vol)} volatility "
            f"(Sharpe {_num(results.orp_optimization.sharpe, 3)})."
        )

    if results.income_metrics:
        im = results.income_metrics
        if im.get("total_annual_income", 0) > 0:
            parts.append(
                f"Estimated annual dividend income: ${im['total_annual_income']:,.2f} "
                f"({_pct(im['portfolio_yield'])} yield)."
            )

    if results.simulations:
        sim = results.simulations[0]
        parts.append(
            f"Monte Carlo simulation ({sim.name}) projects a median value of "
            f"{_dollar(sim.percentiles['P50'])} over {sim.horizon_days // 252} years, "
            f"with {_pct(sim.prob_loss)} probability of loss."
        )

    return " ".join(parts)


def generate_full_interpretation(results: AnalysisResults) -> dict[str, str]:
    """
    Generate all interpretation sections.

    Returns a dict mapping section name to interpretation text.
    """
    return {
        "executive_summary": generate_executive_summary(results),
        "performance": interpret_performance(results),
        "risk": interpret_risk(results),
        "drawdown": interpret_drawdown(results),
        "capm": interpret_capm(results),
        "optimization": interpret_optimization(results),
        "stress_tests": interpret_stress_tests(results),
        "income": interpret_income(results),
        "simulation": interpret_simulation(results),
        "correlation": interpret_correlation(results),
    }
