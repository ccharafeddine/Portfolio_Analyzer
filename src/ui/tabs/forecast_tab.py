"""Forecast tab — Monte Carlo fan charts, simulation comparison, probability
analysis, and historical risk statistics.

Transcribed from app.py's ``with tab_forecast:`` block (lines 933-979).
"""

from __future__ import annotations

import numpy as np

from src.charts import plotly_charts as charts
from src.data import transforms as T

from .web_tab import WebTab


class ForecastTab(WebTab):
    def _populate(self, results) -> None:
        interp = results.interpretations or {}
        self.add_interpretation(interp.get("simulation"))

        # ── Retirement / withdrawal plan (when enabled) ──
        plan = results.plan_result
        if plan is not None:
            self.add_heading("Retirement Plan", explain="retirement_plan")
            self.add_chart(
                charts.simulation_fan_chart(
                    starting_value=plan.starting_value,
                    paths=plan.paths,
                    horizon_days=plan.horizon_days,
                    method_name=f"{plan.horizon_years}-Year Projection",
                ),
                height=420,
            )
            items = [
                ("Success Probability", f"{plan.success_prob:.0%}"),
                ("Depletion Risk", f"{plan.depletion_prob:.0%}"),
                ("Median Ending", f"${plan.median_terminal:,.0f}"),
                ("Worst Case (P5)", f"${plan.percentiles['P5']:,.0f}"),
            ]
            if plan.safe_withdrawal_rate is not None:
                items.append(("Safe Withdrawal Rate", f"{plan.safe_withdrawal_rate:.1%}"))
            if plan.goal_amount > 0:
                items.append(("Goal Probability", f"{plan.goal_prob:.0%}"))
            self.add_heading("Projected Outcomes", explain="plan_metrics")
            self.add_stat_grid(items, columns=4)

        if results.simulations:
            self.add_heading("Monte Carlo Forecast", explain="monte_carlo")
            for sim in results.simulations:
                self.add_chart(
                    charts.simulation_fan_chart(
                        starting_value=sim.starting_value,
                        paths=sim.paths,
                        horizon_days=sim.horizon_days,
                        method_name=sim.name,
                    ),
                    height=400,
                )

            if results.simulation_summary is not None:
                self.add_heading("Simulation Comparison", explain="simulation_comparison")
                self.add_table(results.simulation_summary)

            self.add_heading("Probability Analysis", explain="probability")
            for sim in results.simulations:
                self.add_stat_grid(
                    [
                        (f"{sim.name} · Expected", f"${sim.expected_value:,.0f}"),
                        (f"{sim.name} · P(Loss)", f"{sim.prob_loss:.1%}"),
                        (f"{sim.name} · Median P50", f"${sim.percentiles['P50']:,.0f}"),
                        (f"{sim.name} · Worst P5", f"${sim.percentiles['P5']:,.0f}"),
                        (f"{sim.name} · Best P95", f"${sim.percentiles['P95']:,.0f}"),
                    ],
                    columns=5,
                )
        else:
            self.add_interpretation(
                "Monte Carlo simulations not available. Run an analysis to generate "
                "forecasts."
            )

        # Historical risk statistics (fallback / supplement)
        if results.active:
            rets = results.active.daily_returns
            gtp = T.gain_to_pain(rets)
            var95, cvar95 = T.var_cvar(rets, 0.95)
            self.add_heading("Active Portfolio: Historical Risk Statistics", explain="historical_risk")
            self.add_stat_grid(
                [
                    ("Gain-to-Pain", f"{gtp:.2f}" if gtp else "N/A"),
                    ("Daily VaR (95%)", f"{var95:.2%}" if not np.isnan(var95) else "N/A"),
                    ("Daily CVaR (95%)", f"{cvar95:.2%}" if not np.isnan(cvar95) else "N/A"),
                ],
                columns=3,
            )
