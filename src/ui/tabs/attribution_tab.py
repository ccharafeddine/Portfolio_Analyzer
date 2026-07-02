"""Attribution tab — Brinson-Fachler attribution, sector/factor exposure,
CAPM regressions + scatters, and factor-model regressions.

Transcribed from app.py's ``with tab_attribution:`` block (lines 635-744).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.charts import plotly_charts as charts

from .web_tab import WebTab

# Display order for the Fama-French factor models (keys of results.factor_models).
_FACTOR_MODELS = ("FF3", "Carhart 4-Factor", "FF5")


class AttributionTab(WebTab):
    def _populate(self, results) -> None:
        interp = results.interpretations or {}
        self.add_interpretation(interp.get("capm"))

        # Asset & sector attribution
        if results.asset_attribution is not None and not results.asset_attribution.empty:
            self.add_chart(
                charts.attribution_chart(
                    results.asset_attribution, title="Brinson–Fachler Attribution (Assets)"
                ),
                height=380,
                explain="attribution_assets",
            )
            self.add_table(results.asset_attribution, show_index=True)

        if results.sector_attribution is not None and not results.sector_attribution.empty:
            self.add_chart(
                charts.attribution_chart(
                    results.sector_attribution, title="Brinson–Fachler Attribution (Sectors)"
                ),
                height=380,
                explain="attribution_sectors",
            )
            self.add_table(results.sector_attribution, show_index=True)

        # Sector & factor exposure (each chart carries its own title + '?')
        exposure_figs, exposure_keys = [], []
        if results.sector_weights is not None and not results.sector_weights.empty:
            exposure_figs.append(charts.sector_donut_chart(results.sector_weights))
            exposure_keys.append("sector_weights")
        if results.factor_tilts is not None and not results.factor_tilts.empty:
            exposure_figs.append(charts.factor_tilts_chart(results.factor_tilts))
            exposure_keys.append("factor_tilts")
        if exposure_figs:
            self.add_chart_row(exposure_figs, height=340, explains=exposure_keys)
            if results.sector_weights is not None and not results.sector_weights.empty:
                from src.analytics.exposure import effective_n_sectors

                eff_n = effective_n_sectors(results.sector_weights)
                self.add_stat_grid([("Effective N Sectors", f"{eff_n:.1f}")], columns=4)

        # CAPM regression table + scatters
        if results.capm_results:
            self.add_heading("CAPM Regression Results", explain="capm")
            capm_df = pd.DataFrame(
                [
                    {
                        "Asset": r.ticker,
                        "Alpha (monthly)": f"{r.alpha:.4f}",
                        "Beta": f"{r.beta:.2f}",
                        "t(α)": f"{r.t_alpha:.2f}",
                        "t(β)": f"{r.t_beta:.2f}",
                        "R²": f"{r.r_squared:.3f}",
                    }
                    for r in results.capm_results
                ]
            )
            self.add_table(capm_df)

            if len(results.capm_results) <= 12:
                rets_m = results.monthly_returns
                rf_m = (1 + results.config.risk_free_rate) ** (1 / 12) - 1
                benchmark = results.config.benchmark
                scatters = []
                for r in results.capm_results:
                    if r.ticker in rets_m.columns and benchmark in rets_m.columns:
                        aligned = pd.concat(
                            [rets_m[r.ticker], rets_m[benchmark]], axis=1
                        ).dropna()
                        if not aligned.empty:
                            fig = charts.capm_scatter(
                                r.ticker,
                                aligned.iloc[:, 0] - rf_m,
                                aligned.iloc[:, 1] - rf_m,
                                r.alpha,
                                r.beta,
                            )
                            fig.update_layout(height=300)
                            scatters.append(fig)
                self.add_chart_grid(scatters, columns=3, height=300, explain="capm_scatter")

        # Fama-French factor-model regressions (real loadings, computed by the pipeline)
        factor_models = results.factor_models or {}
        if factor_models:
            self.add_heading("Fama-French Factor Loadings", explain="factor_regression")
            for model_name in _FACTOR_MODELS:
                df = factor_models.get(model_name)
                if df is None or df.empty:
                    continue
                self.add_chart(
                    charts.factor_loadings_chart(df, model_name),
                    height=340,
                    explain="factor_regression",
                )
                self.add_table(df)
