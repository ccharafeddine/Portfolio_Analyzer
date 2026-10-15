"""Attribution tab — Brinson-Fachler attribution, sector/factor exposure,
CAPM regressions + scatters, and factor-model regressions.

Transcribed from app.py's ``with tab_attribution:`` block (lines 635-744).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.charts import plotly_charts as charts

from ..widgets.html_table import Col, rich_table_html
from .web_tab import WebTab

_SECTOR_COLS = [Col("Sector", "Sector", "label"), Col("Port weight", "Port. wt", "pct", 1),
                Col("Bench weight", "Bench wt", "pct", 1), Col("Allocation", "Allocation", "spct"),
                Col("Selection", "Selection", "spct"), Col("Interaction", "Interaction", "spct"),
                Col("Total", "Total", "spct")]


def _factor_display(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Coefficient with its t-stat underneath-style: '1.05 (44.6)'; alpha annualized."""
    factors = [c[:-5] for c in df.columns if c.endswith("_coef") and c != "const_coef"]
    out = pd.DataFrame({"Asset": df["Asset"]})
    out["Alpha"] = [f"{(1 + a) ** 252 - 1:+.1%} ({t:.1f})"
                    for a, t in zip(df["const_coef"], df.get("const_t", [np.nan] * len(df)))]
    for f in factors:
        tcol = df.get(f"{f}_t", pd.Series([np.nan] * len(df)))
        out[f] = [f"{b:.2f} ({t:.1f})" for b, t in zip(df[f"{f}_coef"], tcol)]
    out["R2"] = df["R2"]
    cols = [Col("Asset", "Asset", "sym"), Col("Alpha", "Alpha ann. (t)", "text")]
    cols += [Col(f, f"{f} (t)", "text") for f in factors] + [Col("R2", "R²", "num")]
    if "N" in df.columns:
        out["N"] = df["N"]
        cols.append(Col("N", "Obs", "int"))
    return out, cols

# Fama-French models: results key -> (sub-heading label, explanation key). Ordered.
_FACTOR_MODELS = {
    "FF3": ("FF3 — Market, Size, Value", "factor_ff3"),
    "Carhart 4-Factor": ("Carhart 4-Factor — + Momentum", "factor_carhart4"),
    "FF5": ("FF5 — + Profitability & Investment", "factor_ff5"),
}


class AttributionTab(WebTab):
    def _populate(self, results) -> None:
        interp = results.interpretations or {}
        self.add_interpretation(interp.get("capm"))

        # Sector Brinson-Fachler vs the benchmark fund.
        sa = getattr(results, "sector_attribution", None)
        meta = getattr(results, "sector_attribution_meta", None) or {}
        if sa is not None and not sa.empty:
            tot = {"Sector": "Total", "Port weight": sa["Port weight"].sum(),
                   "Bench weight": sa["Bench weight"].sum()}
            for c in ("Allocation", "Selection", "Interaction", "Total"):
                tot[c] = float(sa[c].sum())
            aside = (f"Portfolio {meta.get('port_return', np.nan):+.1%} · sector benchmark "
                     f"{meta.get('bench_return', np.nan):+.1%} · weights: {meta.get('source', '')}")
            self.add_split([
                (self.panel_html("Sector attribution", self.chart_html(
                    charts.sector_attribution_chart(sa), 360), explain="sector_brinson"), 1),
                (self.panel_html("Allocation vs selection", rich_table_html(
                    sa, _SECTOR_COLS, total=tot, table_id="sect"), aside=aside), 1.25),
            ])

        # Benchmark-relative attribution by holding — point-in-time totals + over time.
        # (A single-ticker benchmark can't support true per-asset Brinson-Fachler, so we
        # decompose active return directly: weight × excess return vs the benchmark.)
        ts = getattr(results, "ts_attribution", None)
        if ts is not None and not ts.empty:
            totals = ts.sum().sort_values(ascending=False)
            self.add_chart(
                charts.contribution_bar_chart(totals), height=360,
                explain="attribution_contribution",
            )
            self.add_table(pd.DataFrame({
                "Holding": totals.index.tolist(),
                "Contribution to Active Return": [f"{v:+.2%}" for v in totals.values],
            }))
            self.add_chart(
                charts.attribution_timeseries_chart(ts), height=400,
                explain="attribution_timeseries",
            )

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

        # Fama-French factor-model regressions (real loadings, computed by the pipeline).
        # One collapsible section; each model nests under it (h4) with its own tooltip.
        factor_models = results.factor_models or {}
        if factor_models:
            self.add_heading("Fama-French Factor Loadings", explain="factor_regression")
            for model_name, (label, explain_key) in _FACTOR_MODELS.items():
                df = factor_models.get(model_name)
                if df is None or df.empty:
                    continue
                self.add_subchart(
                    charts.factor_loadings_chart(df, model_name),
                    height=340,
                    explain=explain_key,
                    title=label,
                )
                disp, cols = _factor_display(df)
                self.add_rich_table(disp, cols)
