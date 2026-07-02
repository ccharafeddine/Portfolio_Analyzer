"""Forward-looking scenario analysis (what-if shocks).

Unlike the historical stress tests (which slice real past windows), this estimates the
portfolio's response to *hypothetical* shocks. Two kinds of driver:

- **Macro factors** (SPY, TLT, USO, GLD, UUP, HYG): the portfolio's sensitivity is a
  partial beta from one joint regression of portfolio returns on all macro factors.
- **Holdings**: the direct effect of shocking a holding is simply its portfolio weight.

Estimated portfolio return for a scenario = sum over shocked drivers of beta * shock.
The pipeline precomputes the betas; the Risk tab does the beta*shock arithmetic live in
the browser so users can drag shocks around without re-running the analysis.
"""

from __future__ import annotations

import pandas as pd

# (display name, proxy ticker)
MACRO_FACTORS: list[tuple[str, str]] = [
    ("Equity (S&P 500)", "SPY"),
    ("Long Treasuries", "TLT"),
    ("Oil", "USO"),
    ("Gold", "GLD"),
    ("US Dollar", "UUP"),
    ("HY Credit", "HYG"),
]

# Preset scenarios: name -> {driver display name: shock in percent}
PRESETS: dict[str, dict[str, float]] = {
    "Equity −20%": {"Equity (S&P 500)": -20.0},
    "Rate spike": {"Long Treasuries": -10.0},
    "Oil shock −30%": {"Oil": -30.0},
    "Risk-off": {
        "Equity (S&P 500)": -15.0, "HY Credit": -8.0,
        "Gold": 5.0, "Long Treasuries": 4.0,
    },
}

_MIN_OBS = 40


def _macro_betas(port_returns: pd.Series, macro_prices: pd.DataFrame) -> list[dict]:
    if macro_prices is None or macro_prices.empty:
        return []
    try:
        import statsmodels.api as sm
    except Exception:
        return []

    rets = macro_prices.pct_change().dropna(how="all")
    present = [(n, t) for n, t in MACRO_FACTORS if t in rets.columns]
    if not present:
        return []
    x = rets[[t for _, t in present]].dropna()
    common = port_returns.index.intersection(x.index)
    if len(common) < _MIN_OBS:
        return []
    try:
        res = sm.OLS(port_returns.loc[common], sm.add_constant(x.loc[common])).fit()
    except Exception:
        return []
    return [
        {"name": n, "ticker": t, "group": "Macro",
         "beta": round(float(res.params.get(t, 0.0)), 3)}
        for n, t in present
    ]


def build_scenario_model(
    port_returns: pd.Series,
    macro_prices: pd.DataFrame,
    weights: dict[str, float],
    value: float,
) -> dict:
    """Precompute drivers + betas for the interactive builder.

    Returns ``{"value", "drivers": [{name, ticker, group, beta}], "presets"}`` where a
    Macro driver's beta is a partial regression beta and a Holding's beta is its weight.
    """
    drivers: list[dict] = []
    if port_returns is not None and not port_returns.empty:
        drivers.extend(_macro_betas(port_returns, macro_prices))

    total_w = sum(v for v in weights.values() if v) or 1.0
    for t, w in weights.items():
        drivers.append({
            "name": t, "ticker": t, "group": "Holding",
            "beta": round(float(w) / total_w, 4),
        })

    return {"value": float(value), "drivers": drivers, "presets": PRESETS}
