import os
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt


# -------------------------------------------------------------
# Helper: Run OLS for one asset vs one factor set
# -------------------------------------------------------------
def _run_single_regression(asset_returns: pd.Series, factors: pd.DataFrame):
    """
    asset_returns: monthly returns for one asset
    factors: factor dataframe including RF column OR proxies

    We coerce both to a common monthly PeriodIndex so that
    month-begin vs month-end issues don't kill the overlap.
    """
    if asset_returns is None or factors is None:
        raise ValueError("asset_returns and factors must be non-null")

    # Ensure Series with a name
    ar = pd.Series(asset_returns).copy()
    asset_name = ar.name or "asset"

    # Align both to monthly periods
    ar.index = pd.to_datetime(ar.index).to_period("M")
    fdf = factors.copy()
    fdf.index = pd.to_datetime(fdf.index).to_period("M")

    df = pd.concat([ar, fdf], axis=1, join="inner").dropna()

    if df.empty:
        raise ValueError(
            f"No overlapping monthly data between {asset_name} and factor set."
        )

    # First column is the dependent variable
    if "RF" in df.columns:
        y = df.iloc[:, 0] - df["RF"]
        X = df.drop(columns=["RF"])
    else:
        # For QUAL/SPLV there is no RF — subtract 0 (already set)
        y = df.iloc[:, 0]
        X = df.drop(columns=["RF"], errors="ignore")

    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()

    return model


# -------------------------------------------------------------
# Helper: Plot bar chart of betas (now in per-model subfolders)
# -------------------------------------------------------------
def _plot_factor_betas(model, asset_name, outdir, model_name):
    """
    Save factor beta bar chart for one asset & factor model into:

        <outdir>/factor_charts/<model_name>/factor_betas_<model>_<asset>.png
    """
    params = model.params.drop("const", errors="ignore")

    # Create per-model folder under outputs/
    charts_root = os.path.join(outdir, "factor_charts", model_name)
    os.makedirs(charts_root, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    params.plot(kind="bar", ax=ax, color="steelblue", alpha=0.85)

    ax.set_title(f"{asset_name} — {model_name} Factor Loadings")
    ax.set_ylabel("Beta")
    ax.grid(True, alpha=0.3)

    fname = f"factor_betas_{model_name}_{asset_name}.png"
    outpath = os.path.join(charts_root, fname)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)

    return outpath


# -------------------------------------------------------------
# Main runner: For each factor model and each asset
# -------------------------------------------------------------
def run_all_factor_models(asset_rets: pd.DataFrame, factors_dict: dict, outdir: str):
    """
    asset_rets: DataFrame of monthly returns for each asset
    factors_dict: {
        "ff3": df,
        "carhart4": df,
        "ff5": df,
        "quality_lowvol": df
    }
    outdir: save location
    """
    os.makedirs(outdir, exist_ok=True)

    # Store summary CSVs and the PNGs for the app
    summary_paths = {}
    beta_chart_paths = []

    for model_name, fdf in factors_dict.items():
        rows = []

        for asset in asset_rets.columns:
            try:
                reg = _run_single_regression(asset_rets[asset], fdf)
                coeffs = reg.params.to_dict()
                tstats = reg.tvalues.to_dict()

                row = {"Asset": asset}
                for k in coeffs:
                    row[f"{k}_coef"] = coeffs[k]
                for k in tstats:
                    row[f"{k}_tstat"] = tstats[k]

                # annualize alpha (const) if present
                row["alpha_annualized"] = coeffs.get("const", 0.0) * 12.0

                rows.append(row)

                # Generate beta bar chart in per-model subfolder
                betapng = _plot_factor_betas(reg, asset, outdir, model_name)
                beta_chart_paths.append(betapng)

            except Exception as e:
                # Don't crash the whole run if one regression fails
                print(f"[factor regression] Failed for {asset} in {model_name}: {e}")

        # If nothing succeeded for this model, skip writing its CSV
        if not rows:
            print(
                f"[factor regression] No successful regressions for model "
                f"{model_name}; skipping summary CSV."
            )
            continue

        # Save summary table at top level of outputs/
        df_summary = pd.DataFrame(rows)
        out_path = os.path.join(outdir, f"factor_regression_{model_name}.csv")
        df_summary.to_csv(out_path, index=False)
        summary_paths[model_name] = out_path

    # Write manifest JSON so app.py can load easily
    manifest = {
        "tables": summary_paths,
        "charts": beta_chart_paths,
    }
    with open(os.path.join(outdir, "factor_regression_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=4)

    print("[factor regression] Completed all factor regression models.")
