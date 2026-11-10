#!/usr/bin/env python3
"""
wls_gls_analysis.py — Weighted Least Squares and Generalized Least Squares.

Fits WLS/GLS/FGLS models for each dataset, compares with OLS,
and generates side-by-side coefficient/SE comparison tables and plots.

Author : Ankan Bera
Created: 2026-11-09
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.linear_model import WLS, GLSAR
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    PROCESSED_DATA_DIR, FIGURES_DIR, TABLES_DIR,
    FIGURE_SIZE, DPI, COLOR_PALETTE, setup_logging,
)

logger = setup_logging(__name__)
warnings.filterwarnings("ignore")


# ============================================================================
# Helpers
# ============================================================================

def get_model_series(model):
    """Safely extract params and bse as pandas Series with named indexes."""
    names = list(getattr(model.model, 'exog_names', [f"x{i}" for i in range(len(model.params))]))
    params = pd.Series(np.asarray(model.params), index=names)
    bse = pd.Series(np.asarray(model.bse), index=names)
    tval = pd.Series(np.asarray(model.tvalues), index=names)
    pval = pd.Series(np.asarray(model.pvalues), index=names)
    return params, bse, tval, pval


def build_comparison_table(models: dict, dataset_name: str) -> pd.DataFrame:
    """Build a side-by-side comparison of coefficients and SEs across models."""
    rows = []
    for label, model in models.items():
        params, bse, tval, pval = get_model_series(model)
        for var in params.index:
            rows.append({
                "Dataset": dataset_name,
                "Variable": var,
                "Method": label,
                "Coef": round(float(params[var]), 6),
                "SE": round(float(bse[var]), 6),
                "t_stat": round(float(tval[var]), 4),
                "p_val": round(float(pval[var]), 6),
            })
    return pd.DataFrame(rows)


def plot_coefficient_comparison(models: dict, dataset_name: str, save_path: str) -> None:
    """Side-by-side coefficient plot with confidence intervals."""
    all_vars = []
    for model in models.values():
        params, _, _, _ = get_model_series(model)
        for v in params.index:
            if v != "const" and v not in all_vars:
                all_vars.append(v)

    n_vars = len(all_vars)
    n_models = len(models)
    fig, ax = plt.subplots(figsize=(max(10, n_vars * 1.5), 7))

    x_base = np.arange(n_vars)
    width = 0.8 / n_models
    offsets = np.linspace(-0.4 + width / 2, 0.4 - width / 2, n_models)

    for i, (label, model) in enumerate(models.items()):
        params, bse, _, _ = get_model_series(model)
        coefs = [float(params[v]) if v in params.index else np.nan for v in all_vars]
        cis = [1.96 * float(bse[v]) if v in bse.index else np.nan for v in all_vars]

        color = COLOR_PALETTE[i % len(COLOR_PALETTE)]
        ax.bar(x_base + offsets[i], coefs, width, yerr=cis,
               label=label, color=color, alpha=0.8, capsize=3)

    ax.set_xticks(x_base)
    ax.set_xticklabels(all_vars, rotation=45, ha="right")
    ax.set_ylabel("Coefficient ± 1.96 SE")
    ax.set_title(f"Coefficient Comparison — {dataset_name.title()}")
    ax.legend(loc="best")
    ax.axhline(y=0, color="grey", linewidth=0.5, linestyle="--")
    fig.tight_layout()
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved coefficient comparison to %s", save_path)


def fgls_weights(ols_model, X_no_const: pd.DataFrame) -> np.ndarray:
    """Estimate FGLS weights by regressing log(ê²) on predictors."""
    resid_sq = np.asarray(ols_model.resid) ** 2
    log_resid_sq = np.log(np.clip(resid_sq, 1e-10, None))

    X_var = sm.add_constant(np.asarray(X_no_const))
    var_model = sm.OLS(log_resid_sq, X_var).fit()
    predicted_log_var = var_model.fittedvalues
    weights = 1.0 / np.exp(predicted_log_var)
    weights = np.where(np.isfinite(weights) & (weights > 0), weights, 1.0)
    return weights


# ============================================================================
# Dataset Analyses
# ============================================================================

def analyze_boston() -> pd.DataFrame:
    logger.info("WLS/GLS analysis: Boston Housing")
    df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "boston_clean.csv"))
    preds = ["crim", "zn", "indus", "rm", "age", "tax", "ptratio", "lstat"]
    data = df[["medv"] + preds].dropna()

    y = data["medv"]
    X_raw = data[preds].astype(float)
    X = sm.add_constant(X_raw)

    ols = sm.OLS(y, X).fit()
    w_lstat = 1.0 / (np.clip(data["lstat"].values, 0.1, None) ** 2)
    wls = sm.WLS(y, X, weights=w_lstat).fit()

    fgls_w = fgls_weights(ols, X_raw)
    fgls_model = sm.WLS(y, X, weights=fgls_w).fit()
    ols_robust = ols.get_robustcov_results(cov_type="HC3")

    models = {"OLS": ols, "WLS (1/lstat²)": wls, "FGLS": fgls_model, "OLS-HC3": ols_robust}
    comp = build_comparison_table(models, "boston")
    plot_coefficient_comparison(models, "boston", os.path.join(FIGURES_DIR, "coef_compare_boston.png"))
    return comp


def analyze_ames() -> pd.DataFrame:
    logger.info("WLS/GLS analysis: Ames Housing")
    df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "ames_clean.csv"))
    preds = ["GrLivArea", "LotArea", "OverallQual", "YearBuilt", "TotalBsmtSF", "GarageCars"]
    data = df[["log_SalePrice"] + preds].dropna()

    y = data["log_SalePrice"]
    X_raw = data[preds].astype(float)
    X = sm.add_constant(X_raw)

    ols = sm.OLS(y, X).fit()
    w_area = 1.0 / np.clip(data["GrLivArea"].values, 1.0, None)
    wls = sm.WLS(y, X, weights=w_area).fit()

    fgls_w = fgls_weights(ols, X_raw)
    fgls_model = sm.WLS(y, X, weights=fgls_w).fit()
    ols_robust = ols.get_robustcov_results(cov_type="HC3")

    models = {"OLS": ols, "WLS (1/GrLivArea)": wls, "FGLS": fgls_model, "OLS-HC3": ols_robust}
    comp = build_comparison_table(models, "ames")
    plot_coefficient_comparison(models, "ames", os.path.join(FIGURES_DIR, "coef_compare_ames.png"))
    return comp


def analyze_wdi() -> pd.DataFrame:
    logger.info("GLS analysis: WDI Panel")
    df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "wdi_panel.csv"))
    preds = ["lag_gdp_pc", "investment_rate", "inflation", "log_pop"]
    data = df[["gdp_growth", "country_id"] + preds].dropna()

    y = data["gdp_growth"]
    X_raw = data[preds].astype(float)
    X = sm.add_constant(X_raw)

    ols = sm.OLS(y, X).fit()

    glsar = GLSAR(y, X, rho=1)
    glsar_result = glsar.iterative_fit(maxiter=20)

    ols_cluster = ols.get_robustcov_results(cov_type="cluster", groups=data["country_id"])
    ols_hac = ols.get_robustcov_results(cov_type="HAC", maxlags=5)

    models = {"Pooled OLS": ols, "GLSAR (AR1)": glsar_result, "OLS-Cluster": ols_cluster, "OLS-HAC": ols_hac}
    comp = build_comparison_table(models, "wdi")
    plot_coefficient_comparison(models, "wdi", os.path.join(FIGURES_DIR, "coef_compare_wdi.png"))
    return comp


def analyze_enterprise() -> pd.DataFrame:
    logger.info("WLS/GLS analysis: Enterprise Survey")
    df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "enterprise_clean.csv"))
    preds = ["log_capital", "log_labor", "firm_size_dummy"]
    data = df[["log_revenue", "country"] + preds].dropna()

    y = data["log_revenue"]
    X_raw = data[preds].astype(float)
    X = sm.add_constant(X_raw)

    ols = sm.OLS(y, X).fit()
    w_size = 1.0 / (data["firm_size_dummy"].values + 1.0)
    wls = sm.WLS(y, X, weights=w_size).fit()

    fgls_w = fgls_weights(ols, X_raw)
    fgls_model = sm.WLS(y, X, weights=fgls_w).fit()
    ols_cluster = ols.get_robustcov_results(cov_type="cluster", groups=data["country"])

    models = {"OLS": ols, "WLS (1/size)": wls, "FGLS": fgls_model, "OLS-Cluster": ols_cluster}
    comp = build_comparison_table(models, "enterprise")
    plot_coefficient_comparison(models, "enterprise", os.path.join(FIGURES_DIR, "coef_compare_enterprise.png"))
    return comp


# ============================================================================
# Main
# ============================================================================

def run_all_wls_gls() -> pd.DataFrame:
    all_comps = []
    for name, func in [("boston", analyze_boston), ("ames", analyze_ames),
                        ("wdi", analyze_wdi), ("enterprise", analyze_enterprise)]:
        try:
            comp = func()
            all_comps.append(comp)
        except Exception as exc:
            logger.error("WLS/GLS failed for %s: %s", name, exc)
            import traceback; traceback.print_exc()

    if all_comps:
        master = pd.concat(all_comps, ignore_index=True)
        out_path = os.path.join(TABLES_DIR, "wls_gls_comparison_all.csv")
        master.to_csv(out_path, index=False)
        logger.info("Saved master WLS/GLS comparison to %s", out_path)

        print(f"\n{'=' * 80}")
        print("  WLS/GLS vs OLS: Standard Error Comparison")
        print(f"{'=' * 80}")
        for ds in master["Dataset"].unique():
            ds_data = master[(master["Dataset"] == ds) & (master["Variable"] != "const")]
            pivot = ds_data.pivot_table(index="Variable", columns="Method", values="SE", aggfunc="first")
            print(f"\n  {ds.upper()}:")
            print(pivot.round(6).to_string())

        return master
    return pd.DataFrame()


def main() -> None:
    run_all_wls_gls()


if __name__ == "__main__":
    main()
