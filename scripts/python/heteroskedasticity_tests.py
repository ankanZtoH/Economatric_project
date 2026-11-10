#!/usr/bin/env python3
"""
heteroskedasticity_tests.py — Diagnostic tests for heteroskedasticity.

Applies Breusch-Pagan, White, and Goldfeld-Quandt tests to all OLS models.
Generates diagnostic plots (residuals² vs fitted, residuals vs predictors).

Author : Ankan Bera
Created: 2026-11-09
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import (
    het_breuschpagan,
    het_white,
    het_goldfeldquandt,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    PROCESSED_DATA_DIR, FIGURES_DIR, TABLES_DIR,
    FIGURE_SIZE, DPI, setup_logging,
)

logger = setup_logging(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================================
# Test Functions
# ============================================================================

def run_breusch_pagan(model) -> dict:
    """
    Run the Breusch-Pagan test for heteroskedasticity.

    Parameters
    ----------
    model : statsmodels OLS result

    Returns
    -------
    dict with keys: LM_stat, LM_pval, F_stat, F_pval
    """
    lm, lm_p, fval, f_p = het_breuschpagan(model.resid, model.model.exog)
    return {
        "Test": "Breusch-Pagan",
        "LM_stat": round(lm, 4),
        "LM_pval": round(lm_p, 6),
        "F_stat": round(fval, 4),
        "F_pval": round(f_p, 6),
        "Reject_5pct": "Yes" if lm_p < 0.05 else "No",
    }


def run_white_test(model) -> dict:
    """
    Run White's test for heteroskedasticity (general form).

    Parameters
    ----------
    model : statsmodels OLS result

    Returns
    -------
    dict with test results
    """
    try:
        lm, lm_p, fval, f_p = het_white(model.resid, model.model.exog)
        return {
            "Test": "White",
            "LM_stat": round(lm, 4),
            "LM_pval": round(lm_p, 6),
            "F_stat": round(fval, 4),
            "F_pval": round(f_p, 6),
            "Reject_5pct": "Yes" if lm_p < 0.05 else "No",
        }
    except Exception as exc:
        logger.warning("White test failed: %s", exc)
        return {
            "Test": "White",
            "LM_stat": np.nan, "LM_pval": np.nan,
            "F_stat": np.nan, "F_pval": np.nan,
            "Reject_5pct": "N/A",
        }


def run_goldfeld_quandt(model, order_var: np.ndarray, name: str = "") -> dict:
    """
    Run the Goldfeld-Quandt test for heteroskedasticity.

    Parameters
    ----------
    model : statsmodels OLS result
    order_var : array
        Variable to order observations by (suspected source of heterosk.).
    name : str
        Name of the ordering variable.

    Returns
    -------
    dict with test results
    """
    try:
        y = model.model.endog
        X = model.model.exog
        # Sort by the ordering variable
        sort_idx = np.argsort(order_var)
        y_sorted = y[sort_idx]
        X_sorted = X[sort_idx]

        fval, f_p, ordering = het_goldfeldquandt(y_sorted, X_sorted)
        return {
            "Test": f"Goldfeld-Quandt (by {name})",
            "LM_stat": np.nan,
            "LM_pval": np.nan,
            "F_stat": round(fval, 4),
            "F_pval": round(f_p, 6),
            "Reject_5pct": "Yes" if f_p < 0.05 else "No",
        }
    except Exception as exc:
        logger.warning("GQ test failed: %s", exc)
        return {
            "Test": f"Goldfeld-Quandt (by {name})",
            "LM_stat": np.nan, "LM_pval": np.nan,
            "F_stat": np.nan, "F_pval": np.nan,
            "Reject_5pct": "N/A",
        }


# ============================================================================
# Diagnostic Plots
# ============================================================================

def plot_resid_squared_vs_fitted(model, title: str, save_path: str) -> None:
    """Plot squared residuals against fitted values."""
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    fitted = model.fittedvalues
    resid_sq = model.resid ** 2
    ax.scatter(fitted, resid_sq, alpha=0.4, s=15, edgecolor="none", c="#C44E52")
    ax.set_xlabel("Fitted Values")
    ax.set_ylabel("Squared Residuals")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_resid_vs_predictors(model, data: pd.DataFrame,
                              predictors: list, dataset_name: str) -> None:
    """Plot residuals against each predictor variable."""
    n_pred = len(predictors)
    ncols = min(3, n_pred)
    nrows = (n_pred + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    if n_pred == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, pred in enumerate(predictors):
        if pred in data.columns:
            axes[i].scatter(data[pred].values[:len(model.resid)],
                           model.resid, alpha=0.3, s=10, edgecolor="none")
            axes[i].axhline(y=0, color="red", linestyle="--", linewidth=0.8)
            axes[i].set_xlabel(pred)
            axes[i].set_ylabel("Residuals")
            axes[i].set_title(f"Residuals vs {pred}")

    # Hide unused axes
    for j in range(n_pred, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"Residuals vs Predictors — {dataset_name.title()}", y=1.02)
    fig.tight_layout()
    save_path = os.path.join(FIGURES_DIR, f"resid_vs_preds_{dataset_name}.png")
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved residuals vs predictors plot to %s", save_path)


# ============================================================================
# Run Tests for a Single Dataset
# ============================================================================

def test_dataset(dataset_name: str, dep_var: str, predictors: list,
                 gq_order_var: str = None) -> list[dict]:
    """
    Run all heteroskedasticity tests for one dataset.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset (must match processed CSV name).
    dep_var : str
        Dependent variable name.
    predictors : list of str
        Predictor variable names.
    gq_order_var : str, optional
        Variable to use for Goldfeld-Quandt ordering.

    Returns
    -------
    list of dict
        Test results.
    """
    csv_map = {
        "boston": "boston_clean.csv",
        "ames": "ames_clean.csv",
        "wdi": "wdi_panel.csv",
        "enterprise": "enterprise_clean.csv",
    }
    df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, csv_map[dataset_name]))
    cols = [dep_var] + predictors
    data = df[cols].dropna()

    y = data[dep_var].values
    X = sm.add_constant(data[predictors].astype(float).values)
    model = sm.OLS(y, X).fit()

    results = []

    # Breusch-Pagan
    bp = run_breusch_pagan(model)
    bp["Dataset"] = dataset_name
    results.append(bp)

    # White
    wh = run_white_test(model)
    wh["Dataset"] = dataset_name
    results.append(wh)

    # Goldfeld-Quandt
    order_col = gq_order_var or predictors[0]
    if order_col in data.columns:
        gq = run_goldfeld_quandt(model, data[order_col].values, order_col)
    else:
        gq = run_goldfeld_quandt(model, data[predictors[0]].values, predictors[0])
    gq["Dataset"] = dataset_name
    results.append(gq)

    # Diagnostic plots
    plot_resid_squared_vs_fitted(
        model,
        f"Squared Residuals vs Fitted — {dataset_name.title()}",
        os.path.join(FIGURES_DIR, f"resid_sq_vs_fitted_{dataset_name}.png"),
    )
    plot_resid_vs_predictors(model, data, predictors, dataset_name)

    return results


# ============================================================================
# Run All Datasets
# ============================================================================

def run_all_tests() -> pd.DataFrame:
    """Run heteroskedasticity tests for all four datasets."""
    logger.info("Running heteroskedasticity tests for all datasets")

    all_results = []

    # Boston
    all_results.extend(test_dataset(
        "boston", "medv",
        ["crim", "zn", "indus", "rm", "age", "tax", "ptratio", "lstat"],
        gq_order_var="rm",
    ))

    # Ames
    all_results.extend(test_dataset(
        "ames", "log_SalePrice",
        ["GrLivArea", "LotArea", "OverallQual", "YearBuilt",
         "TotalBsmtSF", "GarageCars"],
        gq_order_var="GrLivArea",
    ))

    # WDI
    all_results.extend(test_dataset(
        "wdi", "gdp_growth",
        ["lag_gdp_pc", "investment_rate", "inflation", "log_pop"],
        gq_order_var="lag_gdp_pc",
    ))

    # Enterprise
    all_results.extend(test_dataset(
        "enterprise", "log_revenue",
        ["log_capital", "log_labor", "firm_size_dummy"],
        gq_order_var="log_capital",
    ))

    # Compile results
    results_df = pd.DataFrame(all_results)
    col_order = ["Dataset", "Test", "LM_stat", "LM_pval", "F_stat",
                 "F_pval", "Reject_5pct"]
    results_df = results_df[col_order]

    # Save
    out_path = os.path.join(TABLES_DIR, "heterosk_tests.csv")
    results_df.to_csv(out_path, index=False)
    logger.info("Saved heteroskedasticity test results to %s", out_path)

    # Print
    print(f"\n{'=' * 80}")
    print("  Heteroskedasticity Test Results Summary")
    print(f"{'=' * 80}")
    print(results_df.to_string(index=False))

    # Interpretation
    print(f"\n{'─' * 80}")
    print("  Interpretation:")
    for ds in ["boston", "ames", "wdi", "enterprise"]:
        ds_results = results_df[results_df["Dataset"] == ds]
        bp_row = ds_results[ds_results["Test"] == "Breusch-Pagan"]
        if not bp_row.empty:
            pval = bp_row.iloc[0]["LM_pval"]
            if pd.notna(pval) and pval < 0.05:
                print(f"  {ds.upper()}: BP test REJECTS homoskedasticity "
                      f"(p={pval:.4f}) → WLS/GLS recommended.")
            elif pd.notna(pval):
                print(f"  {ds.upper()}: BP test FAILS TO REJECT "
                      f"homoskedasticity (p={pval:.4f}).")
    print(f"{'─' * 80}")

    return results_df


def main() -> None:
    """Entry point for heteroskedasticity testing."""
    run_all_tests()


if __name__ == "__main__":
    main()

