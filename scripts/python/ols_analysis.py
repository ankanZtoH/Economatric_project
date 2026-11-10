#!/usr/bin/env python3
"""
ols_analysis.py — OLS estimation and diagnostics for all four datasets.

Fits OLS models, computes VIF, generates residual diagnostics plots,
and stores results for downstream WLS/GLS comparison.

Author : Ankan Bera
Created: 2026-11-09
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    PROCESSED_DATA_DIR, FIGURES_DIR, TABLES_DIR,
    MODEL_SPECS, FIGURE_SIZE, DPI, MPL_STYLE,
    setup_logging,
)

logger = setup_logging(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================================
# Helper Functions
# ============================================================================

def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Variance Inflation Factors for each predictor.

    Parameters
    ----------
    X : pd.DataFrame
        Design matrix (without constant).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['Variable', 'VIF'].
    """
    X_numeric = X.select_dtypes(include=[np.number])
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X_numeric.columns
    vif_data["VIF"] = [
        variance_inflation_factor(X_numeric.values, i)
        for i in range(X_numeric.shape[1])
    ]
    return vif_data.sort_values("VIF", ascending=False)


def plot_residuals_vs_fitted(model, title: str, save_path: str) -> None:
    """Scatter plot of residuals versus fitted values with LOESS smoother."""
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    fitted = model.fittedvalues
    resid = model.resid
    ax.scatter(fitted, resid, alpha=0.4, s=15, edgecolor="none")
    ax.axhline(y=0, color="red", linestyle="--", linewidth=1)

    # LOESS smoother
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        smoothed = lowess(resid, fitted, frac=0.3)
        ax.plot(smoothed[:, 0], smoothed[:, 1], color="red", linewidth=2)
    except ImportError:
        pass

    ax.set_xlabel("Fitted Values")
    ax.set_ylabel("Residuals")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved residuals vs fitted plot to %s", save_path)


def plot_qq(residuals: np.ndarray, title: str, save_path: str) -> None:
    """Q-Q plot of residuals against normal distribution."""
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved Q-Q plot to %s", save_path)


# ============================================================================
# OLS Fitting Functions
# ============================================================================

def fit_ols(df: pd.DataFrame, dep_var: str, predictors: list,
            dataset_name: str) -> sm.regression.linear_model.RegressionResultsWrapper:
    """
    Fit an OLS model and generate diagnostics.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataset.
    dep_var : str
        Dependent variable column name.
    predictors : list of str
        Predictor column names.
    dataset_name : str
        Label for output files (e.g. 'boston').

    Returns
    -------
    statsmodels RegressionResults
    """
    logger.info("Fitting OLS for %s: %s ~ %s", dataset_name, dep_var,
                " + ".join(predictors))

    # Prepare data — drop rows with NaN in relevant columns
    cols = [dep_var] + predictors
    data = df[cols].dropna()
    y = data[dep_var]
    X = sm.add_constant(data[predictors].astype(float))

    model = sm.OLS(y, X).fit()

    # Print summary
    print(f"\n{'=' * 70}")
    print(f"  OLS Results: {dataset_name.upper()}")
    print(f"{'=' * 70}")
    print(model.summary())

    # VIF
    vif = compute_vif(data[predictors].astype(float))
    print(f"\nVariance Inflation Factors ({dataset_name}):")
    print(vif.to_string(index=False))

    # Save VIF table
    vif.to_csv(os.path.join(TABLES_DIR, f"vif_{dataset_name}.csv"), index=False)

    # Diagnostic plots
    plot_residuals_vs_fitted(
        model,
        f"Residuals vs Fitted — {dataset_name.title()} OLS",
        os.path.join(FIGURES_DIR, f"resid_vs_fitted_{dataset_name}.png"),
    )
    plot_qq(
        model.resid.values,
        f"Q-Q Plot — {dataset_name.title()} OLS Residuals",
        os.path.join(FIGURES_DIR, f"qq_plot_{dataset_name}.png"),
    )

    return model


# ============================================================================
# Dataset-specific OLS
# ============================================================================

def ols_boston() -> sm.regression.linear_model.RegressionResultsWrapper:
    """Fit OLS for Boston Housing."""
    df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "boston_clean.csv"))
    return fit_ols(df, "medv", ["crim", "zn", "indus", "rm", "age", "tax",
                                "ptratio", "lstat"], "boston")


def ols_ames() -> sm.regression.linear_model.RegressionResultsWrapper:
    """Fit OLS for Ames Housing."""
    df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "ames_clean.csv"))
    preds = ["GrLivArea", "LotArea", "OverallQual", "YearBuilt",
             "TotalBsmtSF", "GarageCars"]
    return fit_ols(df, "log_SalePrice", preds, "ames")


def ols_wdi() -> sm.regression.linear_model.RegressionResultsWrapper:
    """Fit pooled OLS for WDI Panel."""
    df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "wdi_panel.csv"))
    return fit_ols(df, "gdp_growth", ["lag_gdp_pc", "investment_rate",
                                       "inflation", "log_pop"], "wdi")


def ols_enterprise() -> sm.regression.linear_model.RegressionResultsWrapper:
    """Fit OLS for Enterprise Survey."""
    df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "enterprise_clean.csv"))
    return fit_ols(df, "log_revenue", ["log_capital", "log_labor",
                                        "firm_size_dummy"], "enterprise")


# ============================================================================
# Run All & Comparison Table
# ============================================================================

def run_all_ols() -> dict:
    """Fit OLS for all datasets and return dict of models."""
    models = {}
    for name, func in [("boston", ols_boston), ("ames", ols_ames),
                        ("wdi", ols_wdi), ("enterprise", ols_enterprise)]:
        try:
            models[name] = func()
        except Exception as exc:
            logger.error("Failed OLS for %s: %s", name, exc)
    
    # Build comparison table
    rows = []
    for name, model in models.items():
        rows.append({
            "Dataset": name,
            "N": int(model.nobs),
            "R²": round(model.rsquared, 4),
            "Adj_R²": round(model.rsquared_adj, 4),
            "F_stat": round(model.fvalue, 2),
            "F_pval": f"{model.f_pvalue:.4e}",
            "AIC": round(model.aic, 2),
            "BIC": round(model.bic, 2),
        })
    
    summary_df = pd.DataFrame(rows)
    summary_path = os.path.join(TABLES_DIR, "ols_summary_all.csv")
    summary_df.to_csv(summary_path, index=False)
    
    print(f"\n{'=' * 70}")
    print("  OLS Model Comparison Summary")
    print(f"{'=' * 70}")
    print(summary_df.to_string(index=False))
    logger.info("Saved OLS summary to %s", summary_path)
    
    return models


def main() -> None:
    """Run all OLS analyses."""
    logger.info("Starting OLS analysis pipeline")
    models = run_all_ols()
    logger.info("OLS analysis complete. %d models fitted.", len(models))


if __name__ == "__main__":
    main()

