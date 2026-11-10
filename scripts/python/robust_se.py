#!/usr/bin/env python3
"""
robust_se.py — Robust and cluster-robust standard error computations.

Computes HC0–HC3, cluster-robust, and Newey-West HAC standard errors
for all OLS models and produces comparison tables.

Author : Ankan Bera
Created: 2026-11-09
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    PROCESSED_DATA_DIR, TABLES_DIR, setup_logging,
)

logger = setup_logging(__name__)
warnings.filterwarnings("ignore")


# ============================================================================
# Core SE Computation Functions
# ============================================================================

def compute_all_robust_se(model, cluster_groups=None, maxlags=None) -> pd.DataFrame:
    """Compute multiple types of standard errors for an OLS model."""
    var_names = list(getattr(model.model, 'exog_names', [f"x{i}" for i in range(len(model.params))]))

    result = pd.DataFrame({
        "Variable": var_names,
        "Coef": np.asarray(model.params)
    })
    result["OLS_SE"] = np.asarray(model.bse)

    # HC0, HC1, HC3
    for hc_type in ["HC0", "HC1", "HC3"]:
        try:
            robust = model.get_robustcov_results(cov_type=hc_type)
            result[hc_type] = np.asarray(robust.bse)
        except Exception:
            result[hc_type] = np.nan

    # Cluster-robust SEs
    if cluster_groups is not None:
        try:
            cl_model = model.get_robustcov_results(
                cov_type="cluster",
                groups=cluster_groups,
            )
            result["Cluster_SE"] = np.asarray(cl_model.bse)
        except Exception as e:
            result["Cluster_SE"] = np.nan
    else:
        result["Cluster_SE"] = np.nan

    # Newey-West HAC SEs
    if maxlags is None:
        maxlags = int(model.nobs ** (1 / 3))
    try:
        hac_model = model.get_robustcov_results(
            cov_type="HAC", maxlags=maxlags
        )
        result["NW_SE"] = np.asarray(hac_model.bse)
    except Exception:
        result["NW_SE"] = np.nan

    return result


def se_ratio_summary(se_table: pd.DataFrame) -> pd.DataFrame:
    """Compute SE ratios relative to OLS for interpretability."""
    ratio = se_table[["Variable"]].copy()
    for col in ["HC0", "HC1", "HC3", "Cluster_SE", "NW_SE"]:
        if col in se_table.columns:
            ratio[f"{col}/OLS"] = (se_table[col] / se_table["OLS_SE"]).round(4)
    return ratio


# ============================================================================
# Dataset-Specific SE Comparisons
# ============================================================================

def robust_se_boston() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "boston_clean.csv"))
    preds = ["crim", "zn", "indus", "rm", "age", "tax", "ptratio", "lstat"]
    data = df[["medv"] + preds].dropna()
    y = data["medv"]
    X = sm.add_constant(data[preds].astype(float))
    model = sm.OLS(y, X).fit()
    return compute_all_robust_se(model)


def robust_se_ames() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "ames_clean.csv"))
    preds = ["GrLivArea", "LotArea", "OverallQual", "YearBuilt", "TotalBsmtSF", "GarageCars"]
    data = df[["log_SalePrice"] + preds].dropna()
    y = data["log_SalePrice"]
    X = sm.add_constant(data[preds].astype(float))
    model = sm.OLS(y, X).fit()
    return compute_all_robust_se(model)


def robust_se_wdi() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "wdi_panel.csv"))
    preds = ["lag_gdp_pc", "investment_rate", "inflation", "log_pop"]
    data = df[["gdp_growth", "country_id"] + preds].dropna()
    y = data["gdp_growth"]
    X = sm.add_constant(data[preds].astype(float))
    model = sm.OLS(y, X).fit()
    return compute_all_robust_se(model, cluster_groups=data["country_id"], maxlags=5)


def robust_se_enterprise() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "enterprise_clean.csv"))
    preds = ["log_capital", "log_labor", "firm_size_dummy"]
    data = df[["log_revenue", "country"] + preds].dropna()
    y = data["log_revenue"]
    X = sm.add_constant(data[preds].astype(float))
    model = sm.OLS(y, X).fit()
    return compute_all_robust_se(model, cluster_groups=data["country"])


# ============================================================================
# Run All
# ============================================================================

def run_all_robust_se() -> None:
    datasets = {
        "boston": robust_se_boston,
        "ames": robust_se_ames,
        "wdi": robust_se_wdi,
        "enterprise": robust_se_enterprise,
    }

    all_tables = []
    for name, func in datasets.items():
        try:
            se_table = func()
            se_table.insert(0, "Dataset", name)
            all_tables.append(se_table)

            print(f"\n{'=' * 80}")
            print(f"  Robust SE Comparison: {name.upper()}")
            print(f"{'=' * 80}")
            print(se_table.round(6).to_string(index=False))

            ratios = se_ratio_summary(se_table)
            print(f"\n  SE Ratios (relative to OLS):")
            print(ratios.to_string(index=False))

        except Exception as exc:
            logger.error("Robust SE failed for %s: %s", name, exc)

    if all_tables:
        master = pd.concat(all_tables, ignore_index=True)
        out_path = os.path.join(TABLES_DIR, "robust_se_comparison.csv")
        master.to_csv(out_path, index=False)
        logger.info("Saved robust SE comparison to %s", out_path)


def main() -> None:
    run_all_robust_se()


if __name__ == "__main__":
    main()
