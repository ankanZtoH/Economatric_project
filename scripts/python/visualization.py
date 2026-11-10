#!/usr/bin/env python3
"""
visualization.py — Reusable plotting functions for the GLS/WLS project.

Provides consistent, publication-quality plotting utilities used by
other modules. All plots are saved at 300 DPI with tight layout.

Author : Ankan Bera
Created: 2026-11-09
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import FIGURE_SIZE, DPI, COLOR_PALETTE, setup_logging

logger = setup_logging(__name__)


# ============================================================================
# Core Plot Functions
# ============================================================================

def plot_residuals_vs_fitted(model, title: str, save_path: str,
                              figsize=FIGURE_SIZE) -> None:
    """
    Scatter plot of residuals vs fitted values with LOWESS smoother.

    Parameters
    ----------
    model : statsmodels regression result
    title : str
    save_path : str
    """
    fig, ax = plt.subplots(figsize=figsize)
    fitted = model.fittedvalues
    resid = model.resid

    ax.scatter(fitted, resid, alpha=0.35, s=15, color=COLOR_PALETTE[0],
               edgecolor="none")
    ax.axhline(y=0, color="black", linestyle="--", linewidth=0.8)

    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        sm = lowess(resid, fitted, frac=0.3)
        ax.plot(sm[:, 0], sm[:, 1], color=COLOR_PALETTE[3], linewidth=2,
                label="LOWESS")
        ax.legend()
    except ImportError:
        pass

    ax.set_xlabel("Fitted Values", fontsize=12)
    ax.set_ylabel("Residuals", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_qq(residuals: np.ndarray, title: str, save_path: str,
            figsize=FIGURE_SIZE) -> None:
    """Q-Q plot of residuals against normal distribution."""
    fig, ax = plt.subplots(figsize=figsize)
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.get_lines()[0].set_markerfacecolor(COLOR_PALETTE[0])
    ax.get_lines()[0].set_alpha(0.5)
    ax.get_lines()[1].set_color(COLOR_PALETTE[3])
    fig.tight_layout()
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_resid_vs_predictor(residuals: np.ndarray, predictor: np.ndarray,
                             pred_name: str, save_path: str,
                             figsize=FIGURE_SIZE) -> None:
    """Scatter plot of residuals vs a single predictor."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(predictor, residuals, alpha=0.35, s=15,
               color=COLOR_PALETTE[1], edgecolor="none")
    ax.axhline(y=0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel(pred_name, fontsize=12)
    ax.set_ylabel("Residuals", fontsize=12)
    ax.set_title(f"Residuals vs {pred_name}", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_coefficient_comparison(models_dict: dict, save_path: str,
                                 title: str = "Coefficient Comparison",
                                 figsize=None) -> None:
    """
    Dot plot of coefficients with 95% CIs for multiple models.

    Parameters
    ----------
    models_dict : dict
        Mapping of label → statsmodels result.
    save_path : str
    title : str
    """
    # Collect all variable names (exclude const)
    all_vars = []
    for model in models_dict.values():
        for v in model.params.index:
            if v != "const" and v not in all_vars:
                all_vars.append(v)

    n_vars = len(all_vars)
    n_models = len(models_dict)
    if figsize is None:
        figsize = (max(8, n_vars * 1.2), max(6, n_models * 1.5))

    fig, ax = plt.subplots(figsize=figsize)
    y_pos = np.arange(n_vars)
    height = 0.8 / n_models

    for i, (label, model) in enumerate(models_dict.items()):
        coefs = []
        errs = []
        for var in all_vars:
            if var in model.params.index:
                coefs.append(model.params[var])
                errs.append(1.96 * model.bse[var])
            else:
                coefs.append(np.nan)
                errs.append(np.nan)

        offset = (i - n_models / 2 + 0.5) * height
        color = COLOR_PALETTE[i % len(COLOR_PALETTE)]
        ax.barh(y_pos + offset, coefs, height, xerr=errs,
                label=label, color=color, alpha=0.8, capsize=2)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(all_vars)
    ax.axvline(x=0, color="grey", linestyle="--", linewidth=0.5)
    ax.set_xlabel("Coefficient ± 1.96 SE")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_mc_results(results_df: pd.DataFrame, metric: str,
                    param_col: str, save_path: str,
                    title: str = "Monte Carlo Results",
                    figsize=FIGURE_SIZE) -> None:
    """
    Grouped bar chart for Monte Carlo results.

    Parameters
    ----------
    results_df : pd.DataFrame
        Must have columns [method, param_col, metric].
    metric : str
        Column name for the Y axis (e.g. 'rmse', 'bias').
    param_col : str
        Column name for grouping on X axis.
    """
    fig, ax = plt.subplots(figsize=figsize)

    methods = results_df["method"].unique()
    params = sorted(results_df[param_col].unique())
    n_m = len(methods)
    x = np.arange(len(params))
    w = 0.8 / n_m

    for i, method in enumerate(methods):
        vals = []
        for p in params:
            sub = results_df[(results_df["method"] == method) &
                             (results_df[param_col] == p)]
            vals.append(sub[metric].values[0] if len(sub) > 0 else 0)

        offset = (i - n_m / 2 + 0.5) * w
        ax.bar(x + offset, vals, w, label=method,
               color=COLOR_PALETTE[i % len(COLOR_PALETTE)], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([str(p) for p in params])
    ax.set_xlabel(param_col, fontsize=12)
    ax.set_ylabel(metric.upper(), fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_ci_coverage(results_df: pd.DataFrame, param_col: str,
                     save_path: str, title: str = "CI Coverage",
                     figsize=FIGURE_SIZE) -> None:
    """Line plot of 95% CI coverage rates with nominal reference line."""
    fig, ax = plt.subplots(figsize=figsize)
    methods = results_df["method"].unique()

    for i, method in enumerate(methods):
        sub = results_df[results_df["method"] == method].sort_values(param_col)
        color = COLOR_PALETTE[i % len(COLOR_PALETTE)]
        ax.plot(sub[param_col], sub["coverage_95"], "o-", label=method,
                color=color, markersize=8, linewidth=2)

    ax.axhline(y=0.95, color="grey", linestyle="--", linewidth=1.5,
               label="Nominal 95%")
    ax.set_xlabel(param_col, fontsize=12)
    ax.set_ylabel("Coverage Rate", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylim(0.70, 1.02)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_weight_function(x: np.ndarray, weights: np.ndarray,
                          save_path: str,
                          title: str = "Weight Function",
                          figsize=FIGURE_SIZE) -> None:
    """Scatter plot showing the relationship between x and WLS weights."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(x, weights, alpha=0.5, s=20, color=COLOR_PALETTE[1],
               edgecolor="none")
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("Weight w(x)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def create_summary_dashboard(save_path: str) -> None:
    """
    Create a 2×2 summary dashboard with key diagnostic plots.
    (Placeholder — to be populated after analysis runs.)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax in axes.flatten():
        ax.text(0.5, 0.5, "Plot placeholder", ha="center", va="center",
                fontsize=14, color="grey")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Diagnostic Dashboard", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved dashboard to %s", save_path)

