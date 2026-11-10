#!/usr/bin/env python3
"""
monte_carlo_simulation.py — Monte Carlo experiments comparing OLS, WLS, GLS.

Three scenarios:
  1. Heteroskedastic errors  (φ = 0, 0.5, 1.0, 1.5)
  2. AR(1) autocorrelation   (ρ = 0, 0.3, 0.6, 0.9)
  3. Clustered errors        (τ² = 0, 0.5, 1.0, 2.0)

Author : Ankan Bera
Created: 2026-11-09
"""

import os
import sys
import time
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
    FIGURES_DIR, TABLES_DIR,
    N_REPS, RANDOM_SEED,
    PHI_VALUES, RHO_VALUES, TAU_SQ_VALUES,
    N_CLUSTERS, OBS_PER_CLUSTER,
    FIGURE_SIZE, DPI, COLOR_PALETTE,
    setup_logging,
)

logger = setup_logging(__name__)
warnings.filterwarnings("ignore")

TRUE_BETA0 = 1.0
TRUE_BETA1 = 2.0
N = 100  # default sample size


def check_coverage(model, target=TRUE_BETA1, param_idx=1):
    """Check if the 95% confidence interval for param_idx covers the true value."""
    try:
        ci = np.asarray(model.conf_int(alpha=0.05))
        return int(ci[param_idx, 0] <= target <= ci[param_idx, 1])
    except Exception:
        return 0


# ============================================================================
# Scenario 1: Heteroskedastic Errors
# ============================================================================

def scenario_heterosk(phi: float, reps: int = N_REPS, n: int = N,
                      seed: int = RANDOM_SEED) -> dict:
    """
    Monte Carlo for heteroskedastic DGP: y = 1 + 2x + ε, Var(ε) = x^(2φ).

    Estimators:
      - OLS (naïve)
      - WLS correct: weights = 1/x^(2φ)
      - WLS misspecified: weights = 1/x²

    Returns
    -------
    dict with bias, variance, RMSE, coverage for each method.
    """
    rng = np.random.default_rng(seed)

    betas = {"ols": [], "wls_correct": [], "wls_misspec": []}
    covers = {"ols": 0, "wls_correct": 0, "wls_misspec": 0}

    for r in range(reps):
        x = rng.uniform(0.5, 10, size=n)  # avoid x≈0 for weight stability
        eps = rng.normal(0, x ** phi)
        y = TRUE_BETA0 + TRUE_BETA1 * x + eps
        X = sm.add_constant(x)

        # OLS
        ols = sm.OLS(y, X).fit()
        betas["ols"].append(ols.params[1])
        covers["ols"] += check_coverage(ols)

        # WLS correct
        w_correct = 1.0 / (x ** (2 * phi)) if phi > 0 else np.ones(n)
        wls_c = sm.WLS(y, X, weights=w_correct).fit()
        betas["wls_correct"].append(wls_c.params[1])
        covers["wls_correct"] += check_coverage(wls_c)

        # WLS misspecified (always use 1/x²)
        w_mis = 1.0 / (x ** 2)
        wls_m = sm.WLS(y, X, weights=w_mis).fit()
        betas["wls_misspec"].append(wls_m.params[1])
        covers["wls_misspec"] += check_coverage(wls_m)

        if (r + 1) % 1000 == 0:
            print(f"  Scenario 1 (φ={phi}): {r + 1}/{reps} reps done")

    results = {}
    for method in betas:
        b = np.array(betas[method])
        bias = np.mean(b) - TRUE_BETA1
        var = np.var(b, ddof=1)
        rmse = np.sqrt(np.mean((b - TRUE_BETA1) ** 2))
        coverage = covers[method] / reps
        results[method] = {
            "phi": phi, "method": method,
            "mean_beta1": round(np.mean(b), 6),
            "bias": round(bias, 6),
            "variance": round(var, 6),
            "rmse": round(rmse, 6),
            "coverage_95": round(coverage, 4),
        }

    return results


# ============================================================================
# Scenario 2: AR(1) Autocorrelation
# ============================================================================

def scenario_autocorr(rho: float, reps: int = N_REPS, T: int = N,
                      seed: int = RANDOM_SEED) -> dict:
    """
    Monte Carlo for AR(1) DGP: y_t = 1 + 2x_t + ε_t, ε_t = ρε_{t-1} + u_t.

    Estimators:
      - OLS
      - GLS with known ρ (Prais-Winsten transformation)
      - FGLS with estimated ρ
    """
    rng = np.random.default_rng(seed)

    betas = {"ols": [], "gls_known": [], "fgls": []}
    covers = {"ols": 0, "gls_known": 0, "fgls": 0}

    for r in range(reps):
        x = rng.uniform(0, 10, size=T)
        u = rng.normal(0, 1, size=T)

        # Generate AR(1) errors
        eps = np.zeros(T)
        eps[0] = u[0]
        for t in range(1, T):
            eps[t] = rho * eps[t - 1] + u[t]

        y = TRUE_BETA0 + TRUE_BETA1 * x + eps
        X = sm.add_constant(x)

        # OLS
        ols = sm.OLS(y, X).fit()
        betas["ols"].append(ols.params[1])
        covers["ols"] += check_coverage(ols)

        # GLS with known ρ (Prais-Winsten)
        if abs(rho) > 1e-6:
            sqrt_factor = np.sqrt(1 - rho ** 2)
            y_star = np.zeros(T)
            X_star = np.zeros((T, 2))
            y_star[0] = sqrt_factor * y[0]
            X_star[0] = sqrt_factor * X[0]
            for t in range(1, T):
                y_star[t] = y[t] - rho * y[t - 1]
                X_star[t] = X[t] - rho * X[t - 1]
            gls_known = sm.OLS(y_star, X_star).fit()
        else:
            gls_known = ols

        betas["gls_known"].append(gls_known.params[1])
        covers["gls_known"] += check_coverage(gls_known)

        # FGLS: estimate ρ from OLS residuals, then apply transformation
        resid = ols.resid
        rho_hat = np.corrcoef(resid[:-1], resid[1:])[0, 1]
        if abs(rho_hat) > 1e-6:
            sqrt_f = np.sqrt(max(1 - rho_hat ** 2, 0.01))
            y_fgls = np.zeros(T)
            X_fgls = np.zeros((T, 2))
            y_fgls[0] = sqrt_f * y[0]
            X_fgls[0] = sqrt_f * X[0]
            for t in range(1, T):
                y_fgls[t] = y[t] - rho_hat * y[t - 1]
                X_fgls[t] = X[t] - rho_hat * X[t - 1]
            fgls_model = sm.OLS(y_fgls, X_fgls).fit()
        else:
            fgls_model = ols

        betas["fgls"].append(fgls_model.params[1])
        covers["fgls"] += check_coverage(fgls_model)

        if (r + 1) % 1000 == 0:
            print(f"  Scenario 2 (ρ={rho}): {r + 1}/{reps} reps done")

    results = {}
    for method in betas:
        b = np.array(betas[method])
        bias = np.mean(b) - TRUE_BETA1
        var = np.var(b, ddof=1)
        rmse = np.sqrt(np.mean((b - TRUE_BETA1) ** 2))
        coverage = covers[method] / reps
        results[method] = {
            "rho": rho, "method": method,
            "mean_beta1": round(np.mean(b), 6),
            "bias": round(bias, 6),
            "variance": round(var, 6),
            "rmse": round(rmse, 6),
            "coverage_95": round(coverage, 4),
        }

    return results


# ============================================================================
# Scenario 3: Clustered Errors
# ============================================================================

def scenario_clustered(tau_sq: float, reps: int = N_REPS,
                       G: int = N_CLUSTERS, ng: int = OBS_PER_CLUSTER,
                       seed: int = RANDOM_SEED) -> dict:
    """
    Monte Carlo for clustered DGP: y_{ic} = 1 + 2x_{ic} + α_c + ε_{ic}.

    α_c ~ N(0, τ²), ε_{ic} ~ N(0, 1).

    Estimators:
      - OLS (naïve SEs)
      - OLS with cluster-robust SEs
      - RE-GLS (random effects)
    """
    rng = np.random.default_rng(seed)
    N_total = G * ng

    betas = {"ols": [], "ols_cluster": [], "re_gls": []}
    covers = {"ols": 0, "ols_cluster": 0, "re_gls": 0}

    clusters = np.repeat(np.arange(G), ng)

    for r in range(reps):
        x = rng.uniform(0, 10, size=N_total)

        # Cluster random effects
        alpha_c = rng.normal(0, np.sqrt(tau_sq), size=G) if tau_sq > 0 else np.zeros(G)
        alpha = np.repeat(alpha_c, ng)

        eps = rng.normal(0, 1, size=N_total)
        y = TRUE_BETA0 + TRUE_BETA1 * x + alpha + eps
        X = sm.add_constant(x)

        # OLS (naïve)
        ols = sm.OLS(y, X).fit()
        betas["ols"].append(ols.params[1])
        covers["ols"] += check_coverage(ols)

        # OLS with cluster-robust SEs
        try:
            ols_cl = ols.get_robustcov_results(cov_type="cluster", groups=clusters)
            betas["ols_cluster"].append(ols_cl.params[1])
            covers["ols_cluster"] += check_coverage(ols_cl)
        except Exception:
            betas["ols_cluster"].append(ols.params[1])

        # RE-GLS: simple random effects estimator
        try:
            resid = ols.resid
            group_means = np.array([resid[clusters == g].mean() for g in range(G)])
            sigma_between = np.var(group_means, ddof=1)
            sigma_within = np.mean([np.var(resid[clusters == g], ddof=1) for g in range(G)])
            sigma_alpha_sq = max(sigma_between - sigma_within / ng, 0)
            sigma_eps_sq = max(sigma_within, 0.01)

            theta = 1 - np.sqrt(sigma_eps_sq / (sigma_eps_sq + ng * sigma_alpha_sq))
            y_star = y.copy()
            X_star = X.copy()
            for g in range(G):
                mask = clusters == g
                y_bar_g = y[mask].mean()
                X_bar_g = X[mask].mean(axis=0)
                y_star[mask] = y[mask] - theta * y_bar_g
                X_star[mask] = X[mask] - theta * X_bar_g

            re_model = sm.OLS(y_star, X_star).fit()
            betas["re_gls"].append(re_model.params[1])
            covers["re_gls"] += check_coverage(re_model)
        except Exception:
            betas["re_gls"].append(ols.params[1])

        if (r + 1) % 1000 == 0:
            print(f"  Scenario 3 (τ²={tau_sq}): {r + 1}/{reps} reps done")

    results = {}
    for method in betas:
        b = np.array(betas[method])
        bias = np.mean(b) - TRUE_BETA1
        var = np.var(b, ddof=1)
        rmse = np.sqrt(np.mean((b - TRUE_BETA1) ** 2))
        coverage = covers[method] / reps
        results[method] = {
            "tau_sq": tau_sq, "method": method,
            "mean_beta1": round(np.mean(b), 6),
            "bias": round(bias, 6),
            "variance": round(var, 6),
            "rmse": round(rmse, 6),
            "coverage_95": round(coverage, 4),
        }

    return results


# ============================================================================
# Plotting Functions
# ============================================================================

def plot_mc_bars(df: pd.DataFrame, metric: str, param_col: str,
                 title: str, save_path: str) -> None:
    """Grouped bar chart of a metric by method and parameter value."""
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    methods = df["method"].unique()
    params = sorted(df[param_col].unique())
    n_methods = len(methods)
    x_base = np.arange(len(params))
    width = 0.8 / n_methods

    for i, method in enumerate(methods):
        subset = df[df["method"] == method].sort_values(param_col)
        vals = [subset[subset[param_col] == p][metric].values[0]
                if len(subset[subset[param_col] == p]) > 0 else 0
                for p in params]
        offset = (i - n_methods / 2 + 0.5) * width
        color = COLOR_PALETTE[i % len(COLOR_PALETTE)]
        ax.bar(x_base + offset, vals, width, label=method,
               color=color, alpha=0.85)

    ax.set_xticks(x_base)
    ax.set_xticklabels([str(p) for p in params])
    ax.set_xlabel(param_col)
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved MC bar chart to %s", save_path)


def plot_coverage(df: pd.DataFrame, param_col: str,
                  title: str, save_path: str) -> None:
    """Line plot of CI coverage by method."""
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    methods = df["method"].unique()

    for i, method in enumerate(methods):
        subset = df[df["method"] == method].sort_values(param_col)
        color = COLOR_PALETTE[i % len(COLOR_PALETTE)]
        ax.plot(subset[param_col], subset["coverage_95"], "o-",
                label=method, color=color, markersize=8, linewidth=2)

    ax.axhline(y=0.95, color="grey", linestyle="--", linewidth=1,
               label="Nominal 95%")
    ax.set_xlabel(param_col)
    ax.set_ylabel("95% CI Coverage")
    ax.set_title(title)
    ax.set_ylim(0.7, 1.02)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved coverage plot to %s", save_path)


# ============================================================================
# Main: Run All Scenarios
# ============================================================================

def run_all_mc() -> dict:
    """Run all three Monte Carlo scenarios and save results."""
    all_results = {}

    # ── Scenario 1: Heteroskedasticity ────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SCENARIO 1: Heteroskedastic Errors")
    print("=" * 70)
    rows_s1 = []
    t0 = time.time()
    for phi in PHI_VALUES:
        print(f"\n  Running φ = {phi} ...")
        res = scenario_heterosk(phi)
        for method_res in res.values():
            rows_s1.append(method_res)
    df_s1 = pd.DataFrame(rows_s1)
    df_s1.to_csv(os.path.join(TABLES_DIR, "mc_scenario1.csv"), index=False)
    print(f"\n  Scenario 1 complete in {time.time() - t0:.1f}s")
    print(df_s1.to_string(index=False))

    plot_mc_bars(df_s1, "rmse", "phi",
                 "Monte Carlo RMSE — Heteroskedastic DGP",
                 os.path.join(FIGURES_DIR, "mc_s1_rmse.png"))
    plot_mc_bars(df_s1, "bias", "phi",
                 "Monte Carlo Bias — Heteroskedastic DGP",
                 os.path.join(FIGURES_DIR, "mc_s1_bias.png"))
    plot_coverage(df_s1, "phi",
                  "CI Coverage — Heteroskedastic DGP",
                  os.path.join(FIGURES_DIR, "mc_s1_coverage.png"))
    all_results["scenario1"] = df_s1

    # ── Scenario 2: Autocorrelation ───────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SCENARIO 2: AR(1) Autocorrelation")
    print("=" * 70)
    rows_s2 = []
    t0 = time.time()
    for rho in RHO_VALUES:
        print(f"\n  Running ρ = {rho} ...")
        res = scenario_autocorr(rho)
        for method_res in res.values():
            rows_s2.append(method_res)
    df_s2 = pd.DataFrame(rows_s2)
    df_s2.to_csv(os.path.join(TABLES_DIR, "mc_scenario2.csv"), index=False)
    print(f"\n  Scenario 2 complete in {time.time() - t0:.1f}s")
    print(df_s2.to_string(index=False))

    plot_mc_bars(df_s2, "rmse", "rho",
                 "Monte Carlo RMSE — AR(1) DGP",
                 os.path.join(FIGURES_DIR, "mc_s2_rmse.png"))
    plot_mc_bars(df_s2, "bias", "rho",
                 "Monte Carlo Bias — AR(1) DGP",
                 os.path.join(FIGURES_DIR, "mc_s2_bias.png"))
    plot_coverage(df_s2, "rho",
                  "CI Coverage — AR(1) DGP",
                  os.path.join(FIGURES_DIR, "mc_s2_coverage.png"))
    all_results["scenario2"] = df_s2

    # ── Scenario 3: Clustered Errors ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SCENARIO 3: Clustered Errors")
    print("=" * 70)
    rows_s3 = []
    t0 = time.time()
    for tau_sq in TAU_SQ_VALUES:
        print(f"\n  Running τ² = {tau_sq} ...")
        res = scenario_clustered(tau_sq)
        for method_res in res.values():
            rows_s3.append(method_res)
    df_s3 = pd.DataFrame(rows_s3)
    df_s3.to_csv(os.path.join(TABLES_DIR, "mc_scenario3.csv"), index=False)
    print(f"\n  Scenario 3 complete in {time.time() - t0:.1f}s")
    print(df_s3.to_string(index=False))

    plot_mc_bars(df_s3, "rmse", "tau_sq",
                 "Monte Carlo RMSE — Clustered DGP",
                 os.path.join(FIGURES_DIR, "mc_s3_rmse.png"))
    plot_mc_bars(df_s3, "bias", "tau_sq",
                 "Monte Carlo Bias — Clustered DGP",
                 os.path.join(FIGURES_DIR, "mc_s3_bias.png"))
    plot_coverage(df_s3, "tau_sq",
                  "CI Coverage — Clustered DGP",
                  os.path.join(FIGURES_DIR, "mc_s3_coverage.png"))
    all_results["scenario3"] = df_s3

    return all_results


def main() -> None:
    """Entry point."""
    logger.info("Starting Monte Carlo simulations (%d reps per scenario)", N_REPS)
    t_start = time.time()
    run_all_mc()
    elapsed = time.time() - t_start
    logger.info("All Monte Carlo simulations complete in %.1f seconds.", elapsed)


if __name__ == "__main__":
    main()

