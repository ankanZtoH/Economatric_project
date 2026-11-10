#!/usr/bin/env python3
"""
run_all.py — Master runner script for the full GLS/WLS analysis pipeline.

Orchestrates all analysis steps in order:
  1. Data cleaning & preparation
  2. OLS estimation & diagnostics
  3. Heteroskedasticity tests
  4. WLS/GLS estimation & comparison
  5. Robust standard errors
  6. Monte Carlo simulations

Author : Ankan Bera
Created: 2026-11-09
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import setup_logging, PROJECT_ROOT

logger = setup_logging("run_all")


def run_step(step_name: str, module_name: str) -> float:
    """
    Import and run a module's main() function, timing execution.

    Parameters
    ----------
    step_name : str
        Human-readable name for the step.
    module_name : str
        Python module name to import and run.

    Returns
    -------
    float
        Elapsed time in seconds.
    """
    print(f"\n{'#' * 70}")
    print(f"  Step: {step_name}")
    print(f"{'#' * 70}\n")

    t0 = time.time()
    try:
        module = __import__(module_name)
        if hasattr(module, "main"):
            module.main()
        else:
            logger.warning("Module %s has no main() function.", module_name)
    except Exception as exc:
        logger.error("Step '%s' failed: %s", step_name, exc)
        import traceback
        traceback.print_exc()

    elapsed = time.time() - t0
    logger.info("Step '%s' completed in %.1f seconds.", step_name, elapsed)
    return elapsed


def main() -> None:
    """Run the complete analysis pipeline."""
    print("=" * 70)
    print("  GLS/WLS Econometrics Project — Full Analysis Pipeline")
    print(f"  Project root: {PROJECT_ROOT}")
    print("=" * 70)

    total_start = time.time()
    timings = {}

    # Step 1: Data Cleaning
    timings["Data Cleaning"] = run_step(
        "Data Cleaning & Preparation", "data_cleaning")

    # Step 2: OLS Analysis
    timings["OLS Analysis"] = run_step(
        "OLS Estimation & Diagnostics", "ols_analysis")

    # Step 3: Heteroskedasticity Tests
    timings["Heterosk Tests"] = run_step(
        "Heteroskedasticity Tests", "heteroskedasticity_tests")

    # Step 4: WLS/GLS Analysis
    timings["WLS/GLS"] = run_step(
        "WLS/GLS Estimation & Comparison", "wls_gls_analysis")

    # Step 5: Robust SEs
    timings["Robust SEs"] = run_step(
        "Robust Standard Errors", "robust_se")

    # Step 6: Monte Carlo Simulations
    print("\n" + "!" * 70)
    print("  NOTE: Monte Carlo simulations may take several minutes.")
    print("!" * 70)
    timings["Monte Carlo"] = run_step(
        "Monte Carlo Simulations", "monte_carlo_simulation")

    # Summary
    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 70}")
    print("  Pipeline Complete — Timing Summary")
    print(f"{'=' * 70}")
    for step, t in timings.items():
        print(f"  {step:30s} : {t:8.1f}s")
    print(f"  {'─' * 42}")
    print(f"  {'TOTAL':30s} : {total_elapsed:8.1f}s")
    print(f"{'=' * 70}")

    # List outputs
    output_dir = os.path.join(PROJECT_ROOT, "output")
    print(f"\n  Output files in {output_dir}:")
    for root, dirs, files in os.walk(output_dir):
        for f in sorted(files):
            if not f.startswith("."):
                rel = os.path.relpath(os.path.join(root, f), PROJECT_ROOT)
                print(f"    {rel}")

    print(f"\n  All done! Total time: {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()

