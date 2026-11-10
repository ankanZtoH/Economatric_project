#!/usr/bin/env python3
"""
data_cleaning.py — Data acquisition, cleaning, and preparation.

Handles four datasets:
    1. Boston Housing   — downloaded from UCI mirror on GitHub.
    2. Ames Housing     — expected at data/raw/ames_housing.csv (Kaggle).
    3. World Bank WDI   — synthetic panel (≈ 200 countries × 20 years).
    4. Enterprise Survey — synthetic firm-level cross-section (≈ 5 000 firms).

Each cleaning function returns a clean DataFrame **and** persists it to
``data/processed/``.

Author : Ankan Bera
Created: 2026-11-09
"""

import os
import sys
import urllib.request
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    BOSTON_CSV_URL,
    RANDOM_SEED,
    setup_logging,
)

logger = setup_logging(__name__)

# ============================================================================
# 1. Boston Housing
# ============================================================================


def load_boston_housing(
    url: str = BOSTON_CSV_URL,
    dest: Optional[os.PathLike] = None,
) -> pd.DataFrame:
    """
    Download the Boston Housing dataset from the UCI mirror and clean it.

    The dataset is fetched from *url* using :mod:`urllib` (no external
    dependencies).  Minimal cleaning is applied because the dataset is
    already well-formed; we simply lower-case column names and drop any
    duplicate rows.

    Parameters
    ----------
    url : str
        URL to the raw CSV file.
    dest : path-like, optional
        Directory to save the raw download.  Defaults to ``RAW_DATA_DIR``.

    Returns
    -------
    pd.DataFrame
        Cleaned Boston Housing data with lower-case column names.
    """
    dest = dest or RAW_DATA_DIR
    raw_path = os.path.join(dest, "boston_housing.csv")

    # Download if not cached ------------------------------------------------
    if not os.path.exists(raw_path):
        logger.info("Downloading Boston Housing CSV from %s …", url)
        try:
            urllib.request.urlretrieve(url, raw_path)
            logger.info("Saved to %s", raw_path)
        except Exception as exc:
            logger.error("Download failed: %s", exc)
            raise
    else:
        logger.info("Using cached Boston Housing CSV at %s", raw_path)

    df = pd.read_csv(raw_path)

    # Standardise column names to lower-case --------------------------------
    df.columns = [c.lower().strip() for c in df.columns]

    # Drop duplicates -------------------------------------------------------
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    logger.info(
        "Boston Housing: %d → %d rows after dedup.", n_before, len(df)
    )

    # Save cleaned copy -----------------------------------------------------
    out_path = os.path.join(PROCESSED_DATA_DIR, "boston_clean.csv")
    df.to_csv(out_path, index=False)
    logger.info("Saved cleaned Boston data to %s", out_path)

    return df


# ============================================================================
# 2. Ames Housing (user-provided from Kaggle)
# ============================================================================


def load_ames_housing(
    raw_path: Optional[os.PathLike] = None,
) -> pd.DataFrame:
    """
    Load and clean the Kaggle Ames Housing dataset.

    Expects the raw CSV at ``data/raw/ames_housing.csv``.  Cleaning steps:
        * Impute missing numeric values with column medians.
        * Impute missing categorical values with column modes.
        * Create ``log_SalePrice = log(SalePrice)``.
        * Save to ``data/processed/ames_clean.csv``.

    Parameters
    ----------
    raw_path : path-like, optional
        Path to the raw CSV.  Defaults to ``data/raw/ames_housing.csv``.

    Returns
    -------
    pd.DataFrame
        Cleaned Ames Housing data.

    Raises
    ------
    FileNotFoundError
        If the raw file is not present.
    """
    raw_path = raw_path or os.path.join(RAW_DATA_DIR, "ames_housing.csv")

    if not os.path.exists(raw_path):
        logger.warning(
            "Ames Housing file not found at %s. "
            "Please download it from Kaggle and place it there. "
            "Generating a synthetic placeholder instead.",
            raw_path,
        )
        df = _generate_synthetic_ames()
    else:
        logger.info("Loading Ames Housing from %s", raw_path)
        df = pd.read_csv(raw_path)

    # --- Imputation --------------------------------------------------------
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        n_miss = df[col].isna().sum()
        if n_miss > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.info("  Imputed %d missing in '%s' with median=%.2f",
                        n_miss, col, median_val)

    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        n_miss = df[col].isna().sum()
        if n_miss > 0:
            mode_val = df[col].mode().iloc[0]
            df[col] = df[col].fillna(mode_val)
            logger.info("  Imputed %d missing in '%s' with mode='%s'",
                        n_miss, col, mode_val)

    # --- Feature engineering -----------------------------------------------
    if "SalePrice" in df.columns:
        df["log_SalePrice"] = np.log(df["SalePrice"].clip(lower=1))
        logger.info("Created log_SalePrice.")

    # --- Save ---------------------------------------------------------------
    out_path = os.path.join(PROCESSED_DATA_DIR, "ames_clean.csv")
    df.to_csv(out_path, index=False)
    logger.info("Saved cleaned Ames data to %s  (%d rows)", out_path, len(df))

    return df


def _generate_synthetic_ames(n: int = 1460) -> pd.DataFrame:
    """
    Generate a synthetic Ames-like dataset as a placeholder.

    This is used only when the real Kaggle file is unavailable.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    df = pd.DataFrame({
        "GrLivArea": rng.integers(500, 4000, size=n),
        "LotArea": rng.integers(1500, 50000, size=n),
        "OverallQual": rng.integers(1, 11, size=n),
        "YearBuilt": rng.integers(1900, 2020, size=n),
        "TotalBsmtSF": rng.integers(0, 3000, size=n),
        "GarageCars": rng.integers(0, 5, size=n),
    })
    # SalePrice is a noisy function of the features
    df["SalePrice"] = (
        50_000
        + 50 * df["GrLivArea"]
        + 0.5 * df["LotArea"]
        + 15_000 * df["OverallQual"]
        + 200 * (df["YearBuilt"] - 1900)
        + 20 * df["TotalBsmtSF"]
        + 10_000 * df["GarageCars"]
        + rng.normal(0, 25_000, size=n)
    ).clip(lower=10_000).astype(int)

    # Sprinkle some missing values
    for col in ["TotalBsmtSF", "GarageCars"]:
        mask = rng.random(n) < 0.05
        df.loc[mask, col] = np.nan

    return df


# ============================================================================
# 3. World Bank WDI — Synthetic Panel
# ============================================================================


def create_wdi_panel(
    n_countries: int = 200,
    n_years: int = 20,
) -> pd.DataFrame:
    """
    Create a synthetic but realistic panel dataset mimicking World Bank WDI.

    .. note::
        For actual research, obtain real data from the World Bank API
        (``wbdata`` or ``wbgapi`` Python packages) or from
        https://databank.worldbank.org/.

    Variables
    ---------
    country_id   : str — ISO-like 3-letter code.
    year         : int — 2003–2022.
    gdp_pc       : float — GDP per capita (current USD).
    gdp_growth   : float — annual GDP growth (%).
    lag_gdp_pc   : float — lagged GDP per capita.
    population   : float — total population.
    log_pop      : float — log(population).
    inflation    : float — consumer price inflation (%).
    life_exp     : float — life expectancy at birth.
    investment_rate : float — gross fixed capital formation (% of GDP).

    Parameters
    ----------
    n_countries : int
        Number of synthetic countries.
    n_years : int
        Number of time periods.

    Returns
    -------
    pd.DataFrame
        Panel data sorted by (country_id, year).
    """
    rng = np.random.default_rng(RANDOM_SEED)
    logger.info("Generating synthetic WDI panel: %d countries × %d years",
                n_countries, n_years)

    records = []
    years = list(range(2003, 2003 + n_years))

    for i in range(n_countries):
        cid = f"C{i:03d}"
        # Country-level fixed effects
        base_gdp = rng.lognormal(mean=8.5, sigma=1.2)  # median ≈ $5000
        base_pop = rng.lognormal(mean=15, sigma=2.0)
        base_infl = rng.uniform(1, 12)
        base_life = rng.uniform(55, 82)
        base_inv = rng.uniform(15, 35)

        gdp_pc = base_gdp
        for t, yr in enumerate(years):
            growth = rng.normal(3.0, 3.0)  # percent
            gdp_pc_new = gdp_pc * (1 + growth / 100)
            pop = base_pop * (1 + rng.normal(0.01, 0.005)) ** t
            inflation = base_infl + rng.normal(0, 2)
            life_exp = base_life + 0.15 * t + rng.normal(0, 0.5)
            inv_rate = base_inv + rng.normal(0, 3)

            records.append({
                "country_id": cid,
                "year": yr,
                "gdp_pc": round(gdp_pc_new, 2),
                "gdp_growth": round(growth, 4),
                "lag_gdp_pc": round(gdp_pc, 2),
                "population": int(pop),
                "log_pop": round(np.log(pop), 4),
                "inflation": round(inflation, 2),
                "life_exp": round(life_exp, 2),
                "investment_rate": round(inv_rate, 2),
            })
            gdp_pc = gdp_pc_new

    df = pd.DataFrame(records).sort_values(["country_id", "year"]).reset_index(drop=True)

    out_path = os.path.join(PROCESSED_DATA_DIR, "wdi_panel.csv")
    df.to_csv(out_path, index=False)
    logger.info("Saved synthetic WDI panel to %s  (%d rows)", out_path, len(df))

    return df


# ============================================================================
# 4. Enterprise Survey — Synthetic Firm-Level Data
# ============================================================================


def create_enterprise_survey(
    n_firms: int = 5000,
    n_countries: int = 30,
) -> pd.DataFrame:
    """
    Create a synthetic firm-level dataset mimicking World Bank Enterprise
    Surveys.

    Variables
    ---------
    firm_id       : int — unique identifier.
    country       : str — country label.
    sector        : str — industry sector.
    firm_size     : str — "small", "medium", or "large".
    firm_size_dummy : int — 0 = small, 1 = medium, 2 = large.
    log_revenue   : float — log of annual revenue.
    log_capital   : float — log of total fixed assets.
    log_labor     : float — log of number of employees.

    Parameters
    ----------
    n_firms : int
        Total number of firms to generate.
    n_countries : int
        Number of countries to distribute firms across.

    Returns
    -------
    pd.DataFrame
    """
    rng = np.random.default_rng(RANDOM_SEED)
    logger.info("Generating synthetic Enterprise Survey: %d firms, %d countries",
                n_firms, n_countries)

    countries = [f"Country_{i:02d}" for i in range(n_countries)]
    sectors = ["Manufacturing", "Retail", "Services", "Construction", "Agriculture"]
    size_labels = ["small", "medium", "large"]
    size_probs = [0.50, 0.35, 0.15]

    records = []
    for fid in range(n_firms):
        country = rng.choice(countries)
        sector = rng.choice(sectors)
        size_idx = rng.choice(len(size_labels), p=size_probs)
        firm_size = size_labels[size_idx]

        # Larger firms have more capital and labour
        size_shift = [0, 1.0, 2.5][size_idx]
        log_capital = rng.normal(10 + size_shift, 1.5)
        log_labor = rng.normal(3 + size_shift * 0.6, 0.8)

        # Cobb-Douglas-like production + noise
        log_revenue = (
            2.0
            + 0.35 * log_capital
            + 0.55 * log_labor
            + 0.1 * size_idx
            + rng.normal(0, 0.6)
        )

        records.append({
            "firm_id": fid,
            "country": country,
            "sector": sector,
            "firm_size": firm_size,
            "firm_size_dummy": size_idx,
            "log_revenue": round(log_revenue, 4),
            "log_capital": round(log_capital, 4),
            "log_labor": round(log_labor, 4),
        })

    df = pd.DataFrame(records)

    out_path = os.path.join(PROCESSED_DATA_DIR, "enterprise_clean.csv")
    df.to_csv(out_path, index=False)
    logger.info("Saved synthetic Enterprise Survey to %s  (%d rows)",
                out_path, len(df))

    return df


# ============================================================================
# Summary utility
# ============================================================================


def _print_summary(name: str, df: pd.DataFrame) -> None:
    """Print basic summary statistics for a dataset."""
    print(f"\n{'=' * 70}")
    print(f"  Dataset: {name}   ({df.shape[0]} rows × {df.shape[1]} cols)")
    print(f"{'=' * 70}")
    numeric = df.select_dtypes(include=[np.number])
    if not numeric.empty:
        print(numeric.describe().round(3).to_string())
    print()


# ============================================================================
# Main entry point
# ============================================================================


def main() -> None:
    """Run all data acquisition and cleaning steps."""
    logger.info("=" * 60)
    logger.info("Starting data cleaning pipeline")
    logger.info("=" * 60)

    boston = load_boston_housing()
    _print_summary("Boston Housing", boston)

    ames = load_ames_housing()
    _print_summary("Ames Housing", ames)

    wdi = create_wdi_panel()
    _print_summary("WDI Panel (synthetic)", wdi)

    enterprise = create_enterprise_survey()
    _print_summary("Enterprise Survey (synthetic)", enterprise)

    logger.info("Data cleaning pipeline complete.")


if __name__ == "__main__":
    main()
