#!/usr/bin/env python3
"""
config.py — Central configuration for the GLS/WLS Econometrics Project.

Defines all paths, simulation parameters, plotting aesthetics, and logging
setup used across every module in the project.

Author : Ankan Bera
Created: 2026-11-09
"""

import os
import logging
from pathlib import Path

# ============================================================================
# 1. Project Paths
# ============================================================================

#: Absolute path to the project root directory.
PROJECT_ROOT: Path = Path("/home/ankanbera/Documents/Economatrics_project")

#: Directory containing raw data files (user-provided or downloaded).
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"

#: Directory for all generated outputs.
OUTPUT_DIR: Path = PROJECT_ROOT / "output"
FIGURES_DIR: Path = OUTPUT_DIR / "figures"
TABLES_DIR: Path = OUTPUT_DIR / "tables"

#: Directory containing the Python scripts themselves.
SCRIPTS_DIR: Path = PROJECT_ROOT / "scripts" / "python"

# Create directories if they do not exist
for _dir in [RAW_DATA_DIR, PROCESSED_DATA_DIR, FIGURES_DIR, TABLES_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 2. Reproducibility
# ============================================================================

#: Global random seed for NumPy, SciPy, and any stochastic procedure.
RANDOM_SEED: int = 42

# ============================================================================
# 3. Monte Carlo Simulation Parameters
# ============================================================================

#: Number of Monte Carlo replications per scenario.
N_REPS: int = 5000

#: Sample sizes to evaluate in simulation studies.
SAMPLE_SIZES: list[int] = [50, 100, 500]

#: Heteroskedasticity exponent values (Scenario 1).
PHI_VALUES: list[float] = [0.0, 0.5, 1.0, 1.5]

#: AR(1) autocorrelation parameters (Scenario 2).
RHO_VALUES: list[float] = [0.0, 0.3, 0.6, 0.9]

#: Cluster-effect variance parameters (Scenario 3).
TAU_SQ_VALUES: list[float] = [0.0, 0.5, 1.0, 2.0]

#: Number of clusters and observations per cluster (Scenario 3).
N_CLUSTERS: int = 30
OBS_PER_CLUSTER: int = 20

# ============================================================================
# 4. Plotting Aesthetics
# ============================================================================

#: Colour palette — a colour-blind-friendly qualitative palette.
COLOR_PALETTE: list[str] = [
    "#4C72B0",  # Steel blue   — OLS
    "#DD8452",  # Sandy orange — WLS
    "#55A868",  # Sage green   — GLS / FGLS
    "#C44E52",  # Brick red    — Robust
    "#8172B3",  # Muted purple — Cluster-robust
    "#937860",  # Brown        — Newey-West
    "#DA8BC3",  # Pink         — auxiliary
    "#8C8C8C",  # Grey         — reference
    "#CCB974",  # Gold         — auxiliary
    "#64B5CD",  # Light blue   — auxiliary
]

#: Default figure size (width, height) in inches.
FIGURE_SIZE: tuple[float, float] = (10.0, 6.0)

#: Resolution for saved figures.
DPI: int = 300

#: Matplotlib style context to use.
MPL_STYLE: str = "seaborn-v0_8-whitegrid"

# ============================================================================
# 5. Logging Setup
# ============================================================================

LOG_FORMAT: str = "%(asctime)s | %(name)-24s | %(levelname)-8s | %(message)s"
LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
LOG_LEVEL: int = logging.INFO


def setup_logging(name: str = "econometrics") -> logging.Logger:
    """
    Configure and return a project-wide logger.

    Parameters
    ----------
    name : str
        Logger name (typically the module's ``__name__``).

    Returns
    -------
    logging.Logger
        Configured logger instance with stream handler.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))
        logger.addHandler(handler)
    logger.setLevel(LOG_LEVEL)
    return logger


# ============================================================================
# 6. OLS Model Specifications (formula strings)
# ============================================================================

MODEL_SPECS: dict[str, str] = {
    "boston": "medv ~ crim + zn + indus + rm + age + tax + ptratio + lstat",
    "ames": "log_SalePrice ~ GrLivArea + LotArea + OverallQual + YearBuilt + TotalBsmtSF + GarageCars",
    "wdi": "gdp_growth ~ lag_gdp_pc + investment_rate + inflation + log_pop",
    "enterprise": "log_revenue ~ log_capital + log_labor + firm_size_dummy",
}

# ============================================================================
# 7. Dataset URLs
# ============================================================================

BOSTON_CSV_URL: str = (
    "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
)


if __name__ == "__main__":
    log = setup_logging(__name__)
    log.info("Project root : %s", PROJECT_ROOT)
    log.info("Raw data dir : %s", RAW_DATA_DIR)
    log.info("Processed dir: %s", PROCESSED_DATA_DIR)
    log.info("Figures dir  : %s", FIGURES_DIR)
    log.info("Tables dir   : %s", TABLES_DIR)
    log.info("Random seed  : %d", RANDOM_SEED)
    log.info("MC reps      : %d", N_REPS)
    log.info("Sample sizes : %s", SAMPLE_SIZES)
