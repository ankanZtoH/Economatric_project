# ===========================================================================
# config.R — Central configuration for the GLS/WLS Econometrics Project.
#
# Defines paths, simulation parameters, plotting theme, and library loading.
# Source this file at the top of every other R script.
#
# Author : Ankan Bera
# Created: 2026-11-09
# ===========================================================================

# ── 1. Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT  <- "/home/ankanbera/Documents/Economatrics_project"
DATA_DIR      <- file.path(PROJECT_ROOT, "data")
RAW_DATA_DIR  <- file.path(DATA_DIR, "raw")
PROCESSED_DIR <- file.path(DATA_DIR, "processed")
OUTPUT_DIR    <- file.path(PROJECT_ROOT, "output")
FIGURES_DIR   <- file.path(OUTPUT_DIR, "figures")
TABLES_DIR    <- file.path(OUTPUT_DIR, "tables")

# Create directories if needed
for (d in c(RAW_DATA_DIR, PROCESSED_DIR, FIGURES_DIR, TABLES_DIR)) {
  dir.create(d, recursive = TRUE, showWarnings = FALSE)
}

# ── 2. Reproducibility ───────────────────────────────────────────────────
SEED <- 42
set.seed(SEED)

# ── 3. Monte Carlo Parameters ────────────────────────────────────────────
MC_N_REPS       <- 5000L
MC_SAMPLE_SIZES <- c(50L, 100L, 500L)
MC_DEFAULT_N    <- 100L
TRUE_BETA0      <- 1.0
TRUE_BETA1      <- 2.0

PHI_VALUES  <- c(0, 0.5, 1.0, 1.5)
RHO_VALUES  <- c(0, 0.3, 0.6, 0.9)
TAU2_VALUES <- c(0, 0.5, 1.0, 2.0)
N_CLUSTERS  <- 30L
CLUSTER_SIZE <- 20L

# ── 4. Packages ──────────────────────────────────────────────────────────
required_pkgs <- c("MASS", "lmtest", "sandwich", "nlme", "car",
                   "ggplot2", "dplyr", "knitr")
optional_pkgs <- c("plm", "stargazer")

for (pkg in required_pkgs) {
  suppressPackageStartupMessages(library(pkg, character.only = TRUE))
}
for (pkg in optional_pkgs) {
  tryCatch(
    suppressPackageStartupMessages(library(pkg, character.only = TRUE)),
    error = function(e) message("  Optional package '", pkg, "' not available.")
  )
}

# ── 5. ggplot2 Theme ─────────────────────────────────────────────────────
COLOR_PALETTE <- c(
  OLS      = "#4C72B0",
  WLS      = "#DD8452",
  GLS      = "#55A868",
  FGLS     = "#8172B3",
  Robust   = "#C44E52",
  Cluster  = "#DA8BC3",
  HAC      = "#937860"
)

theme_project <- theme_bw(base_size = 12) +
  theme(
    plot.title      = element_text(face = "bold", hjust = 0.5),
    legend.position = "bottom",
    panel.grid.minor = element_blank()
  )
theme_set(theme_project)

# ── 6. Helper: Save Plot ─────────────────────────────────────────────────
save_plot <- function(plot, filename, width = 10, height = 7, dpi = 300) {
  path <- file.path(FIGURES_DIR, filename)
  ggsave(path, plot = plot, width = width, height = height, dpi = dpi)
  message("  Saved plot: ", path)
}

# ── 7. Helper: Save Table ────────────────────────────────────────────────
save_table <- function(df, filename) {
  path <- file.path(TABLES_DIR, filename)
  write.csv(df, path, row.names = FALSE)
  message("  Saved table: ", path)
}

message("Config loaded. Project root: ", PROJECT_ROOT)

