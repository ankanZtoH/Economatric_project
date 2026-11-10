# ===========================================================================
# data_cleaning.R — Data loading, cleaning, and synthetic data generation.
#
# Handles four datasets:
#   1. Boston Housing (from MASS package)
#   2. Ames Housing   (from CSV or synthetic fallback)
#   3. WDI Panel      (synthetic)
#   4. Enterprise Survey (synthetic)
#
# Author : Ankan Bera
# Created: 2026-11-09
# ===========================================================================

source(file.path(dirname(sys.frame(1)$ofile %||% "."), "config.R"))

# ============================================================================
# 1. Boston Housing
# ============================================================================

load_boston <- function() {
  message("\n── Loading Boston Housing ──────────────────────────────────")
  data(Boston, package = "MASS")
  df <- Boston
  
  # Check for duplicates
  n_before <- nrow(df)
  df <- unique(df)
  message("  Boston: ", n_before, " → ", nrow(df), " rows after dedup")
  
  # Save
  out_path <- file.path(PROCESSED_DIR, "boston_clean.csv")
  write.csv(df, out_path, row.names = FALSE)
  message("  Saved to: ", out_path)
  
  cat("\n  Summary:\n")
  print(summary(df[, c("medv", "crim", "rm", "tax", "lstat")]))
  
  return(df)
}

# ============================================================================
# 2. Ames Housing
# ============================================================================

load_ames <- function() {
  message("\n── Loading Ames Housing ────────────────────────────────────")
  raw_path <- file.path(RAW_DATA_DIR, "ames_housing.csv")
  
  if (file.exists(raw_path)) {
    df <- read.csv(raw_path, stringsAsFactors = FALSE)
    message("  Loaded from: ", raw_path)
  } else {
    message("  Raw file not found. Generating synthetic Ames data.")
    df <- generate_synthetic_ames()
  }
  
  # Impute numeric NAs with medians
  numeric_cols <- names(df)[sapply(df, is.numeric)]
  for (col in numeric_cols) {
    n_miss <- sum(is.na(df[[col]]))
    if (n_miss > 0) {
      med_val <- median(df[[col]], na.rm = TRUE)
      df[[col]][is.na(df[[col]])] <- med_val
      message("  Imputed ", n_miss, " missing in '", col, "' with median=", round(med_val, 2))
    }
  }
  
  # Create log_SalePrice
  if ("SalePrice" %in% names(df)) {
    df$log_SalePrice <- log(pmax(df$SalePrice, 1))
    message("  Created log_SalePrice")
  }
  
  out_path <- file.path(PROCESSED_DIR, "ames_clean.csv")
  write.csv(df, out_path, row.names = FALSE)
  message("  Saved to: ", out_path, " (", nrow(df), " rows)")
  
  return(df)
}

generate_synthetic_ames <- function(n = 1460) {
  set.seed(SEED)
  df <- data.frame(
    GrLivArea   = sample(500:4000, n, replace = TRUE),
    LotArea     = sample(1500:50000, n, replace = TRUE),
    OverallQual = sample(1:10, n, replace = TRUE),
    YearBuilt   = sample(1900:2020, n, replace = TRUE),
    TotalBsmtSF = sample(0:3000, n, replace = TRUE),
    GarageCars  = sample(0:4, n, replace = TRUE)
  )
  
  df$SalePrice <- pmax(
    50000 +
      50 * df$GrLivArea +
      0.5 * df$LotArea +
      15000 * df$OverallQual +
      200 * (df$YearBuilt - 1900) +
      20 * df$TotalBsmtSF +
      10000 * df$GarageCars +
      rnorm(n, 0, 25000),
    10000
  )
  df$SalePrice <- as.integer(df$SalePrice)
  
  # Sprinkle NAs
  df$TotalBsmtSF[sample(n, round(n * 0.05))] <- NA
  df$GarageCars[sample(n, round(n * 0.05))]  <- NA
  
  return(df)
}

# ============================================================================
# 3. WDI Panel (Synthetic)
# ============================================================================

create_wdi_panel <- function(n_countries = 200, n_years = 20) {
  message("\n── Generating Synthetic WDI Panel ──────────────────────────")
  set.seed(SEED)
  
  years <- 2003:(2003 + n_years - 1)
  records <- list()
  
  for (i in 1:n_countries) {
    cid <- sprintf("C%03d", i)
    base_gdp  <- exp(rnorm(1, 8.5, 1.2))
    base_pop  <- exp(rnorm(1, 15, 2.0))
    base_infl <- runif(1, 1, 12)
    base_life <- runif(1, 55, 82)
    base_inv  <- runif(1, 15, 35)
    
    gdp_pc <- base_gdp
    for (t in seq_along(years)) {
      growth    <- rnorm(1, 3.0, 3.0)
      gdp_new   <- gdp_pc * (1 + growth / 100)
      pop       <- base_pop * (1 + rnorm(1, 0.01, 0.005))^(t - 1)
      inflation <- base_infl + rnorm(1, 0, 2)
      life_exp  <- base_life + 0.15 * (t - 1) + rnorm(1, 0, 0.5)
      inv_rate  <- base_inv + rnorm(1, 0, 3)
      
      records[[length(records) + 1]] <- data.frame(
        country_id      = cid,
        year            = years[t],
        gdp_pc          = round(gdp_new, 2),
        gdp_growth      = round(growth, 4),
        lag_gdp_pc      = round(gdp_pc, 2),
        population      = as.integer(pop),
        log_pop         = round(log(pop), 4),
        inflation       = round(inflation, 2),
        life_exp        = round(life_exp, 2),
        investment_rate = round(inv_rate, 2),
        stringsAsFactors = FALSE
      )
      gdp_pc <- gdp_new
    }
  }
  
  df <- do.call(rbind, records)
  df <- df[order(df$country_id, df$year), ]
  rownames(df) <- NULL
  
  out_path <- file.path(PROCESSED_DIR, "wdi_panel.csv")
  write.csv(df, out_path, row.names = FALSE)
  message("  Saved to: ", out_path, " (", nrow(df), " rows)")
  
  cat("\n  Summary:\n")
  print(summary(df[, c("gdp_pc", "gdp_growth", "inflation", "life_exp")]))
  
  return(df)
}

# ============================================================================
# 4. Enterprise Survey (Synthetic)
# ============================================================================

create_enterprise_survey <- function(n_firms = 5000, n_countries = 30) {
  message("\n── Generating Synthetic Enterprise Survey ──────────────────")
  set.seed(SEED)
  
  countries <- sprintf("Country_%02d", 1:n_countries)
  sectors   <- c("Manufacturing", "Retail", "Services", "Construction", "Agriculture")
  sizes     <- c("small", "medium", "large")
  size_probs <- c(0.50, 0.35, 0.15)
  
  country_vec  <- sample(countries, n_firms, replace = TRUE)
  sector_vec   <- sample(sectors, n_firms, replace = TRUE)
  size_idx     <- sample(1:3, n_firms, replace = TRUE, prob = size_probs)
  size_vec     <- sizes[size_idx]
  size_dummy   <- size_idx - 1  # 0, 1, 2
  
  size_shift <- c(0, 1.0, 2.5)[size_idx]
  log_capital <- rnorm(n_firms, 10 + size_shift, 1.5)
  log_labor   <- rnorm(n_firms, 3 + size_shift * 0.6, 0.8)
  
  # Cobb-Douglas + noise
  log_revenue <- 2.0 + 0.35 * log_capital + 0.55 * log_labor +
    0.1 * (size_idx - 1) + rnorm(n_firms, 0, 0.6)
  
  df <- data.frame(
    firm_id         = 1:n_firms,
    country         = country_vec,
    sector          = sector_vec,
    firm_size       = size_vec,
    firm_size_dummy = size_dummy,
    log_revenue     = round(log_revenue, 4),
    log_capital     = round(log_capital, 4),
    log_labor       = round(log_labor, 4),
    stringsAsFactors = FALSE
  )
  
  out_path <- file.path(PROCESSED_DIR, "enterprise_clean.csv")
  write.csv(df, out_path, row.names = FALSE)
  message("  Saved to: ", out_path, " (", nrow(df), " rows)")
  
  return(df)
}

# ============================================================================
# Main
# ============================================================================

main <- function() {
  message("\n", strrep("=", 60))
  message("  DATA CLEANING PIPELINE (R)")
  message(strrep("=", 60))
  
  boston     <- load_boston()
  ames      <- load_ames()
  wdi       <- create_wdi_panel()
  enterprise <- create_enterprise_survey()
  
  message("\n  Data cleaning complete.")
  invisible(list(boston = boston, ames = ames, wdi = wdi,
                 enterprise = enterprise))
}

# Run if executed directly
if (sys.nframe() == 0L || !interactive()) {
  main()
}

