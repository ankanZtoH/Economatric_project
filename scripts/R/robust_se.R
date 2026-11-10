# ===========================================================================
# robust_se.R — Robust and cluster-robust standard error computations.
#
# Computes HC0–HC3, cluster-robust, and Newey-West HAC standard errors
# for all OLS models using the sandwich and lmtest packages.
#
# Author : Ankan Bera
# Created: 2026-11-09
# ===========================================================================

source(file.path(dirname(sys.frame(1)$ofile %||% "."), "config.R"))

# ============================================================================
# Core Function: Compute all SE types
# ============================================================================

compute_robust_se <- function(model, cluster_var = NULL, maxlags = NULL) {
  #' Compute multiple types of standard errors for an OLS model.
  #'
  #' @param model lm object

  #' @param cluster_var vector of cluster identifiers (optional)
  #' @param maxlags integer, max lags for Newey-West (optional)
  #' @return data.frame with SE comparison
  
  vars <- names(coef(model))
  result <- data.frame(
    Variable = vars,
    Coef     = round(coef(model), 6),
    OLS_SE   = round(summary(model)$coefficients[, 2], 6),
    stringsAsFactors = FALSE
  )
  
  # HC0
  tryCatch({
    vc <- sandwich::vcovHC(model, type = "HC0")
    result$HC0 <- round(sqrt(diag(vc)), 6)
  }, error = function(e) { result$HC0 <<- NA })
  
  # HC1
  tryCatch({
    vc <- sandwich::vcovHC(model, type = "HC1")
    result$HC1 <- round(sqrt(diag(vc)), 6)
  }, error = function(e) { result$HC1 <<- NA })
  
  # HC3
  tryCatch({
    vc <- sandwich::vcovHC(model, type = "HC3")
    result$HC3 <- round(sqrt(diag(vc)), 6)
  }, error = function(e) { result$HC3 <<- NA })
  
  # Cluster-robust
  if (!is.null(cluster_var)) {
    tryCatch({
      vc <- sandwich::vcovCL(model, cluster = cluster_var)
      result$Cluster_SE <- round(sqrt(diag(vc)), 6)
    }, error = function(e) { result$Cluster_SE <<- NA })
  } else {
    result$Cluster_SE <- NA
  }
  
  # Newey-West HAC
  if (is.null(maxlags)) maxlags <- floor(nobs(model)^(1/3))
  tryCatch({
    vc <- sandwich::NeweyWest(model, lag = maxlags)
    result$NW_SE <- round(sqrt(diag(vc)), 6)
  }, error = function(e) { result$NW_SE <<- NA })
  
  return(result)
}

# ============================================================================
# Print with lmtest::coeftest
# ============================================================================

print_coeftest <- function(model, vcov_type, label, ...) {
  cat(sprintf("\n  %s:\n", label))
  tryCatch({
    vc <- do.call(vcov_type, c(list(model), list(...)))
    print(lmtest::coeftest(model, vcov. = vc))
  }, error = function(e) {
    cat("    (failed: ", e$message, ")\n")
  })
}

# ============================================================================
# Dataset-Specific
# ============================================================================

robust_se_boston <- function() {
  message("\n── Robust SEs: Boston ─────────────────────────────────────")
  df <- read.csv(file.path(PROCESSED_DIR, "boston_clean.csv"))
  model <- lm(medv ~ crim + zn + indus + rm + age + tax + ptratio + lstat, data = df)
  
  se_table <- compute_robust_se(model)
  
  # Print coeftest results
  print_coeftest(model, sandwich::vcovHC, "HC3 Robust", type = "HC3")
  
  return(se_table)
}

robust_se_ames <- function() {
  message("\n── Robust SEs: Ames ───────────────────────────────────────")
  df <- read.csv(file.path(PROCESSED_DIR, "ames_clean.csv"))
  model <- lm(log_SalePrice ~ GrLivArea + LotArea + OverallQual +
                 YearBuilt + TotalBsmtSF + GarageCars, data = df)
  
  se_table <- compute_robust_se(model)
  print_coeftest(model, sandwich::vcovHC, "HC3 Robust", type = "HC3")
  
  return(se_table)
}

robust_se_wdi <- function() {
  message("\n── Robust SEs: WDI ────────────────────────────────────────")
  df <- read.csv(file.path(PROCESSED_DIR, "wdi_panel.csv"))
  df <- na.omit(df[, c("country_id", "year", "gdp_growth", "lag_gdp_pc",
                        "investment_rate", "inflation", "log_pop")])
  
  model <- lm(gdp_growth ~ lag_gdp_pc + investment_rate + inflation + log_pop,
               data = df)
  
  se_table <- compute_robust_se(model, cluster_var = df$country_id, maxlags = 5)
  
  print_coeftest(model, sandwich::vcovHC, "HC3 Robust", type = "HC3")
  
  cat("\n  Cluster-robust (by country):\n")
  tryCatch({
    vc_cl <- sandwich::vcovCL(model, cluster = df$country_id)
    print(lmtest::coeftest(model, vcov. = vc_cl))
  }, error = function(e) cat("    Failed: ", e$message, "\n"))
  
  cat("\n  Newey-West HAC (5 lags):\n")
  tryCatch({
    vc_nw <- sandwich::NeweyWest(model, lag = 5)
    print(lmtest::coeftest(model, vcov. = vc_nw))
  }, error = function(e) cat("    Failed: ", e$message, "\n"))
  
  return(se_table)
}

robust_se_enterprise <- function() {
  message("\n── Robust SEs: Enterprise ─────────────────────────────────")
  df <- read.csv(file.path(PROCESSED_DIR, "enterprise_clean.csv"))
  
  model <- lm(log_revenue ~ log_capital + log_labor + firm_size_dummy, data = df)
  
  se_table <- compute_robust_se(model, cluster_var = df$country)
  
  print_coeftest(model, sandwich::vcovHC, "HC3 Robust", type = "HC3")
  
  cat("\n  Cluster-robust (by country):\n")
  tryCatch({
    vc_cl <- sandwich::vcovCL(model, cluster = df$country)
    print(lmtest::coeftest(model, vcov. = vc_cl))
  }, error = function(e) cat("    Failed: ", e$message, "\n"))
  
  return(se_table)
}

# ============================================================================
# Run All
# ============================================================================

main <- function() {
  message("Computing robust standard errors for all datasets (R)")
  
  all_tables <- list()
  
  for (info in list(
    list(name = "boston",     func = robust_se_boston),
    list(name = "ames",      func = robust_se_ames),
    list(name = "wdi",       func = robust_se_wdi),
    list(name = "enterprise", func = robust_se_enterprise)
  )) {
    tryCatch({
      se_tab <- info$func()
      se_tab$Dataset <- info$name
      all_tables[[info$name]] <- se_tab
      
      cat("\n  SE Comparison (", info$name, "):\n")
      print(se_tab)
      
      # Ratio table
      se_cols <- c("HC0", "HC1", "HC3", "Cluster_SE", "NW_SE")
      ratio <- se_tab[, c("Variable", "OLS_SE")]
      for (col in se_cols) {
        if (col %in% names(se_tab) && !all(is.na(se_tab[[col]]))) {
          ratio[[paste0(col, "/OLS")]] <- round(se_tab[[col]] / se_tab$OLS_SE, 4)
        }
      }
      cat("\n  SE Ratios (relative to OLS):\n")
      print(ratio)
      
    }, error = function(e) {
      message("  Failed for ", info$name, ": ", e$message)
    })
  }
  
  # Combine and save
  if (length(all_tables) > 0) {
    master <- do.call(rbind, all_tables)
    rownames(master) <- NULL
    save_table(master, "robust_se_comparison_R.csv")
    message("  Saved robust SE comparison table.")
  }
}

if (sys.nframe() == 0L || !interactive()) {
  main()
}

