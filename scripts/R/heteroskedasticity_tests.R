# ===========================================================================
# heteroskedasticity_tests.R — Diagnostic tests for heteroskedasticity.
#
# Applies Breusch-Pagan, NCV, and Goldfeld-Quandt tests to all OLS models.
# Generates diagnostic plots and summary table.
#
# Author : Ankan Bera
# Created: 2026-11-09
# ===========================================================================

source(file.path(dirname(sys.frame(1)$ofile %||% "."), "config.R"))

# ============================================================================
# Test Functions
# ============================================================================

run_tests_for_model <- function(model, dataset_name, data, gq_order_var = NULL) {
  message("\n── Heteroskedasticity Tests: ", dataset_name, " ──────────────")
  
  results <- list()
  
  # 1. Breusch-Pagan
  bp <- lmtest::bptest(model)
  results[[length(results) + 1]] <- data.frame(
    Dataset    = dataset_name,
    Test       = "Breusch-Pagan",
    Statistic  = round(bp$statistic, 4),
    df         = bp$parameter,
    p_value    = round(bp$p.value, 6),
    Reject_5pct = ifelse(bp$p.value < 0.05, "Yes", "No"),
    stringsAsFactors = FALSE
  )
  cat("  BP test: stat =", round(bp$statistic, 4),
      ", p =", round(bp$p.value, 6), "\n")
  
  # 2. NCV Test (Score test from car)
  tryCatch({
    ncv <- car::ncvTest(model)
    results[[length(results) + 1]] <- data.frame(
      Dataset    = dataset_name,
      Test       = "NCV (Score)",
      Statistic  = round(ncv$ChiSquare, 4),
      df         = ncv$Df,
      p_value    = round(ncv$p, 6),
      Reject_5pct = ifelse(ncv$p < 0.05, "Yes", "No"),
      stringsAsFactors = FALSE
    )
    cat("  NCV test: χ² =", round(ncv$ChiSquare, 4),
        ", p =", round(ncv$p, 6), "\n")
  }, error = function(e) {
    message("  NCV test failed: ", e$message)
  })
  
  # 3. Goldfeld-Quandt
  if (!is.null(gq_order_var) && gq_order_var %in% names(data)) {
    tryCatch({
      order_idx <- order(data[[gq_order_var]])
      data_sorted <- data[order_idx, ]
      model_sorted <- update(model, data = data_sorted)
      gq <- lmtest::gqtest(model_sorted, order.by = data_sorted[[gq_order_var]])
      results[[length(results) + 1]] <- data.frame(
        Dataset    = dataset_name,
        Test       = paste0("Goldfeld-Quandt (", gq_order_var, ")"),
        Statistic  = round(gq$statistic, 4),
        df         = paste(gq$parameter, collapse = ","),
        p_value    = round(gq$p.value, 6),
        Reject_5pct = ifelse(gq$p.value < 0.05, "Yes", "No"),
        stringsAsFactors = FALSE
      )
      cat("  GQ test: F =", round(gq$statistic, 4),
          ", p =", round(gq$p.value, 6), "\n")
    }, error = function(e) {
      message("  GQ test failed: ", e$message)
    })
  }
  
  # Diagnostic plot: squared residuals vs fitted
  df_plot <- data.frame(
    fitted   = fitted(model),
    resid_sq = residuals(model)^2
  )
  p <- ggplot(df_plot, aes(x = fitted, y = resid_sq)) +
    geom_point(alpha = 0.4, size = 1.5, color = COLOR_PALETTE["Robust"]) +
    geom_smooth(method = "loess", se = FALSE, color = "red") +
    labs(
      title = paste("Squared Residuals vs Fitted —", dataset_name),
      x = "Fitted Values", y = "Squared Residuals"
    )
  save_plot(p, paste0("resid_sq_vs_fitted_", dataset_name, "_R.png"))
  
  return(do.call(rbind, results))
}

# ============================================================================
# Run All Tests
# ============================================================================

run_all_tests <- function() {
  all_results <- list()
  
  # Boston
  tryCatch({
    df <- read.csv(file.path(PROCESSED_DIR, "boston_clean.csv"))
    model <- lm(medv ~ crim + zn + indus + rm + age + tax + ptratio + lstat, data = df)
    all_results[["boston"]] <- run_tests_for_model(model, "Boston", df, "rm")
  }, error = function(e) message("  Boston tests failed: ", e$message))
  
  # Ames
  tryCatch({
    df <- read.csv(file.path(PROCESSED_DIR, "ames_clean.csv"))
    model <- lm(log_SalePrice ~ GrLivArea + LotArea + OverallQual +
                   YearBuilt + TotalBsmtSF + GarageCars, data = df)
    all_results[["ames"]] <- run_tests_for_model(model, "Ames", df, "GrLivArea")
  }, error = function(e) message("  Ames tests failed: ", e$message))
  
  # WDI
  tryCatch({
    df <- read.csv(file.path(PROCESSED_DIR, "wdi_panel.csv"))
    model <- lm(gdp_growth ~ lag_gdp_pc + investment_rate + inflation + log_pop, data = df)
    all_results[["wdi"]] <- run_tests_for_model(model, "WDI", df, "lag_gdp_pc")
  }, error = function(e) message("  WDI tests failed: ", e$message))
  
  # Enterprise
  tryCatch({
    df <- read.csv(file.path(PROCESSED_DIR, "enterprise_clean.csv"))
    model <- lm(log_revenue ~ log_capital + log_labor + firm_size_dummy, data = df)
    all_results[["enterprise"]] <- run_tests_for_model(model, "Enterprise", df, "log_capital")
  }, error = function(e) message("  Enterprise tests failed: ", e$message))
  
  # Compile
  results_df <- do.call(rbind, all_results)
  rownames(results_df) <- NULL
  
  cat("\n", strrep("=", 80), "\n")
  cat("  Heteroskedasticity Test Results Summary\n")
  cat(strrep("=", 80), "\n")
  print(results_df)
  
  save_table(results_df, "heterosk_tests_R.csv")
  
  # Interpretation
  cat("\n", strrep("─", 60), "\n")
  cat("  Interpretation:\n")
  for (ds in unique(results_df$Dataset)) {
    bp <- results_df[results_df$Dataset == ds & results_df$Test == "Breusch-Pagan", ]
    if (nrow(bp) > 0) {
      if (bp$p_value[1] < 0.05) {
        cat("  ", ds, ": BP REJECTS homoskedasticity (p=", bp$p_value[1],
            ") → WLS/GLS recommended.\n")
      } else {
        cat("  ", ds, ": BP fails to reject homoskedasticity (p=", bp$p_value[1], ").\n")
      }
    }
  }
  
  return(results_df)
}

main <- function() {
  run_all_tests()
}

if (sys.nframe() == 0L || !interactive()) {
  main()
}

