# ===========================================================================
# ols_analysis.R — OLS estimation and diagnostics for all datasets.
#
# Fits OLS models, computes VIF, generates residual diagnostic plots,
# and produces regression summary tables.
#
# Author : Ankan Bera
# Created: 2026-11-09
# ===========================================================================

source(file.path(dirname(sys.frame(1)$ofile %||% "."), "config.R"))

# ============================================================================
# Helper: Residuals vs Fitted Plot (ggplot2)
# ============================================================================

plot_resid_vs_fitted <- function(model, dataset_name) {
  df_plot <- data.frame(
    fitted = fitted(model),
    resid  = residuals(model)
  )
  p <- ggplot(df_plot, aes(x = fitted, y = resid)) +
    geom_point(alpha = 0.4, size = 1.5, color = COLOR_PALETTE["OLS"]) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
    geom_smooth(method = "loess", se = FALSE, color = "red", linewidth = 1) +
    labs(
      title = paste("Residuals vs Fitted —", dataset_name, "OLS"),
      x = "Fitted Values", y = "Residuals"
    )
  save_plot(p, paste0("resid_vs_fitted_", dataset_name, "_R.png"))
  return(p)
}

# ============================================================================
# Helper: Q-Q Plot
# ============================================================================

plot_qq <- function(model, dataset_name) {
  df_plot <- data.frame(resid = residuals(model))
  p <- ggplot(df_plot, aes(sample = resid)) +
    stat_qq(alpha = 0.4, color = COLOR_PALETTE["OLS"]) +
    stat_qq_line(color = "red") +
    labs(
      title = paste("Q-Q Plot —", dataset_name, "OLS Residuals"),
      x = "Theoretical Quantiles", y = "Sample Quantiles"
    )
  save_plot(p, paste0("qq_plot_", dataset_name, "_R.png"))
  return(p)
}

# ============================================================================
# OLS Fitting for Each Dataset
# ============================================================================

ols_boston <- function() {
  message("\n── OLS: Boston Housing ────────────────────────────────────")
  df <- read.csv(file.path(PROCESSED_DIR, "boston_clean.csv"))
  
  model <- lm(medv ~ crim + zn + indus + rm + age + tax + ptratio + lstat,
               data = df)
  
  cat("\n"); print(summary(model))
  
  # VIF
  vif_vals <- car::vif(model)
  cat("\nVariance Inflation Factors:\n")
  print(round(vif_vals, 3))
  save_table(data.frame(Variable = names(vif_vals), VIF = round(vif_vals, 3)),
             "vif_boston_R.csv")
  
  # Diagnostic plots
  plot_resid_vs_fitted(model, "Boston")
  plot_qq(model, "Boston")
  
  return(model)
}

ols_ames <- function() {
  message("\n── OLS: Ames Housing ──────────────────────────────────────")
  df <- read.csv(file.path(PROCESSED_DIR, "ames_clean.csv"))
  
  model <- lm(log_SalePrice ~ GrLivArea + LotArea + OverallQual +
                 YearBuilt + TotalBsmtSF + GarageCars, data = df)
  
  cat("\n"); print(summary(model))
  
  vif_vals <- car::vif(model)
  cat("\nVariance Inflation Factors:\n")
  print(round(vif_vals, 3))
  save_table(data.frame(Variable = names(vif_vals), VIF = round(vif_vals, 3)),
             "vif_ames_R.csv")
  
  plot_resid_vs_fitted(model, "Ames")
  plot_qq(model, "Ames")
  
  return(model)
}

ols_wdi <- function() {
  message("\n── OLS: WDI Panel (Pooled) ────────────────────────────────")
  df <- read.csv(file.path(PROCESSED_DIR, "wdi_panel.csv"))
  
  model <- lm(gdp_growth ~ lag_gdp_pc + investment_rate + inflation + log_pop,
               data = df)
  
  cat("\n"); print(summary(model))
  
  vif_vals <- car::vif(model)
  cat("\nVariance Inflation Factors:\n")
  print(round(vif_vals, 3))
  save_table(data.frame(Variable = names(vif_vals), VIF = round(vif_vals, 3)),
             "vif_wdi_R.csv")
  
  plot_resid_vs_fitted(model, "WDI")
  plot_qq(model, "WDI")
  
  return(model)
}

ols_enterprise <- function() {
  message("\n── OLS: Enterprise Survey ─────────────────────────────────")
  df <- read.csv(file.path(PROCESSED_DIR, "enterprise_clean.csv"))
  
  model <- lm(log_revenue ~ log_capital + log_labor + firm_size_dummy,
               data = df)
  
  cat("\n"); print(summary(model))
  
  vif_vals <- car::vif(model)
  cat("\nVariance Inflation Factors:\n")
  print(round(vif_vals, 3))
  save_table(data.frame(Variable = names(vif_vals), VIF = round(vif_vals, 3)),
             "vif_enterprise_R.csv")
  
  plot_resid_vs_fitted(model, "Enterprise")
  plot_qq(model, "Enterprise")
  
  return(model)
}

# ============================================================================
# Run All
# ============================================================================

run_all_ols <- function() {
  models <- list()
  
  for (info in list(
    list(name = "boston",     func = ols_boston),
    list(name = "ames",      func = ols_ames),
    list(name = "wdi",       func = ols_wdi),
    list(name = "enterprise", func = ols_enterprise)
  )) {
    tryCatch({
      models[[info$name]] <- info$func()
    }, error = function(e) {
      message("  ERROR in OLS for ", info$name, ": ", e$message)
    })
  }
  
  # Summary comparison table
  summary_rows <- lapply(names(models), function(nm) {
    m <- models[[nm]]
    s <- summary(m)
    data.frame(
      Dataset  = nm,
      N        = nobs(m),
      R2       = round(s$r.squared, 4),
      Adj_R2   = round(s$adj.r.squared, 4),
      F_stat   = round(s$fstatistic[1], 2),
      RSE      = round(s$sigma, 4),
      stringsAsFactors = FALSE
    )
  })
  summary_df <- do.call(rbind, summary_rows)
  
  cat("\n", strrep("=", 60), "\n")
  cat("  OLS Model Comparison Summary\n")
  cat(strrep("=", 60), "\n")
  print(summary_df)
  save_table(summary_df, "ols_summary_all_R.csv")
  
  # Stargazer table (if available)
  if (requireNamespace("stargazer", quietly = TRUE)) {
    stargazer::stargazer(
      models[["boston"]], models[["ames"]],
      models[["wdi"]], models[["enterprise"]],
      type = "text",
      title = "OLS Regression Results",
      column.labels = names(models),
      out = file.path(TABLES_DIR, "ols_stargazer_R.txt")
    )
  }
  
  return(models)
}

main <- function() {
  message("Starting OLS analysis pipeline (R)")
  models <- run_all_ols()
  message("OLS analysis complete. ", length(models), " models fitted.")
  invisible(models)
}

if (sys.nframe() == 0L || !interactive()) {
  main()
}

