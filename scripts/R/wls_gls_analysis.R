# ===========================================================================
# wls_gls_analysis.R — WLS, GLS, and FGLS estimation for all datasets.
#
# Fits weighted and generalized least squares models, compares with OLS,
# and produces side-by-side tables and coefficient comparison plots.
#
# Author : Ankan Bera
# Created: 2026-11-09
# ===========================================================================

source(file.path(dirname(sys.frame(1)$ofile %||% "."), "config.R"))

# ============================================================================
# Helper: Coefficient comparison plot
# ============================================================================

plot_coef_comparison <- function(models, labels, dataset_name) {
  # Extract coefficients and SEs (exclude intercept for cleaner plots)
  coef_list <- lapply(seq_along(models), function(i) {
    m <- models[[i]]
    cf <- coef(m)
    se <- sqrt(diag(vcov(m)))
    # Remove intercept
    idx <- !grepl("^\\(Intercept\\)$", names(cf))
    data.frame(
      Variable = names(cf)[idx],
      Coef     = cf[idx],
      SE       = se[idx],
      Method   = labels[i],
      stringsAsFactors = FALSE
    )
  })
  df <- do.call(rbind, coef_list)
  rownames(df) <- NULL
  
  p <- ggplot(df, aes(x = Variable, y = Coef, fill = Method)) +
    geom_bar(stat = "identity", position = position_dodge(width = 0.8),
             alpha = 0.8, width = 0.7) +
    geom_errorbar(aes(ymin = Coef - 1.96 * SE, ymax = Coef + 1.96 * SE),
                  position = position_dodge(width = 0.8), width = 0.25) +
    labs(
      title = paste("Coefficient Comparison —", dataset_name),
      y = "Coefficient ± 1.96 SE"
    ) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    scale_fill_manual(values = unname(COLOR_PALETTE[1:length(labels)]))
  
  save_plot(p, paste0("coef_compare_", tolower(dataset_name), "_R.png"))
  return(df)
}

# ============================================================================
# FGLS helper: estimate variance function
# ============================================================================

fgls_weights <- function(ols_model) {
  resid_sq <- residuals(ols_model)^2
  log_resid_sq <- log(pmax(resid_sq, 1e-10))
  
  # Regress log(e²) on original predictors
  X_var <- model.matrix(ols_model)
  var_model <- lm(log_resid_sq ~ X_var - 1)
  pred_log_var <- fitted(var_model)
  
  weights <- 1 / exp(pred_log_var)
  weights <- ifelse(is.finite(weights) & weights > 0, weights, 1)
  return(weights)
}

# ============================================================================
# Dataset-Specific Analyses
# ============================================================================

analyze_boston <- function() {
  message("\n── WLS/GLS: Boston Housing ────────────────────────────────")
  df <- read.csv(file.path(PROCESSED_DIR, "boston_clean.csv"))
  
  # OLS
  ols <- lm(medv ~ crim + zn + indus + rm + age + tax + ptratio + lstat, data = df)
  
  # WLS: w = 1/lstat²
  wls <- lm(medv ~ crim + zn + indus + rm + age + tax + ptratio + lstat,
             data = df, weights = 1 / pmax(lstat, 0.1)^2)
  
  # FGLS
  fgls_w <- fgls_weights(ols)
  fgls_m <- lm(medv ~ crim + zn + indus + rm + age + tax + ptratio + lstat,
                data = df, weights = fgls_w)
  
  # Print comparison
  cat("\n  OLS vs WLS vs FGLS — Boston:\n")
  if (requireNamespace("stargazer", quietly = TRUE)) {
    stargazer::stargazer(ols, wls, fgls_m, type = "text",
                         column.labels = c("OLS", "WLS", "FGLS"),
                         title = "Boston Housing: OLS vs WLS vs FGLS")
  } else {
    cat("\n  OLS coefficients:\n"); print(round(coef(ols), 4))
    cat("\n  WLS coefficients:\n"); print(round(coef(wls), 4))
    cat("\n  FGLS coefficients:\n"); print(round(coef(fgls_m), 4))
  }
  
  # Coefficient comparison
  plot_coef_comparison(list(ols, wls, fgls_m),
                       c("OLS", "WLS (1/lstat²)", "FGLS"),
                       "Boston")
  
  # SE comparison
  se_comp <- data.frame(
    Variable = names(coef(ols)),
    OLS_SE   = round(summary(ols)$coefficients[, 2], 6),
    WLS_SE   = round(summary(wls)$coefficients[, 2], 6),
    FGLS_SE  = round(summary(fgls_m)$coefficients[, 2], 6)
  )
  cat("\n  Standard Error Comparison — Boston:\n")
  print(se_comp)
  save_table(se_comp, "se_compare_boston_R.csv")
  
  return(list(ols = ols, wls = wls, fgls = fgls_m))
}

analyze_ames <- function() {
  message("\n── WLS/GLS: Ames Housing ──────────────────────────────────")
  df <- read.csv(file.path(PROCESSED_DIR, "ames_clean.csv"))
  
  ols <- lm(log_SalePrice ~ GrLivArea + LotArea + OverallQual +
               YearBuilt + TotalBsmtSF + GarageCars, data = df)
  
  wls <- lm(log_SalePrice ~ GrLivArea + LotArea + OverallQual +
               YearBuilt + TotalBsmtSF + GarageCars,
             data = df, weights = 1 / pmax(GrLivArea, 1))
  
  fgls_w <- fgls_weights(ols)
  fgls_m <- lm(log_SalePrice ~ GrLivArea + LotArea + OverallQual +
                  YearBuilt + TotalBsmtSF + GarageCars,
                data = df, weights = fgls_w)
  
  if (requireNamespace("stargazer", quietly = TRUE)) {
    stargazer::stargazer(ols, wls, fgls_m, type = "text",
                         column.labels = c("OLS", "WLS", "FGLS"))
  }
  
  plot_coef_comparison(list(ols, wls, fgls_m),
                       c("OLS", "WLS (1/GrLivArea)", "FGLS"), "Ames")
  
  se_comp <- data.frame(
    Variable = names(coef(ols)),
    OLS_SE   = round(summary(ols)$coefficients[, 2], 6),
    WLS_SE   = round(summary(wls)$coefficients[, 2], 6),
    FGLS_SE  = round(summary(fgls_m)$coefficients[, 2], 6)
  )
  save_table(se_comp, "se_compare_ames_R.csv")
  
  return(list(ols = ols, wls = wls, fgls = fgls_m))
}

analyze_wdi <- function() {
  message("\n── GLS: WDI Panel ────────────────────────────────────────")
  df <- read.csv(file.path(PROCESSED_DIR, "wdi_panel.csv"))
  df <- na.omit(df[, c("country_id", "year", "gdp_growth", "lag_gdp_pc",
                        "investment_rate", "inflation", "log_pop")])
  
  # Pooled OLS
  ols <- lm(gdp_growth ~ lag_gdp_pc + investment_rate + inflation + log_pop,
             data = df)
  
  # GLS with AR(1) via nlme::gls
  tryCatch({
    gls_ar1 <- nlme::gls(
      gdp_growth ~ lag_gdp_pc + investment_rate + inflation + log_pop,
      data = df,
      correlation = nlme::corAR1(form = ~ year | country_id)
    )
    cat("\n  GLS AR(1) Summary:\n")
    print(summary(gls_ar1))
  }, error = function(e) {
    message("  GLS AR(1) failed: ", e$message)
    gls_ar1 <<- ols  # fallback
  })
  
  # Panel random effects via plm (if available)
  tryCatch({
    if (requireNamespace("plm", quietly = TRUE)) {
      pdf <- plm::pdata.frame(df, index = c("country_id", "year"))
      re_model <- plm::plm(
        gdp_growth ~ lag_gdp_pc + investment_rate + inflation + log_pop,
        data = pdf, model = "random"
      )
      cat("\n  Panel Random Effects Summary:\n")
      print(summary(re_model))
    }
  }, error = function(e) {
    message("  Panel RE failed: ", e$message)
  })
  
  # SE comparison
  se_comp <- data.frame(
    Variable = names(coef(ols)),
    OLS_SE   = round(summary(ols)$coefficients[, 2], 6)
  )
  tryCatch({
    se_comp$GLS_AR1_SE <- round(summary(gls_ar1)$tTable[, 2], 6)
  }, error = function(e) NULL)
  
  cat("\n  SE Comparison — WDI:\n")
  print(se_comp)
  save_table(se_comp, "se_compare_wdi_R.csv")
  
  return(list(ols = ols))
}

analyze_enterprise <- function() {
  message("\n── WLS/GLS: Enterprise Survey ─────────────────────────────")
  df <- read.csv(file.path(PROCESSED_DIR, "enterprise_clean.csv"))
  
  ols <- lm(log_revenue ~ log_capital + log_labor + firm_size_dummy, data = df)
  
  # WLS: w = 1 / (firm_size_dummy + 1)
  wls <- lm(log_revenue ~ log_capital + log_labor + firm_size_dummy,
             data = df, weights = 1 / (firm_size_dummy + 1))
  
  fgls_w <- fgls_weights(ols)
  fgls_m <- lm(log_revenue ~ log_capital + log_labor + firm_size_dummy,
                data = df, weights = fgls_w)
  
  if (requireNamespace("stargazer", quietly = TRUE)) {
    stargazer::stargazer(ols, wls, fgls_m, type = "text",
                         column.labels = c("OLS", "WLS", "FGLS"))
  }
  
  plot_coef_comparison(list(ols, wls, fgls_m),
                       c("OLS", "WLS (1/size)", "FGLS"), "Enterprise")
  
  se_comp <- data.frame(
    Variable = names(coef(ols)),
    OLS_SE   = round(summary(ols)$coefficients[, 2], 6),
    WLS_SE   = round(summary(wls)$coefficients[, 2], 6),
    FGLS_SE  = round(summary(fgls_m)$coefficients[, 2], 6)
  )
  save_table(se_comp, "se_compare_enterprise_R.csv")
  
  return(list(ols = ols, wls = wls, fgls = fgls_m))
}

# ============================================================================
# Run All
# ============================================================================

main <- function() {
  message("Starting WLS/GLS analysis pipeline (R)")
  
  tryCatch(analyze_boston(), error = function(e) message("Boston failed: ", e$message))
  tryCatch(analyze_ames(), error = function(e) message("Ames failed: ", e$message))
  tryCatch(analyze_wdi(), error = function(e) message("WDI failed: ", e$message))
  tryCatch(analyze_enterprise(), error = function(e) message("Enterprise failed: ", e$message))
  
  message("\nWLS/GLS analysis complete.")
}

if (sys.nframe() == 0L || !interactive()) {
  main()
}

