# ===========================================================================
# run_all.R — Master runner for the complete GLS/WLS analysis pipeline (R).
#
# Sources and executes all analysis scripts in order, timing each step.
#
# Author : Ankan Bera
# Created: 2026-11-09
# ===========================================================================

SCRIPTS_DIR <- file.path("/home/ankanbera/Documents/Economatrics_project",
                         "scripts", "R")

run_step <- function(script_name, step_desc) {
  cat("\n", strrep("#", 70), "\n")
  cat("  Step: ", step_desc, "\n")
  cat(strrep("#", 70), "\n\n")
  
  script_path <- file.path(SCRIPTS_DIR, script_name)
  t0 <- proc.time()
  
  tryCatch({
    source(script_path, local = new.env(parent = globalenv()))
  }, error = function(e) {
    cat("  ERROR in '", step_desc, "': ", e$message, "\n")
  })
  
  elapsed <- (proc.time() - t0)["elapsed"]
  cat(sprintf("\n  Step '%s' completed in %.1f seconds.\n", step_desc, elapsed))
  return(elapsed)
}

main <- function() {
  cat(strrep("=", 70), "\n")
  cat("  GLS/WLS Econometrics Project — Full Analysis Pipeline (R)\n")
  cat(strrep("=", 70), "\n")
  
  total_start <- proc.time()
  timings <- list()
  
  timings[["Data Cleaning"]] <- run_step("data_cleaning.R",
                                          "Data Cleaning & Preparation")
  
  timings[["OLS Analysis"]] <- run_step("ols_analysis.R",
                                         "OLS Estimation & Diagnostics")
  
  timings[["Heterosk Tests"]] <- run_step("heteroskedasticity_tests.R",
                                           "Heteroskedasticity Tests")
  
  timings[["WLS/GLS"]] <- run_step("wls_gls_analysis.R",
                                    "WLS/GLS Estimation & Comparison")
  
  timings[["Robust SEs"]] <- run_step("robust_se.R",
                                       "Robust Standard Errors")
  
  cat("\n", strrep("!", 70), "\n")
  cat("  NOTE: Monte Carlo simulations may take several minutes.\n")
  cat(strrep("!", 70), "\n")
  timings[["Monte Carlo"]] <- run_step("monte_carlo_simulation.R",
                                        "Monte Carlo Simulations")
  
  # Summary
  total_elapsed <- (proc.time() - total_start)["elapsed"]
  cat("\n", strrep("=", 70), "\n")
  cat("  Pipeline Complete — Timing Summary\n")
  cat(strrep("=", 70), "\n")
  for (nm in names(timings)) {
    cat(sprintf("  %-30s : %8.1fs\n", nm, timings[[nm]]))
  }
  cat(strrep("─", 42), "\n")
  cat(sprintf("  %-30s : %8.1fs\n", "TOTAL", total_elapsed))
  cat(strrep("=", 70), "\n")
  
  # List output files
  output_dir <- file.path("/home/ankanbera/Documents/Economatrics_project", "output")
  cat("\n  Output files:\n")
  files <- list.files(output_dir, recursive = TRUE, full.names = FALSE)
  files <- files[!startsWith(files, ".")]
  for (f in sort(files)) cat("    ", f, "\n")
  
  cat(sprintf("\n  All done! Total time: %.1fs\n", total_elapsed))
}

main()

