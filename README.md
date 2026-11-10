# Generalized Least Squares (GLS) & Weighted Least Squares (WLS)
## A Comprehensive Econometric Analysis, Simulation Study, and Real-World Application

An end-to-end econometric research project covering:
1. **Econometric Methodology:** Full mathematical derivations of OLS, WLS, GLS, Feasible GLS (FGLS), Heteroskedasticity-Consistent (White HC0–HC3) standard errors, Cluster-Robust SEs, and Newey-West HAC SEs, alongside Breusch–Pagan, White, and Goldfeld–Quandt diagnostic tests.
2. **Monte Carlo Simulation Study:** 5,000-replication controlled experiments across 3 scenarios:
   - Scenario 1: Heteroskedastic Errors ($\text{Var}(\varepsilon_i) \propto x_i^{2\varphi}$)
   - Scenario 2: AR(1) Autocorrelation ($\varepsilon_t = \rho \varepsilon_{t-1} + u_t$)
   - Scenario 3: Clustered Random Effects ($y_{ic} = X\beta + \alpha_c + \varepsilon_{ic}$)
3. **Application to Real-World Datasets:** Four major public datasets + California Housing:
   - **Boston Housing (UCI / MASS):** 506 suburbs, 14 variables (Cross-sectional price dispersion)
   - **Ames Housing (De Cock, 2011):** 2,930 homes, 12 features (Living area vs lot variance)
   - **California Housing (1990 US Census):** 20,640 census block groups (Large-scale housing)
   - **World Bank World Development Indicators (WDI):** 5,118 observations across 262 countries over 2003–2022 (Panel cross-country heteroskedasticity and autocorrelation)
   - **World Bank Enterprise Survey:** 5,000 firms across 30 economies calibrated with real WDI country GDP (Clustered microdata)
4. **Interactive Streamlit Web Interface:** Full interactive dashboard (`app.py`) for live regression fitting, interactive residual diagnostics, dynamic Monte Carlo simulations with sliders, and method comparisons.

---

## 📁 Repository Structure

```
Economatrics_project/
├── app.py                       # 🌟 Interactive Streamlit Dashboard (Full Web UI)
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation and complete guide
│
├── data/
│   ├── raw/                     # Original downloaded datasets (Boston, Ames XLS, WDI, CA)
│   └── processed/               # Cleaned real-world CSV files ready for estimation
│       ├── boston_clean.csv     # 506 obs × 14 vars
│       ├── ames_clean.csv       # 2,930 obs × 12 vars
│       ├── california_clean.csv # 20,640 obs × 10 vars
│       ├── wdi_panel.csv        # 5,118 obs × 10 vars (262 countries, 2003-2022)
│       └── enterprise_clean.csv # 5,000 obs × 8 vars (30 countries)
│
├── scripts/
│   ├── python/                  # Python econometric engine
│   │   ├── config.py            # Global paths, seeds, plot themes
│   │   ├── prepare_real_data.py # Automated fetcher for all real datasets
│   │   ├── ols_analysis.py      # OLS models, VIF, residual & Q-Q plots
│   │   ├── heteroskedasticity_tests.py # BP, White, GQ tests & residual² plots
│   │   ├── wls_gls_analysis.py  # WLS, FGLS, GLSAR models & SE comparison
│   │   ├── robust_se.py         # HC0-HC3, cluster-robust, Newey-West SEs
│   │   ├── monte_carlo_simulation.py # 5,000 rep simulations across 3 DGPs
│   │   ├── visualization.py     # Reusable publication-grade plotting utilities
│   │   └── run_all.py           # Master CLI runner orchestrating all steps
│   │
│   └── R/                       # Complete parallel R implementation
│       ├── config.R             # Global paths, ggplot2 theme, libraries
│       ├── data_cleaning.R      # R data processing
│       ├── ols_analysis.R       # lm(), VIF (car), diagnostic plots
│       ├── heteroskedasticity_tests.R # bptest, ncvTest, gqtest (lmtest, car)
│       ├── wls_gls_analysis.R   # lm(weights=), nlme::gls, plm
│       ├── robust_se.R          # sandwich::vcovHC, vcovCL, NeweyWest
│       ├── monte_carlo_simulation.R # 5,000 reps in R
│       └── run_all.R            # Master R runner
│
├── output/
│   ├── figures/                 # 28 publication-grade PNG plots (300 DPI)
│   │   ├── resid_vs_fitted_*.png
│   │   ├── qq_plot_*.png
│   │   ├── resid_sq_vs_fitted_*.png
│   │   ├── resid_vs_preds_*.png
│   │   ├── coef_compare_*.png
│   │   └── mc_s{1,2,3}_*.png    # Monte Carlo RMSE, Bias, Coverage charts
│   └── tables/                  # Exported econometric summary CSV tables
│       ├── ols_summary_all.csv
│       ├── heterosk_tests.csv
│       ├── wls_gls_comparison_all.csv
│       ├── robust_se_comparison.csv
│       ├── vif_*.csv
│       └── mc_scenario{1,2,3}.csv
│
└── report/
    └── report_template.md       # Formatted paper draft covering all sections
```

---

## 🚀 How to Run the Interactive UI

The easiest and most intuitive way to explore the project is via the interactive **Streamlit dashboard**.

### Step 1: Activate Environment
```bash
cd /home/ankanbera/Documents/Economatrics_project
source .venv/bin/activate
```

### Step 2: Launch Streamlit
```bash
streamlit run app.py
```

### Step 3: Open in Browser
Visit the URL displayed in your terminal (usually `http://localhost:8501`).

### What You Can Do in the UI:
1. **📐 Methodology:** Interactive derivation cards for Gauss–Markov assumptions, WLS minimization, full GLS matrix algebra, FGLS iterative variance functions, White HC0–HC3 formulas, cluster corrections, and diagnostic test procedures.
2. **🗂️ Data Explorer:** Switch between Boston, Ames, California Housing, World Bank WDI, and Enterprise Surveys. View summary statistics, histograms, box plots, correlation heatmaps, and scatter matrices.
3. **📈 OLS & Diagnostics:** Fit OLS on any dataset, customize the predictors dynamically, view coefficients with 95% CIs, inspect interactive Residuals vs Fitted with LOWESS smoothers, Q-Q plots, and execute Breusch–Pagan, White, and Goldfeld–Quandt tests with automatic p-value interpretation.
4. **⚖️ WLS / GLS Analysis:** Interactively select weighting variables (e.g., $1/\text{lstat}^2$, $1/\text{GrLivArea}$, $1/\text{size}$) and weight transformation powers ($1/x$, $1/x^2$, $1/\sqrt{x}$). Compare OLS vs WLS vs FGLS side-by-side with grouped bar charts and SE ratios.
5. **🛡️ Robust Standard Errors:** Inspect and contrast naive OLS SEs, HC0, HC1, HC3, Cluster-Robust SEs (by country), and Newey-West HAC SEs with a color-coded ratio heatmap.
6. **🎲 Monte Carlo Simulation:** Run live simulation experiments directly from the browser! Adjust replications, sample size ($N$), true $\beta_1$, and parameter grids ($\varphi, \rho, \tau^2$) to observe how heteroskedasticity and clustering degrade naive OLS confidence interval coverage in real time.
7. **🏆 Full Comparison:** Comprehensive master scorecard summarizing fit, test rejection status, and efficiency ratios across all datasets simultaneously.

---

## 💻 How to Run from the Command Line

### 1. Python Pipeline

#### Run the Entire Pipeline End-to-End:
```bash
cd /home/ankanbera/Documents/Economatrics_project
source .venv/bin/activate
python scripts/python/run_all.py
```

#### Run Individual Scripts:
```bash
# 1. Fetch & clean all real datasets
python scripts/python/prepare_real_data.py

# 2. Fit OLS regressions and generate diagnostic plots
python scripts/python/ols_analysis.py

# 3. Perform Breusch-Pagan, White, and Goldfeld-Quandt tests
python scripts/python/heteroskedasticity_tests.py

# 4. Fit WLS, FGLS, GLSAR and compare with OLS
python scripts/python/wls_gls_analysis.py

# 5. Compute HC0-HC3, Cluster, and Newey-West standard errors
python scripts/python/robust_se.py

# 6. Run 5,000-replication Monte Carlo simulation study
python scripts/python/monte_carlo_simulation.py
```

---

### 2. R Pipeline (Parallel Implementation)

If you or your instructor prefer R:
```bash
cd /home/ankanbera/Documents/Economatrics_project
Rscript scripts/R/run_all.R
```
Or run individual R modules in RStudio or terminal:
```r
source("scripts/R/data_cleaning.R")
source("scripts/R/ols_analysis.R")
source("scripts/R/heteroskedasticity_tests.R")
source("scripts/R/wls_gls_analysis.R")
source("scripts/R/robust_se.R")
source("scripts/R/monte_carlo_simulation.R")
```

---

## 📊 Summary of Econometric Results

### 1. Real Data Diagnostics (Breusch–Pagan Tests)
| Dataset | $N$ | BP LM Stat | BP p-value | Conclusion | Remedial Action |
|---------|-----|------------|------------|------------|-----------------|
| **Boston Housing** | 506 | 42.99 | $< 10^{-5}$ | Heteroskedasticity | WLS ($1/\text{lstat}^2$) / FGLS |
| **Ames Housing** | 2,930 | 294.05 | $< 10^{-10}$ | Extreme Heteroskedasticity | WLS ($1/\text{GrLivArea}$) / FGLS |
| **World Bank WDI** | 3,761 | 31.40 | $< 10^{-5}$ | Panel Heteroskedasticity | Cluster SEs by country + GLSAR |
| **Enterprise Survey** | 5,000 | 655.74 | $< 10^{-15}$ | Size-dependent Variance | WLS ($1/\text{size}$) + Cluster SE |

### 2. Monte Carlo Simulation Highlights (5,000 Replications)
- **Heteroskedasticity ($\varphi = 1.5$):** Naive OLS 95% CI coverage collapses to **90.8%**, while WLS restores coverage to **94.9%** and slashes RMSE from **0.691** to **0.291** (a **58% efficiency improvement**).
- **AR(1) Autocorrelation ($\rho = 0.9$):** GLS with Prais–Winsten reduces RMSE from **0.071** down to **0.026** (a **63% reduction in estimation error**).
- **Clustering ($\tau^2 = 2.0$):** Naive OLS standard errors underestimate variance, whereas cluster-robust standard errors and RE-GLS preserve nominal 95% coverage and capture the group-level random component.
