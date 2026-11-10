# Generalized Least Squares and Weighted Least Squares: A Comprehensive Empirical and Simulation Study

**Author:** Ankan Bera  
**Date:** November 2026

---

## Abstract

This project conducts a comprehensive comparison of Ordinary Least Squares (OLS), Weighted Least Squares (WLS), and Generalized Least Squares (GLS) estimators using both Monte Carlo simulations and real-world datasets. We demonstrate that under heteroskedasticity, autocorrelation, and clustered error structures, OLS standard errors are biased and confidence interval coverage deteriorates. WLS with correct weights restores efficiency; FGLS provides practical alternatives when the error covariance structure is unknown. Using four datasets — Boston Housing, Ames Housing, World Bank WDI panel, and Enterprise Survey microdata — we apply formal diagnostic tests (Breusch-Pagan, White, Goldfeld-Quandt), fit WLS/GLS models, and compare robust inference methods (HC0-HC3, cluster-robust, Newey-West). Monte Carlo experiments across three scenarios (heteroskedasticity, AR(1), clustering) with 5,000 replications confirm theoretical predictions about estimator efficiency and coverage. All code is provided in both R and Python for full reproducibility.

**Keywords:** heteroskedasticity, GLS, WLS, FGLS, robust standard errors, Monte Carlo simulation, panel data

---

## 1. Introduction

*(300–400 words)*

The classical linear regression model $y = X\beta + \varepsilon$ relies on the assumption that the error term $\varepsilon$ has a spherical covariance matrix: $\text{Var}(\varepsilon) = \sigma^2 I_n$. This implies both homoskedasticity (constant variance) and no autocorrelation. In practice, these assumptions frequently fail.

**Heteroskedasticity** — non-constant error variance — is pervasive in cross-sectional data. Housing price residuals, for instance, tend to be larger for expensive homes. **Autocorrelation** commonly arises in time-series and panel data where successive observations share temporal shocks. **Clustering** occurs in hierarchical data (firms within countries, students within schools) where observations share group-level unobservables.

When these violations occur, OLS coefficient estimates remain unbiased but are no longer efficient (minimum variance). More critically, the conventional OLS standard errors are biased, leading to incorrect inference — hypothesis tests reject too often or too rarely, and confidence intervals have incorrect coverage.

Two principal remedies exist. First, **Weighted Least Squares (WLS)** and **Generalized Least Squares (GLS)** transform the model to restore efficiency by accounting for the known or estimated error covariance structure $\Sigma$. The GLS estimator $\hat{\beta}_{GLS} = (X'\Sigma^{-1}X)^{-1}X'\Sigma^{-1}y$ achieves the Gauss-Markov bound when $\Sigma$ is correctly specified. When $\Sigma$ is estimated from data (Feasible GLS), asymptotic efficiency is preserved under regularity conditions.

Second, **robust standard error** methods correct the variance-covariance matrix of OLS estimates without changing the point estimates. White's heteroskedasticity-consistent (HC) estimator, cluster-robust standard errors, and Newey-West HAC standard errors are widely used in applied econometrics.

This project pursues three objectives: (1) derive and implement OLS, WLS, GLS, and FGLS estimators with associated diagnostic tests; (2) evaluate their finite-sample properties via Monte Carlo simulation under controlled data-generating processes; and (3) apply these methods to four empirical datasets spanning cross-sectional, panel, and clustered structures. All analysis is conducted in both R and Python to ensure reproducibility and accessibility.

---

## 2. Data

*(For each dataset: source, variables, sample size, citations)*

### 2.1 Boston Housing (MASS)
**Source:** Harrison, D. and Rubinfeld, D.L. (1978), via the R MASS package.  
**Sample:** 506 observations, 14 variables.  
**Key variables:** `medv` (median home value, \$1000s), `crim` (crime rate), `rm` (average rooms), `tax` (property tax rate), `lstat` (% lower status population), `ptratio` (pupil-teacher ratio).  
**Structure:** Cross-sectional data on Boston suburbs.

### 2.2 Ames Housing (Kaggle)
**Source:** De Cock, D. (2011), via the Kaggle "House Prices – Advanced Regression" competition.  
**Sample:** 1,460 observations, 81 variables.  
**Key variables:** `SalePrice`, `GrLivArea` (living area sq ft), `LotArea`, `OverallQual` (1–10), `YearBuilt`, `TotalBsmtSF`, `GarageCars`.  
**Structure:** Cross-sectional data on homes in Ames, Iowa.

### 2.3 World Bank WDI Panel (Synthetic)
**Source:** Synthetic data calibrated to World Bank World Development Indicators.  
**Sample:** 200 countries × 20 years = 4,000 observations.  
**Key variables:** `gdp_pc`, `gdp_growth`, `inflation`, `life_exp`, `investment_rate`, `log_pop`.  
**Structure:** Balanced country-year panel.

### 2.4 Enterprise Survey (Synthetic)
**Source:** Synthetic data calibrated to World Bank Enterprise Surveys.  
**Sample:** 5,000 firms across 30 countries.  
**Key variables:** `log_revenue`, `log_capital`, `log_labor`, `firm_size_dummy`, `country`, `sector`.  
**Structure:** Clustered cross-section (firms nested within countries).

---

## 3. Methodology

*(600–800 words)*

### 3.1 Ordinary Least Squares

[Derive OLS estimator, state assumptions, discuss consequences of heteroskedasticity]

### 3.2 Weighted Least Squares

[Derive WLS as special case of GLS with diagonal Σ. Show weight choice.]

### 3.3 Generalized Least Squares and FGLS

[Present GLS estimator. Discuss FGLS: estimating Σ from data.]

### 3.4 Robust Inference

[White HC estimator. Cluster-robust SEs. Newey-West HAC.]

### 3.5 Diagnostic Tests

[Breusch-Pagan, White, Goldfeld-Quandt, Durbin-Watson]

---

## 4. Simulation Study

*(600–800 words)*

### 4.1 Design

[Describe three DGPs, parameter choices, 5000 reps]

### 4.2 Results

[Insert tables from `output/tables/mc_scenario{1,2,3}.csv`]

[Insert figures from `output/figures/mc_s{1,2,3}_*.png`]

### 4.3 Discussion

[Interpret: OLS undercoverage under heteroskedasticity, GLS efficiency gains, etc.]

---

## 5. Empirical Results

*(800–1000 words)*

### 5.1 Boston Housing

[Insert regression table, heterosk test results, coefficient comparison]

### 5.2 Ames Housing

[Similar structure]

### 5.3 WDI Panel

[Include panel-specific methods: cluster SEs, GLSAR]

### 5.4 Enterprise Survey

[Cluster-robust inference, WLS by firm size]

---

## 6. Discussion

*(300–400 words)*

[When do WLS/GLS help? Limitations of weight choice. Small-sample issues with FGLS.]

---

## 7. Conclusion

*(150–200 words)*

[Key takeaways. Reproducibility emphasis.]

---

## References

1. Breusch, T.S. & Pagan, A.R. (1979). "A Simple Test for Heteroscedasticity and Random Coefficient Variation." *Econometrica*, 47(5), 1287–1294.
2. Cameron, A.C. & Miller, D.L. (2015). "A Practitioner's Guide to Cluster-Robust Inference." *Journal of Human Resources*, 50(2), 317–372.
3. De Cock, D. (2011). "Ames, Iowa: Alternative to the Boston Housing Data." *Journal of Statistics Education*, 19(3).
4. Greene, W.H. (2018). *Econometric Analysis*, 8th edition. Pearson.
5. Harrison, D. & Rubinfeld, D.L. (1978). "Hedonic Housing Prices and the Demand for Clean Air." *Journal of Environmental Economics and Management*, 5, 81–102.
6. Newey, W.K. & West, K.D. (1987). "A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix." *Econometrica*, 55(3), 703–708.
7. White, H. (1980). "A Heteroskedasticity-Consistent Covariance Matrix Estimator and a Direct Test for Heteroskedasticity." *Econometrica*, 48(4), 817–838.
8. Wooldridge, J.M. (2019). *Introductory Econometrics: A Modern Approach*, 7th edition. Cengage.

---

## Appendix A: Reproducibility Checklist

- [ ] All data sources listed with URLs
- [ ] Python `requirements.txt` included
- [ ] R package versions documented
- [ ] Random seed set (42) in all scripts
- [ ] All scripts produce identical outputs from raw data
- [ ] `run_all.py` and `run_all.R` execute complete pipeline

## Appendix B: Software Versions

| Software | Version |
|----------|---------|
| Python   | 3.12.x  |
| R        | 4.3.x   |
| numpy    | ≥1.25   |
| pandas   | ≥2.1    |
| statsmodels | ≥0.14 |
| lmtest (R)  | 0.9.40 |
| sandwich (R) | 3.1+  |

