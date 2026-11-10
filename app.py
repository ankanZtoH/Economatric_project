#!/usr/bin/env python3
"""
app.py — Interactive Streamlit Dashboard for the GLS/WLS Econometrics Project.

Covers all three pillars:
  (i)   Econometric Methodology (theory, formulas, derivations)
  (ii)  Simulation Study (interactive Monte Carlo experiments)
  (iii) Application to Real Data (4 datasets with full diagnostics)

Run:  streamlit run app.py
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, het_white, het_goldfeldquandt
from statsmodels.regression.linear_model import WLS, GLSAR
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")

# ─── Page Config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GLS/WLS Econometrics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1f4e79; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.1rem; color: #555; margin-bottom: 1.5rem; }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px; padding: 1rem 1.2rem; color: white; text-align: center;
    }
    .metric-card h3 { margin: 0; font-size: 0.85rem; opacity: 0.9; }
    .metric-card h1 { margin: 0.2rem 0 0 0; font-size: 1.8rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 20px; border-radius: 6px 6px 0 0;
        font-weight: 600;
    }
    div[data-testid="stExpander"] details summary p { font-weight: 600; font-size: 1.05rem; }
</style>
""", unsafe_allow_html=True)


###############################################################################
# DATA LOADING
###############################################################################

@st.cache_data
def load_dataset(name):
    """Load a processed dataset by name."""
    path_map = {
        "Boston Housing": "boston_clean.csv",
        "Ames Housing (Real)": "ames_clean.csv",
        "California Housing": "california_clean.csv",
        "World Bank WDI (Real)": "wdi_panel.csv",
        "Enterprise Survey": "enterprise_clean.csv",
    }
    fname = path_map.get(name)
    if fname and os.path.exists(os.path.join(DATA_DIR, fname)):
        return pd.read_csv(os.path.join(DATA_DIR, fname))
    # Fallback for California Housing from raw
    if name == "California Housing":
        p = os.path.join(RAW_DIR, "california_housing.csv")
        if os.path.exists(p):
            df = pd.read_csv(p)
            df['log_MedHouseVal'] = np.log(df['MedHouseVal'].clip(lower=0.01))
            return df
    return None


def get_available_datasets():
    """Return list of available dataset names."""
    names = []
    for name in ["Boston Housing", "Ames Housing (Real)", "California Housing",
                  "World Bank WDI (Real)", "Enterprise Survey"]:
        df = load_dataset(name)
        if df is not None and len(df) > 0:
            names.append(name)
    return names


# Dataset specifications: dep_var, predictors, weight_var, cluster_var
DATASET_SPECS = {
    "Boston Housing": {
        "dep_var": "medv",
        "predictors": ["crim", "zn", "indus", "rm", "age", "tax", "ptratio", "lstat"],
        "weight_var": "lstat",
        "cluster_var": None,
        "description": "506 suburbs in Boston. Median home value vs socioeconomic features.",
    },
    "Ames Housing (Real)": {
        "dep_var": "log_SalePrice",
        "predictors": ["GrLivArea", "LotArea", "OverallQual", "YearBuilt", "TotalBsmtSF", "GarageCars"],
        "weight_var": "GrLivArea",
        "cluster_var": None,
        "description": "2,930 homes in Ames, Iowa. Rich features on housing characteristics.",
    },
    "California Housing": {
        "dep_var": "log_MedHouseVal",
        "predictors": ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup"],
        "weight_var": "Population",
        "cluster_var": None,
        "description": "20,640 census tracts in California. Median house value prediction.",
    },
    "World Bank WDI (Real)": {
        "dep_var": "gdp_growth",
        "predictors": ["lag_gdp_pc", "investment_rate", "inflation", "log_pop"],
        "weight_var": None,
        "cluster_var": "country_id",
        "description": "Country-year panel from World Bank. GDP growth drivers.",
    },
    "Enterprise Survey": {
        "dep_var": "log_revenue",
        "predictors": ["log_capital", "log_labor", "firm_size_dummy"],
        "weight_var": "firm_size_dummy",
        "cluster_var": "country",
        "description": "5,000 firms across 30 countries. Firm production function.",
    },
}


###############################################################################
# HELPER: FIT MODELS
###############################################################################

def fit_ols(df, dep_var, predictors):
    """Fit OLS and return results."""
    data = df[[dep_var] + predictors].dropna()
    y = data[dep_var]
    X = sm.add_constant(data[predictors].astype(float))
    return sm.OLS(y, X).fit(), data


def fit_wls(df, dep_var, predictors, weight_var, weight_transform="1/x"):
    """Fit WLS with specified weights."""
    cols = list(dict.fromkeys([dep_var] + predictors + [weight_var]))
    data = df[cols].dropna()
    y = data[dep_var]
    X = sm.add_constant(data[predictors].astype(float))
    w_raw = np.asarray(data[weight_var]).astype(float).flatten()
    w_raw = np.clip(w_raw, 0.01, None)
    if weight_transform == "1/x":
        weights = 1.0 / w_raw
    elif weight_transform == "1/x²":
        weights = 1.0 / (w_raw ** 2)
    elif weight_transform == "1/√x":
        weights = 1.0 / np.sqrt(w_raw)
    else:
        weights = 1.0 / w_raw
    return sm.WLS(y, X, weights=weights).fit(), data


def fit_fgls(ols_model, df, dep_var, predictors):
    """Fit FGLS: estimate variance function from OLS residuals."""
    cols = list(dict.fromkeys([dep_var] + predictors))
    data = df[cols].dropna()
    y = data[dep_var]
    X = sm.add_constant(data[predictors].astype(float))
    resid_sq = np.asarray(ols_model.resid) ** 2
    log_resid_sq = np.log(np.clip(resid_sq, 1e-10, None))
    var_model = sm.OLS(log_resid_sq, X).fit()
    weights = 1.0 / np.exp(var_model.fittedvalues)
    weights = np.where(np.isfinite(weights) & (weights > 0), weights, 1.0)
    return sm.WLS(y, X, weights=weights).fit()


def run_heterosk_tests(model):
    """Run heteroskedasticity tests. Returns dict of results."""
    results = {}
    try:
        lm, lm_p, f, f_p = het_breuschpagan(model.resid, model.model.exog)
        results["Breusch-Pagan"] = {"stat": lm, "p_value": lm_p, "type": "LM"}
    except:
        pass
    try:
        lm, lm_p, f, f_p = het_white(model.resid, model.model.exog)
        results["White"] = {"stat": lm, "p_value": lm_p, "type": "LM"}
    except:
        pass
    try:
        f, f_p, _ = het_goldfeldquandt(model.model.endog, model.model.exog)
        results["Goldfeld-Quandt"] = {"stat": f, "p_value": f_p, "type": "F"}
    except:
        pass
    return results


def get_model_estimates(model, exog_names):
    """Safely return params, bse, tvalues, and pvalues as Series with exog_names."""
    names = list(exog_names)
    params = pd.Series(np.asarray(model.params), index=names)
    bse = pd.Series(np.asarray(model.bse), index=names)
    tvals = pd.Series(np.asarray(model.tvalues), index=names)
    pvals = pd.Series(np.asarray(model.pvalues), index=names)
    return params, bse, tvals, pvals


def compute_robust_se(model, cluster_groups=None):
    """Return dict of SE types."""
    se_dict = {"OLS": np.asarray(model.bse)}
    for hc in ["HC0", "HC1", "HC3"]:
        try:
            se_dict[hc] = np.asarray(model.get_robustcov_results(cov_type=hc).bse)
        except:
            pass
    if cluster_groups is not None:
        try:
            se_dict["Cluster"] = np.asarray(model.get_robustcov_results(
                cov_type="cluster", groups=cluster_groups
            ).bse)
        except:
            pass
    try:
        se_dict["HAC (NW)"] = np.asarray(model.get_robustcov_results(
            cov_type="HAC", maxlags=int(model.nobs**(1/3))
        ).bse)
    except:
        pass
    return se_dict


###############################################################################
# PAGE: METHODOLOGY
###############################################################################

def page_methodology():
    st.markdown('<div class="main-header">📐 Econometric Methodology</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Theoretical foundations of OLS, WLS, GLS, and diagnostic tests</div>',
                unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔹 OLS Assumptions", "🔸 WLS Derivation", "🔹 GLS & FGLS",
        "🔸 Robust Inference", "🔹 Diagnostic Tests"
    ])

    with tab1:
        st.markdown(r"""
### The Classical Linear Model

The standard linear regression model is:

$$y = X\beta + \varepsilon$$

where $y$ is an $n \times 1$ vector of observations, $X$ is $n \times k$ design matrix, 
$\beta$ is $k \times 1$ parameter vector, and $\varepsilon$ is $n \times 1$ error vector.

### OLS Assumptions (Gauss-Markov)

| # | Assumption | Mathematical Form | Consequence if Violated |
|---|-----------|-------------------|------------------------|
| A1 | **Linearity** | $y = X\beta + \varepsilon$ | Biased estimates |
| A2 | **Strict exogeneity** | $E[\varepsilon \mid X] = 0$ | Biased, inconsistent |
| A3 | **No multicollinearity** | $\text{rank}(X) = k$ | Cannot compute $\hat\beta$ |
| A4 | **Spherical errors** | $\text{Var}(\varepsilon \mid X) = \sigma^2 I_n$ | **Inefficient, wrong SEs** |
| A5 | **Normality** (optional) | $\varepsilon \sim N(0, \sigma^2 I_n)$ | Inexact finite-sample inference |

### The OLS Estimator

$$\hat{\beta}_{OLS} = (X'X)^{-1}X'y$$

Under A1–A4, OLS is **BLUE** (Best Linear Unbiased Estimator) by the Gauss-Markov Theorem.

### What Happens When A4 Fails?

When $\text{Var}(\varepsilon) = \Sigma \neq \sigma^2 I$:
- $\hat{\beta}_{OLS}$ remains **unbiased** but is **no longer efficient**
- The conventional variance formula $\hat{\sigma}^2(X'X)^{-1}$ is **biased**
- Confidence intervals and hypothesis tests become **invalid**
- t-statistics may be too large or too small → wrong conclusions
        """)

    with tab2:
        st.markdown(r"""
### Weighted Least Squares (WLS)

When errors are **heteroskedastic** with known structure:

$$\text{Var}(\varepsilon_i) = \sigma^2 / w_i$$

where $w_i > 0$ are known weights. The WLS estimator minimises:

$$\min_\beta \sum_{i=1}^{n} w_i (y_i - x_i'\beta)^2$$

**Solution:**

$$\hat{\beta}_{WLS} = (X'WX)^{-1} X'Wy$$

where $W = \text{diag}(w_1, \ldots, w_n)$.

### Equivalence to Transformed OLS

Define $W^{1/2} = \text{diag}(\sqrt{w_1}, \ldots, \sqrt{w_n})$. Then WLS is OLS on:

$$W^{1/2}y = W^{1/2}X\beta + W^{1/2}\varepsilon$$

The transformed errors have $\text{Var}(W^{1/2}\varepsilon) = \sigma^2 I$ ✓

### Practical Weight Choices

| Data Pattern | Weight Choice | Rationale |
|-------------|--------------|-----------|
| $\text{Var}(\varepsilon_i) \propto x_i^2$ | $w_i = 1/x_i^2$ | Variance grows with predictor |
| $\text{Var}(\varepsilon_i) \propto \hat{y}_i^2$ | $w_i = 1/\hat{y}_i^2$ | Variance grows with response |
| Group sizes $n_g$ | $w_i = n_g$ | Grouped data |
| Known from data | Estimate from $\hat\varepsilon^2$ | **FGLS approach** |
        """)

    with tab3:
        st.markdown(r"""
### Generalized Least Squares (GLS)

For the general case $\text{Var}(\varepsilon) = \sigma^2 \Sigma$ where $\Sigma$ is known and positive definite:

$$\hat{\beta}_{GLS} = (X'\Sigma^{-1}X)^{-1} X'\Sigma^{-1}y$$

**Properties:**
- GLS is **BLUE** under the generalised Gauss-Markov conditions
- When $\Sigma$ is diagonal → **WLS** (heteroskedasticity only)
- When $\Sigma$ has off-diagonal elements → accounts for **autocorrelation** or **cross-correlation**

### Common Covariance Structures

| Structure | $\Sigma$ Form | Application |
|-----------|--------------|-------------|
| **Heteroskedastic** | $\text{diag}(\sigma_1^2, \ldots, \sigma_n^2)$ | Cross-section |
| **AR(1)** | $\sigma^2/(1-\rho^2) \cdot \rho^{\lvert i-j\rvert}$ | Time series |
| **Random Effects** | $\sigma_\varepsilon^2 I + \sigma_\alpha^2 (\iota\iota')$ | Panel data |

### Feasible GLS (FGLS)

In practice, $\Sigma$ is **unknown** and must be estimated:

1. **Step 1:** Fit OLS, obtain residuals $\hat{\varepsilon}$
2. **Step 2:** Estimate $\hat{\Sigma}$ from $\hat{\varepsilon}$ (e.g., regress $\log(\hat\varepsilon^2)$ on $X$)
3. **Step 3:** Compute $\hat{\beta}_{FGLS} = (X'\hat{\Sigma}^{-1}X)^{-1} X'\hat{\Sigma}^{-1}y$

FGLS is **asymptotically efficient** but may perform poorly in small samples.
        """)

    with tab4:
        st.markdown(r"""
### Heteroskedasticity-Consistent (HC) Standard Errors

Instead of transforming the model, we can fix the SEs while keeping OLS estimates:

$$\hat{V}_{HC} = (X'X)^{-1} \left(\sum_{i=1}^n \hat{u}_i^2 x_i x_i'\right) (X'X)^{-1}$$

| Variant | Scaling | Notes |
|---------|---------|-------|
| **HC0** (White) | $\hat{u}_i^2$ | Original White (1980) |
| **HC1** | $\frac{n}{n-k}\hat{u}_i^2$ | Degrees-of-freedom correction |
| **HC2** | $\frac{\hat{u}_i^2}{1-h_{ii}}$ | Leverage-adjusted |
| **HC3** | $\frac{\hat{u}_i^2}{(1-h_{ii})^2}$ | **Recommended** for small samples |

### Cluster-Robust Standard Errors

For data clustered in $G$ groups (e.g., firms within countries):

$$\hat{V}_{CL} = (X'X)^{-1}\left(\sum_{g=1}^G X_g'\hat{u}_g\hat{u}_g'X_g\right)(X'X)^{-1}$$

This accounts for arbitrary within-cluster correlation.

### Newey-West HAC Standard Errors

For time-series data with autocorrelation:

$$\hat{V}_{HAC} = (X'X)^{-1}\hat{S}(X'X)^{-1}$$

where $\hat{S}$ uses a kernel-weighted sum of autocovariance matrices up to lag $L$.
        """)

    with tab5:
        st.markdown(r"""
### Diagnostic Tests for Heteroskedasticity

| Test | $H_0$ | Method | Best For |
|------|--------|--------|----------|
| **Breusch-Pagan** | Homoskedasticity | Regress $\hat{u}^2$ on $X$; LM test | Linear heteroskedasticity |
| **White** | Homoskedasticity | Regress $\hat{u}^2$ on $X$, $X^2$, cross-products | General (nonlinear) forms |
| **Goldfeld-Quandt** | Equal variance in subsamples | F-test on split samples | Monotonic heteroskedasticity |

### Decision Flowchart

```
OLS Estimation
    │
    ▼
Plot residuals vs fitted ──→ Visual pattern?
    │                              │
    ▼                              ▼ Yes
Run BP/White tests           Suspect heteroskedasticity
    │                              │
    ▼                              ▼
p < 0.05? ─── Yes ──→  Choose remedy:
    │                    ├── WLS (if variance structure known)
    │                    ├── FGLS (estimate variance function)
    │                    ├── HC robust SEs (leave estimates, fix SEs)
    │                    └── Log transform (may stabilise variance)
    ▼ No
OLS is valid ✓
```
        """)


###############################################################################
# PAGE: DATA EXPLORER
###############################################################################

def page_data_explorer():
    st.markdown('<div class="main-header">🗂️ Data Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Explore real-world datasets used in this project</div>',
                unsafe_allow_html=True)

    datasets = get_available_datasets()
    selected = st.selectbox("Select Dataset", datasets)
    df = load_dataset(selected)
    spec = DATASET_SPECS.get(selected, {})

    if df is None:
        st.error(f"Dataset '{selected}' not found.")
        return

    # Info cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Observations", f"{len(df):,}")
    col2.metric("Variables", f"{len(df.columns)}")
    col3.metric("Dependent Var", spec.get("dep_var", "—"))
    col4.metric("Predictors", str(len(spec.get("predictors", []))))

    st.info(f"**{selected}:** {spec.get('description', '')}")

    tab1, tab2, tab3 = st.tabs(["📋 Summary Statistics", "📊 Distributions", "🔗 Correlations"])

    with tab1:
        numeric_df = df.select_dtypes(include=[np.number])
        st.dataframe(numeric_df.describe().round(3).T, width="stretch")
        with st.expander("View Raw Data (first 100 rows)"):
            st.dataframe(df.head(100), width="stretch")

    with tab2:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        col_to_plot = st.selectbox("Variable to plot", num_cols, key="dist_var")
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(df, x=col_to_plot, nbins=50, title=f"Distribution of {col_to_plot}",
                               color_discrete_sequence=["#4C72B0"])
            fig.update_layout(height=400)
            st.plotly_chart(fig, width="stretch")
        with c2:
            fig = px.box(df, y=col_to_plot, title=f"Box Plot of {col_to_plot}",
                         color_discrete_sequence=["#DD8452"])
            fig.update_layout(height=400)
            st.plotly_chart(fig, width="stretch")

    with tab3:
        corr_cols = spec.get("predictors", num_cols[:8])
        dep = spec.get("dep_var")
        if dep and dep in df.columns:
            corr_cols = [dep] + [c for c in corr_cols if c in df.columns]
        else:
            corr_cols = [c for c in corr_cols if c in df.columns]
        corr_matrix = df[corr_cols].corr().round(3)
        fig = px.imshow(corr_matrix, text_auto=True, color_continuous_scale="RdBu_r",
                        title="Correlation Matrix", zmin=-1, zmax=1,
                        aspect="auto")
        fig.update_layout(height=500)
        st.plotly_chart(fig, width="stretch")

        # Scatter matrix of key vars
        if dep and dep in df.columns:
            scatter_vars = [dep] + [c for c in spec.get("predictors", [])[:3] if c in df.columns]
            fig = px.scatter_matrix(df[scatter_vars].dropna().sample(min(1000, len(df))),
                                    dimensions=scatter_vars, title="Scatter Matrix (key variables)")
            fig.update_layout(height=600)
            st.plotly_chart(fig, width="stretch")


###############################################################################
# PAGE: OLS & DIAGNOSTICS
###############################################################################

def page_ols_diagnostics():
    st.markdown('<div class="main-header">📈 OLS Estimation & Diagnostics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Fit OLS, test for heteroskedasticity, examine residuals</div>',
                unsafe_allow_html=True)

    datasets = get_available_datasets()
    selected = st.selectbox("Select Dataset", datasets, key="ols_ds")
    df = load_dataset(selected)
    spec = DATASET_SPECS.get(selected, {})

    if df is None:
        st.error("Dataset not found"); return

    dep_var = spec["dep_var"]
    predictors = [p for p in spec["predictors"] if p in df.columns]

    if not predictors or dep_var not in df.columns:
        st.error(f"Required columns missing. Available: {df.columns.tolist()}")
        return

    # Allow user to customise predictors
    with st.expander("⚙️ Customise Model Specification"):
        selected_preds = st.multiselect("Predictors", predictors, default=predictors)
        if selected_preds:
            predictors = selected_preds

    # Fit OLS
    model, data = fit_ols(df, dep_var, predictors)

    # Results
    tab1, tab2, tab3 = st.tabs(["📊 Regression Results", "🔍 Residual Diagnostics",
                                  "🧪 Heteroskedasticity Tests"])

    with tab1:
        col1, col2 = st.columns([2, 1])
        with col1:
            # Coefficient table
            coef_df = pd.DataFrame({
                "Variable": model.params.index,
                "Coefficient": model.params.values.round(6),
                "Std Error": model.bse.values.round(6),
                "t-statistic": model.tvalues.values.round(4),
                "P-value": model.pvalues.values.round(6),
            })
            st.dataframe(coef_df, width="stretch", hide_index=True)

        with col2:
            st.metric("R²", f"{model.rsquared:.4f}")
            st.metric("Adj R²", f"{model.rsquared_adj:.4f}")
            st.metric("F-statistic", f"{model.fvalue:.2f}")
            st.metric("N", f"{int(model.nobs)}")
            st.metric("AIC", f"{model.aic:.1f}")

        # Coefficient plot
        coef_plot = coef_df[coef_df["Variable"] != "const"].copy()
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=coef_plot["Variable"], y=coef_plot["Coefficient"],
            error_y=dict(type="data", array=1.96 * coef_plot["Std Error"].values),
            marker_color="#4C72B0", name="Coefficient ± 1.96 SE"
        ))
        fig.update_layout(title="OLS Coefficients with 95% CI", height=400,
                         yaxis_title="Coefficient")
        st.plotly_chart(fig, width="stretch")

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            # Residuals vs Fitted
            fig = px.scatter(x=model.fittedvalues, y=model.resid,
                           labels={"x": "Fitted Values", "y": "Residuals"},
                           title="Residuals vs Fitted Values",
                           opacity=0.4)
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            # Add LOWESS
            from statsmodels.nonparametric.smoothers_lowess import lowess
            sm_vals = lowess(model.resid, model.fittedvalues, frac=0.3)
            fig.add_trace(go.Scatter(x=sm_vals[:, 0], y=sm_vals[:, 1],
                                     mode="lines", name="LOWESS",
                                     line=dict(color="red", width=2)))
            fig.update_layout(height=450)
            st.plotly_chart(fig, width="stretch")

        with c2:
            # Q-Q Plot
            sorted_resid = np.sort(model.resid)
            theoretical = stats.norm.ppf(np.linspace(0.01, 0.99, len(sorted_resid)))
            fig = px.scatter(x=theoretical, y=sorted_resid,
                           labels={"x": "Theoretical Quantiles", "y": "Sample Quantiles"},
                           title="Q-Q Plot of Residuals", opacity=0.4)
            min_val = min(theoretical.min(), sorted_resid.min())
            max_val = max(theoretical.max(), sorted_resid.max())
            fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                                     mode="lines", line=dict(color="red", dash="dash"),
                                     name="45° line"))
            fig.update_layout(height=450)
            st.plotly_chart(fig, width="stretch")

        # Residuals vs each predictor
        st.subheader("Residuals vs Predictors")
        n_cols = min(3, len(predictors))
        cols = st.columns(n_cols)
        for i, pred in enumerate(predictors[:6]):
            with cols[i % n_cols]:
                if pred in data.columns:
                    fig = px.scatter(x=data[pred].values[:len(model.resid)], y=model.resid,
                                   labels={"x": pred, "y": "Residuals"},
                                   title=f"Resid vs {pred}", opacity=0.3)
                    fig.add_hline(y=0, line_dash="dash", line_color="red")
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, width="stretch")

    with tab3:
        st.subheader("Formal Heteroskedasticity Tests")
        tests = run_heterosk_tests(model)

        if tests:
            test_rows = []
            for name, res in tests.items():
                test_rows.append({
                    "Test": name,
                    "Statistic": round(res["stat"], 4),
                    "p-value": round(res["p_value"], 6),
                    "Type": res["type"],
                    "Reject H₀ (α=0.05)": "✅ Yes" if res["p_value"] < 0.05 else "❌ No",
                })
            test_df = pd.DataFrame(test_rows)
            st.dataframe(test_df, width="stretch", hide_index=True)

            # Interpretation
            bp = tests.get("Breusch-Pagan", {})
            if bp:
                if bp["p_value"] < 0.05:
                    st.error(f"🔴 **Heteroskedasticity detected!** Breusch-Pagan p = {bp['p_value']:.4f}. "
                             "OLS standard errors are unreliable. Use WLS, FGLS, or robust SEs.")
                else:
                    st.success(f"🟢 **No evidence of heteroskedasticity.** Breusch-Pagan p = {bp['p_value']:.4f}. "
                               "OLS inference appears valid.")

        # Squared residuals plot
        fig = px.scatter(x=model.fittedvalues, y=model.resid**2,
                        labels={"x": "Fitted Values", "y": "Squared Residuals"},
                        title="Squared Residuals vs Fitted (visual heteroskedasticity check)",
                        opacity=0.4, color_discrete_sequence=["#C44E52"])
        fig.update_layout(height=400)
        st.plotly_chart(fig, width="stretch")


###############################################################################
# PAGE: WLS / GLS ANALYSIS
###############################################################################

def page_wls_gls():
    st.markdown('<div class="main-header">⚖️ WLS / GLS / FGLS Estimation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Apply weighted and generalized least squares, compare with OLS</div>',
                unsafe_allow_html=True)

    datasets = get_available_datasets()
    selected = st.selectbox("Select Dataset", datasets, key="wls_ds")
    df = load_dataset(selected)
    spec = DATASET_SPECS.get(selected, {})

    if df is None:
        st.error("Dataset not found"); return

    dep_var = spec["dep_var"]
    predictors = [p for p in spec["predictors"] if p in df.columns]
    weight_var = spec.get("weight_var")
    cluster_var = spec.get("cluster_var")

    # Sidebar controls
    st.sidebar.markdown("### WLS/GLS Settings")

    # Weight variable selection
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != dep_var]
    if weight_var and weight_var in numeric_cols:
        default_idx = numeric_cols.index(weight_var)
    else:
        default_idx = 0
    weight_var = st.sidebar.selectbox("Weight Variable", numeric_cols, index=default_idx)
    weight_transform = st.sidebar.selectbox("Weight Transform", ["1/x", "1/x²", "1/√x"])

    # Fit models
    ols_model, data = fit_ols(df, dep_var, predictors)

    try:
        wls_model, _ = fit_wls(df, dep_var, predictors, weight_var, weight_transform)
    except:
        wls_model = None
        st.warning("WLS fitting failed. Check weight variable.")

    try:
        fgls_model = fit_fgls(ols_model, df, dep_var, predictors)
    except:
        fgls_model = None

    # Results comparison
    tab1, tab2, tab3 = st.tabs(["📊 Coefficient Comparison", "📐 SE Comparison",
                                  "📈 Weighted Residuals"])

    with tab1:
        # Build comparison table
        comp_rows = []
        models_dict = {"OLS": ols_model}
        if wls_model:
            models_dict[f"WLS ({weight_transform} {weight_var})"] = wls_model
        if fgls_model:
            models_dict["FGLS"] = fgls_model

        # Also compute robust
        ols_hc3 = ols_model.get_robustcov_results(cov_type="HC3")
        models_dict["OLS + HC3"] = ols_hc3

        var_names = list(getattr(ols_model.model, 'exog_names', [f"x{i}" for i in range(len(ols_model.params))]))

        for label, m in models_dict.items():
            params, bse, tval, pval = get_model_estimates(m, var_names)
            for var in var_names:
                comp_rows.append({
                    "Method": label, "Variable": var,
                    "Coefficient": round(float(params[var]), 6),
                    "Std Error": round(float(bse[var]), 6),
                    "t-stat": round(float(tval[var]), 4),
                    "p-value": round(float(pval[var]), 6),
                })
        comp_df = pd.DataFrame(comp_rows)
        pivot = comp_df.pivot_table(index="Variable", columns="Method",
                                     values=["Coefficient", "Std Error"], aggfunc="first")
        st.dataframe(pivot.round(6), width="stretch")

        # Grouped bar chart
        plot_data = comp_df[comp_df["Variable"] != "const"]
        fig = px.bar(plot_data, x="Variable", y="Coefficient", color="Method",
                    barmode="group", error_y="Std Error",
                    title="Coefficient Comparison: OLS vs WLS vs FGLS",
                    color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(height=500)
        st.plotly_chart(fig, width="stretch")

    with tab2:
        # SE comparison
        se_rows = []
        for i, var in enumerate(var_names):
            ols_se = float(np.asarray(ols_model.bse)[i])
            row = {"Variable": var, "OLS SE": round(ols_se, 6)}
            if wls_model:
                w_se = float(np.asarray(wls_model.bse)[i])
                row["WLS SE"] = round(w_se, 6)
                row["WLS/OLS Ratio"] = round(w_se / ols_se, 4) if ols_se > 0 else np.nan
            if fgls_model:
                f_se = float(np.asarray(fgls_model.bse)[i])
                row["FGLS SE"] = round(f_se, 6)
                row["FGLS/OLS Ratio"] = round(f_se / ols_se, 4) if ols_se > 0 else np.nan
            hc3_se = float(np.asarray(ols_hc3.bse)[i])
            row["HC3 SE"] = round(hc3_se, 6)
            row["HC3/OLS Ratio"] = round(hc3_se / ols_se, 4) if ols_se > 0 else np.nan
            se_rows.append(row)
        se_df = pd.DataFrame(se_rows)
        st.dataframe(se_df, width="stretch", hide_index=True)

        st.markdown("""
        **Interpretation Guide:**
        - Ratio < 1 → method produces **tighter** SEs than OLS (more efficient)
        - Ratio > 1 → method produces **wider** SEs (often the HC3 correction when OLS undercounts)
        - If heteroskedasticity is present, HC3 > OLS suggests OLS SEs are **too small**
        """)

    with tab3:
        if wls_model:
            c1, c2 = st.columns(2)
            with c1:
                fig = px.scatter(x=ols_model.fittedvalues, y=ols_model.resid,
                               title="OLS Residuals vs Fitted", opacity=0.4,
                               labels={"x": "Fitted", "y": "Residuals"})
                fig.add_hline(y=0, line_dash="dash", line_color="red")
                fig.update_layout(height=400)
                st.plotly_chart(fig, width="stretch")
            with c2:
                fig = px.scatter(x=wls_model.fittedvalues, y=wls_model.resid,
                               title="WLS Residuals vs Fitted", opacity=0.4,
                               labels={"x": "Fitted", "y": "Residuals"},
                               color_discrete_sequence=["#DD8452"])
                fig.add_hline(y=0, line_dash="dash", line_color="red")
                fig.update_layout(height=400)
                st.plotly_chart(fig, width="stretch")

            st.markdown("**Key insight:** If WLS is correctly specified, the residual plot should show "
                        "more uniform spread (homoskedastic pattern) compared to OLS.")


###############################################################################
# PAGE: ROBUST STANDARD ERRORS
###############################################################################

def page_robust_se():
    st.markdown('<div class="main-header">🛡️ Robust Standard Errors</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">HC0–HC3, cluster-robust, and Newey-West HAC standard errors</div>',
                unsafe_allow_html=True)

    datasets = get_available_datasets()
    selected = st.selectbox("Select Dataset", datasets, key="robust_ds")
    df = load_dataset(selected)
    spec = DATASET_SPECS.get(selected, {})

    if df is None:
        st.error("Dataset not found"); return

    dep_var = spec["dep_var"]
    predictors = [p for p in spec["predictors"] if p in df.columns]
    cluster_var = spec.get("cluster_var")

    model, data = fit_ols(df, dep_var, predictors)

    cluster_groups = None
    if cluster_var and cluster_var in df.columns:
        cluster_groups = df[cluster_var].values[:len(model.resid)]

    se_dict = compute_robust_se(model, cluster_groups)

    # Build table
    var_names = model.params.index.tolist()
    rows = []
    for i, var in enumerate(var_names):
        row = {"Variable": var, "Coefficient": round(model.params.iloc[i], 6)}
        for se_type, se_vals in se_dict.items():
            row[f"{se_type} SE"] = round(se_vals[i], 6)
        rows.append(row)
    se_df = pd.DataFrame(rows)
    st.dataframe(se_df, width="stretch", hide_index=True)

    # Ratio heatmap
    st.subheader("SE Ratios Relative to OLS")
    ratio_rows = []
    for i, var in enumerate(var_names):
        if var == "const":
            continue
        row = {"Variable": var}
        ols_se = se_dict["OLS"][i]
        for se_type, se_vals in se_dict.items():
            if se_type != "OLS":
                row[se_type] = round(se_vals[i] / ols_se, 4) if ols_se > 0 else np.nan
        ratio_rows.append(row)
    ratio_df = pd.DataFrame(ratio_rows).set_index("Variable")

    fig = px.imshow(ratio_df.values, x=ratio_df.columns.tolist(),
                    y=ratio_df.index.tolist(), text_auto=".3f",
                    color_continuous_scale="RdYlGn_r",
                    title="SE Ratio Heatmap (values > 1 = OLS underestimates uncertainty)",
                    zmin=0.5, zmax=2.0, aspect="auto")
    fig.update_layout(height=400)
    st.plotly_chart(fig, width="stretch")

    st.markdown("""
    **Reading the heatmap:**
    - 🟢 **Green (< 1.0):** Robust SE is smaller → OLS may overestimate uncertainty
    - 🟡 **Yellow (≈ 1.0):** Robust and OLS agree → homoskedastic errors likely
    - 🔴 **Red (> 1.0):** Robust SE is larger → OLS **underestimates** uncertainty (heteroskedasticity!)
    """)


###############################################################################
# PAGE: MONTE CARLO SIMULATION
###############################################################################

def page_simulation():
    st.markdown('<div class="main-header">🎲 Monte Carlo Simulation Study</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Compare OLS, WLS, GLS under controlled data-generating processes</div>',
                unsafe_allow_html=True)

    scenario = st.selectbox("Select Scenario", [
        "1️⃣ Heteroskedastic Errors",
        "2️⃣ AR(1) Autocorrelation",
        "3️⃣ Clustered Errors",
    ])

    # Common controls
    col1, col2, col3 = st.columns(3)
    with col1:
        n_reps = st.slider("Monte Carlo Replications", 100, 5000, 1000, 100,
                            help="More reps = more precise but slower")
    with col2:
        n_obs = st.slider("Sample Size (N)", 30, 500, 100, 10)
    with col3:
        true_beta = st.number_input("True β₁", value=2.0, step=0.5)

    if "1️⃣" in scenario:
        sim_heteroskedasticity(n_reps, n_obs, true_beta)
    elif "2️⃣" in scenario:
        sim_autocorrelation(n_reps, n_obs, true_beta)
    else:
        sim_clustered(n_reps, n_obs, true_beta)


def check_coverage(model, target=2.0, param_idx=1):
    """Check if the 95% confidence interval for param_idx covers the true value."""
    try:
        ci = np.asarray(model.conf_int(alpha=0.05))
        return int(ci[param_idx, 0] <= target <= ci[param_idx, 1])
    except Exception:
        return 0


def sim_heteroskedasticity(reps, n, true_beta):
    st.markdown("### Scenario 1: Heteroskedastic Errors")
    st.latex(r"y_i = 1 + \beta_1 x_i + \varepsilon_i, \quad \varepsilon_i \sim N(0, x_i^{2\varphi})")

    phi_values = st.multiselect("φ values (heteroskedasticity intensity)",
                                [0.0, 0.25, 0.5, 0.75, 1.0, 1.5],
                                default=[0.0, 0.5, 1.0, 1.5])

    if st.button("🚀 Run Simulation", key="run_s1"):
        progress = st.progress(0, text="Running simulations...")
        results = []
        total = len(phi_values)

        for idx, phi in enumerate(phi_values):
            rng = np.random.default_rng(42)
            betas = {"OLS": [], "WLS (correct)": [], "WLS (misspec)": []}
            covers = {"OLS": 0, "WLS (correct)": 0, "WLS (misspec)": 0}

            for r in range(reps):
                x = rng.uniform(0.5, 10, n)
                eps = rng.normal(0, x**phi)
                y = 1 + true_beta * x + eps
                X = sm.add_constant(x)

                ols = sm.OLS(y, X).fit()
                betas["OLS"].append(ols.params[1])
                covers["OLS"] += check_coverage(ols, true_beta)

                w_c = 1.0 / (x**(2*phi)) if phi > 0 else np.ones(n)
                wls_c = sm.WLS(y, X, weights=w_c).fit()
                betas["WLS (correct)"].append(wls_c.params[1])
                covers["WLS (correct)"] += check_coverage(wls_c, true_beta)

                wls_m = sm.WLS(y, X, weights=1.0/(x**2)).fit()
                betas["WLS (misspec)"].append(wls_m.params[1])
                covers["WLS (misspec)"] += check_coverage(wls_m, true_beta)

            for method in betas:
                b = np.array(betas[method])
                results.append({
                    "φ": phi, "Method": method,
                    "Mean β̂₁": round(np.mean(b), 4),
                    "Bias": round(np.mean(b) - true_beta, 4),
                    "Variance": round(np.var(b, ddof=1), 6),
                    "RMSE": round(np.sqrt(np.mean((b - true_beta)**2)), 4),
                    "95% Coverage": round(covers[method] / reps, 4),
                })

            progress.progress((idx + 1) / total,
                              text=f"Completed φ = {phi} ({idx+1}/{total})")

        progress.empty()
        results_df = pd.DataFrame(results)

        # Display results
        st.dataframe(results_df, width="stretch", hide_index=True)

        # Plots
        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(results_df, x="φ", y="RMSE", color="Method", barmode="group",
                        title="RMSE by Method and Heteroskedasticity Level",
                        color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(height=450)
            st.plotly_chart(fig, width="stretch")
        with c2:
            fig = px.line(results_df, x="φ", y="95% Coverage", color="Method",
                         markers=True, title="95% CI Coverage Rate",
                         color_discrete_sequence=px.colors.qualitative.Set2)
            fig.add_hline(y=0.95, line_dash="dash", line_color="grey",
                         annotation_text="Nominal 95%")
            fig.update_layout(height=450, yaxis_range=[0.7, 1.02])
            st.plotly_chart(fig, width="stretch")

        # Key findings
        st.markdown("### 🔑 Key Findings")
        if len(phi_values) > 1:
            ols_cov_zero = results_df[(results_df["Method"]=="OLS") &
                                      (results_df["φ"]==0.0)]["95% Coverage"].values
            ols_cov_max = results_df[(results_df["Method"]=="OLS") &
                                     (results_df["φ"]==max(phi_values))]["95% Coverage"].values
            if len(ols_cov_zero) and len(ols_cov_max):
                st.markdown(f"""
                - Under **homoskedasticity (φ=0):** All methods perform similarly, coverage ≈ {ols_cov_zero[0]:.1%}
                - Under **strong heteroskedasticity (φ={max(phi_values)}):** OLS coverage drops to {ols_cov_max[0]:.1%}
                - **WLS with correct weights** maintains ≈95% coverage across all settings
                - **Misspecified WLS** can still improve over OLS but isn't optimal
                """)


def sim_autocorrelation(reps, n, true_beta):
    st.markdown("### Scenario 2: AR(1) Autocorrelation")
    st.latex(r"y_t = 1 + \beta_1 x_t + \varepsilon_t, \quad \varepsilon_t = \rho\varepsilon_{t-1} + u_t")

    rho_values = st.multiselect("ρ values (autocorrelation parameter)",
                                [0.0, 0.3, 0.5, 0.6, 0.8, 0.9],
                                default=[0.0, 0.3, 0.6, 0.9])

    if st.button("🚀 Run Simulation", key="run_s2"):
        progress = st.progress(0, text="Running simulations...")
        results = []

        for idx, rho in enumerate(rho_values):
            rng = np.random.default_rng(42)
            betas = {"OLS": [], "GLS (known ρ)": [], "FGLS": []}
            covers = {"OLS": 0, "GLS (known ρ)": 0, "FGLS": 0}

            for r in range(reps):
                x = rng.uniform(0, 10, n)
                u = rng.normal(0, 1, n)
                eps = np.zeros(n)
                eps[0] = u[0]
                for t in range(1, n):
                    eps[t] = rho * eps[t-1] + u[t]
                y = 1 + true_beta * x + eps
                X = sm.add_constant(x)

                ols = sm.OLS(y, X).fit()
                betas["OLS"].append(ols.params[1])
                covers["OLS"] += check_coverage(ols, true_beta)

                # GLS known rho
                if abs(rho) > 1e-6:
                    sf = np.sqrt(1 - rho**2)
                    y_s = np.concatenate([[sf*y[0]], y[1:] - rho*y[:-1]])
                    X_s = np.vstack([sf*X[0], X[1:] - rho*X[:-1]])
                    gls = sm.OLS(y_s, X_s).fit()
                else:
                    gls = ols
                betas["GLS (known ρ)"].append(gls.params[1])
                covers["GLS (known ρ)"] += check_coverage(gls, true_beta)

                # FGLS
                resid = ols.resid
                rho_hat = np.corrcoef(resid[:-1], resid[1:])[0, 1]
                if abs(rho_hat) > 1e-6:
                    sf_h = np.sqrt(max(1 - rho_hat**2, 0.01))
                    y_f = np.concatenate([[sf_h*y[0]], y[1:] - rho_hat*y[:-1]])
                    X_f = np.vstack([sf_h*X[0], X[1:] - rho_hat*X[:-1]])
                    fgls = sm.OLS(y_f, X_f).fit()
                else:
                    fgls = ols
                betas["FGLS"].append(fgls.params[1])
                covers["FGLS"] += check_coverage(fgls, true_beta)

            for method in betas:
                b = np.array(betas[method])
                results.append({
                    "ρ": rho, "Method": method,
                    "Mean β̂₁": round(np.mean(b), 4),
                    "Bias": round(np.mean(b) - true_beta, 4),
                    "Variance": round(np.var(b, ddof=1), 6),
                    "RMSE": round(np.sqrt(np.mean((b - true_beta)**2)), 4),
                    "95% Coverage": round(covers[method] / reps, 4),
                })
            progress.progress((idx+1)/len(rho_values),
                              text=f"Completed ρ = {rho}")

        progress.empty()
        results_df = pd.DataFrame(results)
        st.dataframe(results_df, width="stretch", hide_index=True)

        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(results_df, x="ρ", y="RMSE", color="Method", barmode="group",
                        title="RMSE by Autocorrelation Level",
                        color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig, width="stretch")
        with c2:
            fig = px.line(results_df, x="ρ", y="95% Coverage", color="Method",
                         markers=True, title="95% CI Coverage Rate",
                         color_discrete_sequence=px.colors.qualitative.Set2)
            fig.add_hline(y=0.95, line_dash="dash", line_color="grey")
            fig.update_layout(yaxis_range=[0.7, 1.02])
            st.plotly_chart(fig, width="stretch")


def sim_clustered(reps, n_per_cluster, true_beta):
    st.markdown("### Scenario 3: Clustered Errors")
    st.latex(r"y_{ic} = 1 + \beta_1 x_{ic} + \alpha_c + \varepsilon_{ic}, \quad \alpha_c \sim N(0, \tau^2)")

    tau_values = st.multiselect("τ² values (cluster variance)",
                                [0.0, 0.5, 1.0, 2.0, 4.0],
                                default=[0.0, 0.5, 1.0, 2.0])
    G = st.slider("Number of clusters (G)", 10, 100, 30, 5)

    if st.button("🚀 Run Simulation", key="run_s3"):
        progress = st.progress(0, text="Running simulations...")
        results = []
        N_total = G * n_per_cluster
        clusters = np.repeat(np.arange(G), n_per_cluster)

        for idx, tau_sq in enumerate(tau_values):
            rng = np.random.default_rng(42)
            betas = {"OLS": [], "OLS + Cluster SE": [], "RE-GLS": []}
            covers = {"OLS": 0, "OLS + Cluster SE": 0, "RE-GLS": 0}

            for r in range(reps):
                x = rng.uniform(0, 10, N_total)
                alpha = np.repeat(rng.normal(0, np.sqrt(tau_sq), G) if tau_sq > 0
                                  else np.zeros(G), n_per_cluster)
                eps = rng.normal(0, 1, N_total)
                y = 1 + true_beta * x + alpha + eps
                X = sm.add_constant(x)

                ols = sm.OLS(y, X).fit()
                betas["OLS"].append(ols.params[1])
                covers["OLS"] += check_coverage(ols, true_beta)

                try:
                    cl = ols.get_robustcov_results(cov_type="cluster", groups=clusters)
                    betas["OLS + Cluster SE"].append(cl.params[1])
                    covers["OLS + Cluster SE"] += check_coverage(cl, true_beta)
                except:
                    betas["OLS + Cluster SE"].append(ols.params[1])

                # RE-GLS
                try:
                    resid = ols.resid
                    grp_means = np.array([resid[clusters==g].mean() for g in range(G)])
                    sig_bw = np.var(grp_means, ddof=1)
                    sig_wn = np.mean([np.var(resid[clusters==g], ddof=1) for g in range(G)])
                    sig_a2 = max(sig_bw - sig_wn / n_per_cluster, 0)
                    theta = 1 - np.sqrt(sig_wn / (sig_wn + n_per_cluster * sig_a2)) if sig_a2 > 0 else 0

                    y_s = y.copy(); X_s = X.copy()
                    for g in range(G):
                        mask = clusters == g
                        y_s[mask] = y[mask] - theta * y[mask].mean()
                        X_s[mask] = X[mask] - theta * X[mask].mean(axis=0)
                    re = sm.OLS(y_s, X_s).fit()
                    betas["RE-GLS"].append(re.params[1])
                    covers["RE-GLS"] += check_coverage(re, true_beta)
                except:
                    betas["RE-GLS"].append(ols.params[1])

            for method in betas:
                b = np.array(betas[method])
                results.append({
                    "τ²": tau_sq, "Method": method,
                    "Mean β̂₁": round(np.mean(b), 4),
                    "Bias": round(np.mean(b) - true_beta, 4),
                    "Variance": round(np.var(b, ddof=1), 6),
                    "RMSE": round(np.sqrt(np.mean((b - true_beta)**2)), 4),
                    "95% Coverage": round(covers[method] / reps, 4),
                })
            progress.progress((idx+1)/len(tau_values),
                              text=f"Completed τ² = {tau_sq}")

        progress.empty()
        results_df = pd.DataFrame(results)
        st.dataframe(results_df, width="stretch", hide_index=True)

        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(results_df, x="τ²", y="RMSE", color="Method", barmode="group",
                        title="RMSE by Cluster Variance",
                        color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig, width="stretch")
        with c2:
            fig = px.line(results_df, x="τ²", y="95% Coverage", color="Method",
                         markers=True, title="95% CI Coverage Rate",
                         color_discrete_sequence=px.colors.qualitative.Set2)
            fig.add_hline(y=0.95, line_dash="dash", line_color="grey")
            fig.update_layout(yaxis_range=[0.5, 1.05])
            st.plotly_chart(fig, width="stretch")

        st.markdown("### 🔑 Key Findings")
        st.markdown("""
        - **OLS with naïve SEs** drastically **over-rejects** as cluster variance (τ²) increases
        - **Cluster-robust SEs** restore correct coverage by accounting for within-group correlation
        - **RE-GLS** can improve efficiency by explicitly modelling the random effect structure
        """)


###############################################################################
# PAGE: FULL COMPARISON
###############################################################################

def page_full_comparison():
    st.markdown('<div class="main-header">🏆 Full Method Comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Side-by-side OLS vs WLS vs FGLS vs Robust across all datasets</div>',
                unsafe_allow_html=True)

    all_results = []
    datasets = get_available_datasets()

    for ds_name in datasets:
        df = load_dataset(ds_name)
        spec = DATASET_SPECS.get(ds_name, {})
        if df is None or "dep_var" not in spec:
            continue

        dep_var = spec["dep_var"]
        predictors = [p for p in spec["predictors"] if p in df.columns]
        weight_var = spec.get("weight_var")
        cluster_var = spec.get("cluster_var")

        if not predictors or dep_var not in df.columns:
            continue

        try:
            ols, data = fit_ols(df, dep_var, predictors)
            row = {
                "Dataset": ds_name,
                "N": int(ols.nobs),
                "OLS R²": round(ols.rsquared, 4),
            }

            # Het test
            tests = run_heterosk_tests(ols)
            bp = tests.get("Breusch-Pagan", {})
            row["BP p-value"] = round(bp.get("p_value", np.nan), 4)
            row["Heteroskedastic?"] = "Yes" if bp.get("p_value", 1) < 0.05 else "No"

            # WLS
            if weight_var and weight_var in df.columns:
                try:
                    wls, _ = fit_wls(df, dep_var, predictors, weight_var)
                    row["WLS R²"] = round(wls.rsquared, 4)
                except:
                    row["WLS R²"] = "—"
            else:
                row["WLS R²"] = "—"

            # FGLS
            try:
                fgls = fit_fgls(ols, df, dep_var, predictors)
                row["FGLS R²"] = round(fgls.rsquared, 4)
            except:
                row["FGLS R²"] = "—"

            # HC3 ratio (median across predictors)
            try:
                hc3 = ols.get_robustcov_results(cov_type="HC3")
                ratios = hc3.bse.values[1:] / ols.bse.values[1:]
                row["Median HC3/OLS Ratio"] = round(np.median(ratios), 3)
            except:
                row["Median HC3/OLS Ratio"] = "—"

            all_results.append(row)
        except Exception as e:
            st.warning(f"Error processing {ds_name}: {e}")

    if all_results:
        summary_df = pd.DataFrame(all_results)
        st.dataframe(summary_df, width="stretch", hide_index=True)

        st.markdown("""
        ### Interpretation Guide

        | Metric | What It Tells You |
        |--------|-------------------|
        | **BP p-value < 0.05** | Heteroskedasticity present → WLS/GLS needed |
        | **WLS R² > OLS R²** | WLS achieves better fit (but R² comparison not always meaningful for WLS) |
        | **HC3/OLS Ratio > 1** | OLS understates SEs → inference is anti-conservative |
        | **HC3/OLS Ratio < 1** | OLS overstates SEs → inference is conservative |
        """)


###############################################################################
# MAIN APP
###############################################################################

def main():
    st.sidebar.markdown("# 📊 GLS/WLS Dashboard")
    st.sidebar.markdown("**Econometrics Project**")
    st.sidebar.markdown("Ankan Bera, Nov 2026")
    st.sidebar.markdown("---")

    page = st.sidebar.radio("Navigate", [
        "📐 Methodology",
        "🗂️ Data Explorer",
        "📈 OLS & Diagnostics",
        "⚖️ WLS / GLS Analysis",
        "🛡️ Robust Standard Errors",
        "🎲 Monte Carlo Simulation",
        "🏆 Full Comparison",
    ])

    if "Methodology" in page:
        page_methodology()
    elif "Data Explorer" in page:
        page_data_explorer()
    elif "OLS" in page:
        page_ols_diagnostics()
    elif "WLS" in page:
        page_wls_gls()
    elif "Robust" in page:
        page_robust_se()
    elif "Monte Carlo" in page:
        page_simulation()
    elif "Full Comparison" in page:
        page_full_comparison()


if __name__ == "__main__":
    main()

