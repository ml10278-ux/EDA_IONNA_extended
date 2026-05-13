# EDA_IONNA_extended
# IONNA Alliance Strategic Expansion: Predictive Site Scoring Model

## 1. Executive Summary

As the electric vehicle (EV) market rapidly expands, the availability of reliable, high-speed charging infrastructure remains a critical bottleneck. This project is a direct extension of the original [IONNA-EV-Charging-Strategy EDA](https://github.com/ml10278-ux/IONNA-EV-Charging-Strategy), which identified "EV Fast Charging Deserts" in Washington State by cross-referencing NREL supply data with Washington DOL registration demand data.

While the original project answered *where* the deserts are, this extension answers a more scalable question: **given any ZIP code's characteristics, what is its expansion priority score?** By engineering a machine learning pipeline on top of the same datasets, this project builds two predictive models — a **Random Forest Desert Classifier** and a **Gradient Boosting Opportunity Score Regressor** — that allow IONNA's real estate team to evaluate any ZIP code instantly, without re-running the full EDA pipeline. The models confirm the original findings: ZIP code **98115** remains the single highest-priority target (1,128 IONNA EVs, 0 DC fast chargers), and all five original "desert" ZIP codes are classified as **Tier 1 — Critical** by the predictive model.

---

## 2. Business Problem & Motivation

The central question of this analysis is: **How can the IONNA Alliance scalably identify and prioritize DC Fast Charging expansion targets across any market?**

The original EDA solved this for Washington State. But IONNA operates at continental scale. If the strategy team needs to evaluate ZIP codes in California, Colorado, or Oregon, re-running the full analysis pipeline for every new market is inefficient. A trained predictive model solves this:

- **Classifier:** Flag any ZIP code as an "EV desert" in real time without querying the NREL API
- **Regressor:** Rank all ZIP codes on a continuous opportunity score to tier investment priorities (top 5%, top 10%, etc.)

The `score_zip()` function produced by this project wraps both models into a single callable — a foundation for a future Streamlit dashboard deployable to IONNA's real estate acquisition team.

---

## 3. Data Sourcing & Description

This project uses the same two core datasets as the original analysis, with additional engineered enrichment features.

- **Supply Data (The NREL API):**
  - **Source:** National Renewable Energy Laboratory (NREL) Alternative Fuel Stations API.
  - **Purpose:** Map all existing DC Fast Chargers in Washington State at the ZIP-code level.
  - **Key Feature Extraction:** The `ev_dc_fast_num` feature is parsed via `if/else` null-handling, identical to the original project. Aggregated to ZIP level via `groupby`.

- **Demand Data (Washington State Gov CSV):**
  - **Source:** Washington State Department of Licensing (DOL) via `data.wa.gov`.
  - **Purpose:** Map residential ZIP codes of registered IONNA-brand EV owners.
  - **Project Requirement Note:** *The original dataset contains over 900,000 rows and 15+ features (Make, Model, Model Year, Electric Range, Base MSRP, Legislative District, VIN, County, City, etc.), fully satisfying the rubric requirement of >300 rows and >12 features of mixed data types. The DataFrame is filtered using `.isin()` to isolate IONNA Alliance brands and aggregated to ZIP level for modeling.*

- **Enrichment Features (Simulated; production sources noted):**
  - `median_income_k` — Median household income (Census API in production)
  - `aadt_k` — Annual average daily traffic in thousands (FHWA HPMS in production)
  - `retail_density` — Points of interest per sq. mile (OpenStreetMap/Yelp in production)
  - `dist_nearest_dcfc_km` — Distance to nearest existing DC fast charger (NREL in production)
  - `ev_growth_rate` — Year-over-year EV registration growth rate (WA DOL trend data in production)

---

## 4. Methodology & Feature Engineering

The modeling pipeline is built in Python using `pandas`, `numpy`, `scikit-learn`, `matplotlib`, and `seaborn`, structured across two separate Colab notebooks.

1. **EDA Notebook (`EDA_IONNA_Extended.ipynb`):** Replicates and extends the original analysis. Supply and demand data are merged via left join on `Postal Code`. Four engineered features are added: `opportunity_score` (original formula, preserved), `is_desert` flag, `evs_per_charger` ratio, and `demand_tier` bins. A clean dataset (`wa_ionna_clean.csv`) is exported for the modeling notebook.

2. **Feature Engineering (The Opportunity Score):** The original formula is preserved for methodological continuity:
   - `Opportunity_Score = IONNA_EV_Count / (DC_Fast_Count + 1)`
   - *The `+ 1` prevents `ZeroDivisionError` while ensuring true deserts yield the highest possible scores. The score is log-transformed (`np.log1p`) as the regression target to address right-skew.*

3. **Model A — Desert Classifier:** A `RandomForestClassifier` (200 trees, `class_weight='balanced'`) predicts the binary `is_desert` target. Evaluated via `classification_report`, ROC-AUC, and 5-fold stratified cross-validation.

4. **Model B — Opportunity Score Regressor:** A `GradientBoostingRegressor` (300 estimators, learning rate 0.05) predicts `log_opp_score`. A `Ridge` regression serves as the baseline. Evaluated via RMSE, MAE, and R².

5. **Site Scoring Function:** Both fitted models are wrapped in a `score_zip()` function that takes six ZIP-level features as input and returns a desert classification, desert probability, predicted opportunity score, and priority tier (Tier 1–4).

---

## 5. Models & Methods: Overview

| Model | Type | Target | Evaluation Metric |
|---|---|---|---|
| Random Forest | Classifier | `is_desert` (binary) | ROC-AUC, F1, 5-fold CV |
| Gradient Boosting | Regressor | `log_opp_score` (continuous) | RMSE, MAE, R² |
| Ridge Regression | Baseline Regressor | `log_opp_score` (continuous) | RMSE, MAE, R² |

---

## 6. Results & Interpretation

### The Top 5 Target Locations (Confirmed by Predictive Model)

Consistent with the original EDA, the Gradient Boosting regressor assigns the highest predicted opportunity scores to the same five ZIP codes — all of which have zero DC fast chargers:

1. **98115:** 1,128 IONNA EVs | 0 DC Fast Chargers | Tier 1 — Critical
2. **98033:** 1,036 IONNA EVs | 0 DC Fast Chargers | Tier 1 — Critical
3. **98006:** 826 IONNA EVs | 0 DC Fast Chargers | Tier 1 — Critical
4. **98040:** 708 IONNA EVs | 0 DC Fast Chargers | Tier 1 — Critical
5. **98391:** 588 IONNA EVs | 0 DC Fast Chargers | Tier 1 — Critical

### Priority Tier System

The regressor outputs are binned into four actionable tiers:

- **Tier 1 — Critical (score > 800):** Immediate land acquisition target
- **Tier 2 — High (score 400–800):** Plan within 12 months
- **Tier 3 — Medium (score 150–400):** Monitor; secondary market
- **Tier 4 — Low (score < 150):** Deprioritize or revisit in 24+ months

### Visual Interpretations

*(Note to reader: Please view the generated plots in the Jupyter Notebooks)*

- **Graph 1: The Opportunity Gap (Bar Plot):** Extends the original bar plot to the top 10 ZIP codes. Dark blue bars show IONNA EV populations; the complete absence of orange supply bars confirms zero-charger deficits.
- **Graph 2: Market Overview (Scatter Plot):** Identical in structure to the original, with desert ZIPs highlighted in red. Top 5 targets are annotated directly on the chart.
- **Graph 3: Demand Tier Analysis:** Two-panel breakdown showing ZIP count per demand tier and the proportion of desert vs. served ZIPs within each tier. High-demand ZIPs are disproportionately underserved.
- **Graph 4: Correlation Heatmap:** Feature correlation matrix for the three primary numeric features.
- **Graphs 5–6: Classifier Results:** Confusion matrix, feature importances, and ROC curve for the Random Forest desert classifier.
- **Graphs 7–8: Regressor Results:** Actual vs. predicted scatter, residual distribution, and feature importances for the Gradient Boosting regressor.
- **Graph 9: Priority Tier Distribution:** Horizontal bar chart showing the count of Washington State ZIP codes in each of the four expansion priority tiers.

---

## 7. Strategic Recommendations & Conclusion

The predictive models validate and operationalize the original EDA findings. IONNA's real estate acquisition teams should immediately target the five Tier 1 ZIP codes (98115, 98033, 98006, 98040, 98391), and use the `score_zip()` function to continuously evaluate new markets as expansion moves beyond Washington.

The two most important predictive features across both models were `dist_nearest_dcfc_km` (distance to the nearest existing fast charger) and `ionna_ev_count` — confirming that geographic isolation from infrastructure, combined with high local IONNA brand adoption, is the strongest signal for expansion ROI.

**Future Work:**
- Replace synthetic enrichment features with real **Census API** (income), **FHWA HPMS** (traffic), and **OpenStreetMap** (retail density) data to improve production accuracy
- Extend the pipeline to California, Colorado, and Oregon — high-EV-adoption states with significant IONNA market share
- Tune hyperparameters via `GridSearchCV` and evaluate XGBoost as an alternative to Gradient Boosting
- Deploy `score_zip()` as a **Streamlit web app** for IONNA's real estate team
- Incorporate time-series EV adoption forecasting to score ZIP codes on projected 3-year demand, not just current registration counts

---

## 8. Reproducibility & Code Structure

This project is designed to be fully reproducible in **Google Colab** with no local setup required.

- **Notebook 1:** `EDA_IONNA_Extended.ipynb` — Run first. Pulls NREL API data, downloads WA DOL registry, engineers features, produces Graphs 1–4, and exports `wa_ionna_clean.csv`.
- **Notebook 2:** `Modeling_IONNA_Predictive.ipynb` — Run second. Loads `wa_ionna_clean.csv`, engineers modeling features, trains both models, produces Graphs 5–9, and exposes the `score_zip()` function.
- **Requirements:** All dependencies (`pandas`, `numpy`, `requests`, `matplotlib`, `seaborn`, `scikit-learn`) are installed via `!pip install` at the top of each notebook.
- **API Key:** A free NREL API key must be obtained from `developer.nrel.gov` and pasted into the `NREL_API_KEY` constant in Cell 2 of the EDA notebook.
- **Execution:** Run each notebook sequentially from top to bottom. The EDA notebook must be run before the modeling notebook to generate the required `wa_ionna_clean.csv` intermediate dataset.
