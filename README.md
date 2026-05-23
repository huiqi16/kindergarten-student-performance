# Kindergarten Achievement Prediction | Project STAR

> Predicting early academic performance using student demographics, teacher attributes, and school context. Built on Tennessee's landmark **Project STAR** randomised controlled trial dataset.

---

## Problem Statement

Early childhood achievement is a strong predictor of long-term outcomes including earnings, college attendance, and neighbourhood quality (Chetty et al., 2011). This project asks: **which factors most strongly predict kindergarten reading and math scores, and which ML models best capture those relationships?**

Using a cross-section of ~5,000 kindergarten students, we compare 9 model families and apply SHAP-based feature importance analysis to identify the key drivers of performance.

---

## Results

| Model | MSE (Reading) | MSE (Math) | Combined MSE |
|---|---|---|---|
| OLS (Baseline) | 624.73 | 1841.26 | 1232.998 |
| LASSO | 627.17 | 1830.82 | 1228.99 |
| Ridge | 627.14 | 1830.91 | 1229.02 |
| Elastic Net | 627.17 | 1830.13 | 1228.65 |
| Neural Network (MLP) | 694.39 | 1800.46 | 1247.42 |
| Decision Tree | 801.29 | 2694.71 | 1748.00 |
| Random Forest | 545.78 | 1659.12 | 1102.45 |
| LightGBM | 467.42 | 1514.18 | 990.80 |
| **XGBoost ✅** | **467.45** | **1513.21** | **990.33** |

**XGBoost achieved the lowest combined MSE of 990.33**, a ~20% improvement over Random Forest and ~24% improvement over the linear baseline. It was selected as the final model.

---

## Key Findings

- **Socioeconomic status** (proxied by free-lunch eligibility) is the single strongest predictor for both reading and math — confirmed by both XGBoost feature importance and SHAP analysis
- **School ID** captures large between-school quality variance, ranking as the top feature by mean absolute SHAP value (6.67 for reading, 11.20 for math)
- **Teacher experience** has a consistent positive effect (SHAP mean abs: 2.68 reading, 3.90 math)
- **Small class size** positively impacts outcomes — directly consistent with the original Project STAR experimental findings
- **School location** interacts differently by subject: rural schools show stronger reading performance; suburban schools show better math outcomes
- **Neural networks underperformed** linear baselines, likely due to the limited dataset size (~5k rows) combined with sparse one-hot encoded inputs
- **Regularised linear models** (LASSO, Ridge, Elastic Net) outperformed OLS, confirming multicollinearity among predictors

---

## 📊 SHAP Feature Importance (XGBoost)

Top grouped drivers ranked by mean absolute SHAP value:

| Rank | Feature | Reading SHAP | Math SHAP |
|---|---|---|---|
| 1 | School ID | 6.67 | 11.20 |
| 2 | Lunch (SES proxy) | 2.81 | 4.23 |
| 3 | Teacher Experience | 2.68 | 3.90 |
| 4 | Ethnicity | 1.35 | 3.23 |
| 5 | Gender (male) | 2.25 | 2.67 |
| 6 | Class type | 1.31 | 2.37 |
| 7 | School location | 0.63 | 0.92 |

---

## Repo Structure

```
├── data/
|   ├── kinder_clean.csv           # cleaned dataset (4,463 rows, 12 features)
│   └── sample_kinder_data.csv       
│
├── notebooks/
│   ├── 01_eda.ipynb              # missing values, distributions, bivariate plots, correlation (r=0.71)
│   ├── 02_preprocessing.ipynb    # cleaning pipeline, outlier removal, 80/20 train-test split
│   ├── 03_models.ipynb           # all models trained + compared, GridSearchCV tuning
│   └── 04_feature_importance.ipynb  # XGBoost importances + grouped SHAP analysis
│
├── src/
│   ├── preprocessing.py          # clean_data(), remove_outliers_iqr(), prepare_data()
│   └── models.py                 # training functions for all 9 model families
│
├── Report.pdf
├── requirements.txt
└── README.md
```

---

## Methodology

### Data Preprocessing
- **Median imputation** for numeric variables; `'unknown'` category for missing categoricals
- Sparse categories collapsed; `ladder` and `schooldistrict_id` dropped (unstable/redundant)
- One-hot encoding for all categorical features
- **StandardScaler** applied within a scikit-learn pipeline (fit on training data only to prevent data leakage)
- **80/20 train-test split**

### Modelling Strategy
All models trained twice — once for `score_read`, once for `score_math`. Performance evaluated on the held-out 20% test set using MSE. Combined MSE = mean of both individual MSEs.

Models compared: OLS, LASSO, Ridge, Elastic Net, Decision Tree, Random Forest, Neural Network (MLP), LightGBM, XGBoost.

---

## Quickstart

```bash
git clone https://github.com/huiqi16/kindergarten-star
cd kindergarten-star
pip install -r requirements.txt
jupyter notebook 01_eda.ipynb
```

---

## Dataset

Derived from **Project STAR** (Student/Teacher Achievement Ratio), a randomised controlled trial in Tennessee studying the effect of class size on early academic performance. Variables span three conceptual levels:

- **Student demographics**: `gender`, `ethnicity`, `birth`, `lunch`
- **Teacher attributes**: `degree`, `experience`
- **School/classroom context**: `class_type`, `school`, `school_id`, `school_location`

---

## Tech Stack

`pandas` · `scikit-learn` · `XGBoost` · `LightGBM` · `SHAP` · `matplotlib` · `seaborn`

---

## Reference

Chetty, R., Friedman, J. N., Hilger, N., Saez, E., Schanzenbach, D. W., & Yagan, D. (2011). How does your kindergarten classroom affect your earnings? *The Quarterly Journal of Economics*, 126(4), 1593–1660.
