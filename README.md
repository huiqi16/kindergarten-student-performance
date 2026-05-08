# Student Performance Prediction — Project STAR Dataset

Predicting kindergarten **reading and math scores** using student demographics, teacher attributes, and school context. Six model families are compared; **XGBoost** is selected as the final model.

---

## Results

| Model | MSE (Read) | MSE (Math) | Combined MSE_o |
|---|---|---|---|
| OLS | — | — | baseline |
| LASSO / Ridge / ElasticNet | — | — | similar to OLS |
| Decision Tree | — | — | worst (overfits) |
| Random Forest | — | — | strong |
| Neural Network (MLP) | — | — | similar to linear |
| **XGBoost** ✓ | — | — | **best** |
| LightGBM | — | — | close to XGBoost |

> Run the notebooks to populate exact MSE values.

---

## Key Findings

- **Lunch status** (free-lunch eligibility, a proxy for SES) is the single strongest predictor for both scores
- **School ID** captures large between-school quality variance
- **Teacher experience** has a consistent positive effect, more so for math
- **Small class type** improves outcomes — consistent with the original Project STAR experiment
- Neural Networks underperformed linear baselines, likely due to sparse one-hot encoded features

---

## Repo Structure

```
├── src/
│   ├── preprocessing.py   # clean_data(), remove_outliers_iqr(), prepare_data()
│   └── models.py          # training functions for all models
│
├── notebooks/
│   ├── 01_eda.ipynb               # missing values, distributions, bivariate plots, correlation
│   ├── 02_preprocessing.ipynb     # cleaning steps, outlier removal, train/test split
│   ├── 03_models.ipynb            # all models trained + compared, GridSearchCV archive
│   └── 04_feature_importance.ipynb # XGBoost importances + grouped SHAP analysis
│
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
git clone https://github.com/YOUR_USERNAME/mooketing-student-performance
cd mooketing-student-performance
pip install -r requirements.txt
jupyter notebook notebooks/01_eda.ipynb
```

Data is loaded automatically from a public Google Drive URL — no local file needed.

---

## Dataset

Derived from **Project STAR** (Student/Teacher Achievement Ratio), a randomised controlled trial in Tennessee studying the effect of class size on early academic performance.

Variables include student demographics (`gender`, `ethnicity`, `birth`, `lunch`), teacher attributes (`degree`, `experience`, `ladder`), and school context (`class_type`, `school`, `school_id`).

---

## Tech Stack

`pandas` · `scikit-learn` · `XGBoost` · `LightGBM` · `SHAP` · `matplotlib` · `seaborn`
