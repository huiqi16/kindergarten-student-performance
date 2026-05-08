"""
models.py
Training functions for all models used in the Mooketing student performance project.
Each function returns a fitted sklearn Pipeline (preprocessor + model).
"""

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, ElasticNetCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def evaluate(pipe_read, pipe_math, X_test, y_read_test, y_math_test) -> dict:
    """Return MSE for reading, math, and the combined MSE_o metric."""
    pred_read = pipe_read.predict(X_test)
    pred_math = pipe_math.predict(X_test)
    mse_read = mean_squared_error(y_read_test, pred_read)
    mse_math = mean_squared_error(y_math_test, pred_math)
    mse_o = ((pred_read - y_read_test) ** 2 + (pred_math - y_math_test) ** 2).mean() / 2
    return {"mse_read": mse_read, "mse_math": mse_math, "mse_o": mse_o}


def _make_pipe(preprocessor, model):
    return Pipeline([("prep", preprocessor), ("model", model)])


# ---------------------------------------------------------------------------
# Linear models (OLS, LASSO, Ridge, ElasticNet)
# ---------------------------------------------------------------------------

def train_linear_models(preprocessor, X_train, y_read_train, y_math_train):
    """
    Train OLS, LASSO, Ridge and ElasticNet for both targets.
    Returns dict of {name: (pipe_read, pipe_math)}.
    """
    model_defs = {
        "OLS": LinearRegression(),
        "LASSO": LassoCV(cv=5, random_state=42),
        "Ridge": RidgeCV(alphas=np.logspace(-3, 3, 100), cv=5),
        "ElasticNet": ElasticNetCV(cv=5, l1_ratio=np.linspace(0.1, 1, 10), random_state=42),
    }

    fitted = {}
    for name, model in model_defs.items():
        pipe_read = _make_pipe(preprocessor, model)
        pipe_read.fit(X_train, y_read_train)

        # Re-instantiate so math gets its own fitted copy
        pipe_math = _make_pipe(preprocessor, type(model)(**model.get_params()))
        pipe_math.fit(X_train, y_math_train)

        fitted[name] = (pipe_read, pipe_math)

    return fitted


# ---------------------------------------------------------------------------
# Decision Tree
# ---------------------------------------------------------------------------

def train_decision_tree(preprocessor, X_train, y_read_train, y_math_train):
    pipe_read = _make_pipe(preprocessor, DecisionTreeRegressor(random_state=42))
    pipe_math = _make_pipe(preprocessor, DecisionTreeRegressor(random_state=42))
    pipe_read.fit(X_train, y_read_train)
    pipe_math.fit(X_train, y_math_train)
    return pipe_read, pipe_math


# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------

def train_random_forest(preprocessor, X_train, y_read_train, y_math_train):
    rf_params = dict(
        n_estimators=500,
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        bootstrap=True,
        n_jobs=-1,
        random_state=42,
    )
    pipe_read = _make_pipe(preprocessor, RandomForestRegressor(**rf_params))
    pipe_math = _make_pipe(preprocessor, RandomForestRegressor(**rf_params))
    pipe_read.fit(X_train, y_read_train)
    pipe_math.fit(X_train, y_math_train)
    return pipe_read, pipe_math


# ---------------------------------------------------------------------------
# Neural Network (MLP) - tuned params from GridSearchCV
# ---------------------------------------------------------------------------

def train_neural_network(preprocessor, X_train, y_read_train, y_math_train):
    # Best params found via GridSearchCV (see notebooks/03_models.ipynb)
    pipe_read = _make_pipe(
        preprocessor,
        MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=0.0001,
            learning_rate_init=0.003,
            max_iter=500,
            early_stopping=True,
            random_state=42,
        ),
    )
    pipe_math = _make_pipe(
        preprocessor,
        MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=0.001,
            learning_rate_init=0.003,
            max_iter=500,
            early_stopping=True,
            random_state=42,
        ),
    )
    pipe_read.fit(X_train, y_read_train)
    pipe_math.fit(X_train, y_math_train)
    return pipe_read, pipe_math


# ---------------------------------------------------------------------------
# XGBoost - tuned params from GridSearchCV
# ---------------------------------------------------------------------------

def train_xgboost(preprocessor, X_train, y_read_train, y_math_train):
    from xgboost import XGBRegressor

    pipe_read = _make_pipe(
        preprocessor,
        XGBRegressor(
            objective="reg:squarederror",
            n_estimators=300,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=1.0,
            gamma=0.1,
            reg_lambda=1,
            reg_alpha=0,
            random_state=42,
            n_jobs=-1,
        ),
    )
    pipe_math = _make_pipe(
        preprocessor,
        XGBRegressor(
            objective="reg:squarederror",
            n_estimators=300,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=1.0,
            gamma=0,
            reg_lambda=1,
            reg_alpha=0.5,
            random_state=42,
            n_jobs=-1,
        ),
    )
    pipe_read.fit(X_train, y_read_train)
    pipe_math.fit(X_train, y_math_train)
    return pipe_read, pipe_math


# ---------------------------------------------------------------------------
# LightGBM
# ---------------------------------------------------------------------------

def train_lightgbm(preprocessor, X_train, y_read_train, y_math_train):
    from lightgbm import LGBMRegressor

    params = dict(
        colsample_bytree=0.8,
        learning_rate=0.03,
        n_estimators=500,
        num_leaves=31,
        subsample=1.0,
        random_state=42,
        verbose=-1,
    )
    pipe_read = _make_pipe(preprocessor, LGBMRegressor(**params))
    pipe_math = _make_pipe(preprocessor, LGBMRegressor(**params))
    pipe_read.fit(X_train, y_read_train)
    pipe_math.fit(X_train, y_math_train)
    return pipe_read, pipe_math
