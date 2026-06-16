# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, RandomForestRegressor, StackingRegressor, VotingRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OrdinalEncoder, RobustScaler, StandardScaler

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None

try:
    import optuna
except Exception:
    optuna = None

APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent

DATA_DIR_CANDIDATES = [
    Path(os.getenv("DRUG_DEMAND_DATA_DIR", "")) if os.getenv("DRUG_DEMAND_DATA_DIR") else None,
    PROJECT_DIR / "data",
    APP_DIR / "data",
    Path.home() / "Downloads",
    Path(r"C:\Users\404_14\Desktop\MLPro\data"),
]
DATA_DIR = next((p for p in DATA_DIR_CANDIDATES if p and p.exists()), PROJECT_DIR / "data")
OUTPUT_DIR = Path(os.getenv("DRUG_DEMAND_OUTPUT_DIR", APP_DIR))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DRUG_FILES = {
    "\ud574\uc5f4\uc9c4\ud1b5\uc18c\uc5fc\uc81c": "flu_cleaned.csv",
    "\ud56d\ud788\uc2a4\ud0c0\ubbfc\uc81c": "hist_cleaned.csv",
    "\uc9c4\ud574\uac70\ub2f4\uc81c": "cacp_cleaned.csv",
}

COL_DATE = "\uc77c\uc2dc"
COL_DRUG = "\uc57d\ud488\uad6c\ubd84"
COL_DISTRICT = "\uc2dc\uad70\uad6c\uba85\uce6d"
COL_PRED = "\uc608\uce21\uc218\ub7c9"
MODEL_COL = "\ud559\uc2b5\ubaa8\ub378"
METHOD_COL = "\ud559\uc2b5\ubc29\ubc95"

FEATURE_COLUMNS = [
    "year", "month", "quarter", "month_sin", "month_cos",
    "avg_temp", "max_temp", "min_temp", "rainfall", "max_rainfall", "avg_wind", "max_wind",
    "temp_range", "cold_index", "drug_type_code", "district_code",
    "qty_lag1", "qty_lag2", "qty_lag3", "qty_lag12", "qty_ma3", "qty_ma6",
]


def read_csv_with_fallback(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "utf-8", "cp949"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def parse_month_date(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    parsed = pd.to_datetime(s, format="%Y-%m", errors="coerce")
    for fmt in ("%b-%y", "%Y.%m", "%Y/%m"):
        mask = parsed.isna()
        if mask.any():
            parsed.loc[mask] = pd.to_datetime(s[mask], format=fmt, errors="coerce")
    mask = parsed.isna()
    if mask.any():
        parsed.loc[mask] = pd.to_datetime(s[mask], errors="coerce")
    if parsed.isna().any():
        bad_values = s[parsed.isna()].unique()[:10]
        raise ValueError(f"\ub0a0\uc9dc \ubcc0\ud658 \uc2e4\ud328 \uac12 \uc608\uc2dc: {bad_values}")
    return parsed.dt.to_period("M").dt.to_timestamp()


def load_weather() -> pd.DataFrame:
    path = DATA_DIR / "temp.csv"
    if not path.exists():
        raise FileNotFoundError(f"temp.csv\ub97c \ucc3e\uc744 \uc218 \uc5c6\uc2b5\ub2c8\ub2e4: {path}")
    weather = read_csv_with_fallback(path).copy()
    weather = weather.iloc[:, [2, 3, 4, 5, 6, 7, 8, 9]].copy()
    weather.columns = ["date", "avg_temp", "max_temp", "min_temp", "rainfall", "max_rainfall", "avg_wind", "max_wind"]
    weather["date"] = parse_month_date(weather["date"])
    for col in weather.columns.drop("date"):
        weather[col] = pd.to_numeric(weather[col], errors="coerce")
    return weather.sort_values("date").reset_index(drop=True)


def load_demand() -> pd.DataFrame:
    frames = []
    missing = []
    for drug_type, file_name in DRUG_FILES.items():
        path = DATA_DIR / file_name
        if not path.exists():
            missing.append(file_name)
            continue
        raw = read_csv_with_fallback(path).copy()
        raw = raw.iloc[:, [0, 1, 2, 3]].copy()
        raw.columns = ["date", "province", "district", "qty"]
        raw["date"] = parse_month_date(raw["date"])
        raw["qty"] = pd.to_numeric(raw["qty"], errors="coerce")
        raw["drug_type"] = drug_type
        frames.append(raw)
    if missing:
        print("\uac74\ub108\ub6f4 \uc218\uc694 CSV:", ", ".join(missing))
    if not frames:
        raise FileNotFoundError(f"\uc218\uc694 CSV\ub97c \ucc3e\uc744 \uc218 \uc5c6\uc2b5\ub2c8\ub2e4. data \ud3f4\ub354 \ub610\ub294 DRUG_DEMAND_DATA_DIR\ub97c \ud655\uc778\ud558\uc138\uc694: {DATA_DIR}")
    demand = pd.concat(frames, ignore_index=True)
    return (
        demand.groupby(["date", "province", "district", "drug_type"], as_index=False)
        .agg(qty=("qty", "sum"))
        .sort_values(["drug_type", "district", "date"])
        .reset_index(drop=True)
    )


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["drug_type", "district", "date"]).copy()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["temp_range"] = df["max_temp"] - df["min_temp"]
    df["cold_index"] = (df["avg_temp"] < 5).astype(int)
    grouped = df.groupby(["drug_type", "district"], sort=False)["qty"]
    for lag in (1, 2, 3, 12):
        df[f"qty_lag{lag}"] = grouped.shift(lag)
    df["qty_ma3"] = grouped.transform(lambda s: s.shift(1).rolling(3).mean())
    df["qty_ma6"] = grouped.transform(lambda s: s.shift(1).rolling(6).mean())
    return df


def prepare_training_data() -> Tuple[pd.DataFrame, OrdinalEncoder]:
    demand = load_demand()
    weather = load_weather()
    df = demand.merge(weather, on="date", how="left")
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    encoded = encoder.fit_transform(df[["drug_type", "district"]])
    df["drug_type_code"] = encoded[:, 0]
    df["district_code"] = encoded[:, 1]
    df = add_features(df)
    return df.dropna(subset=FEATURE_COLUMNS + ["qty"]).reset_index(drop=True), encoder



RUN_GRID_SEARCH = os.getenv("RUN_GRID_SEARCH", "1") != "0"
RUN_OPTUNA = os.getenv("RUN_OPTUNA", "1") != "0"
OPTUNA_TRIALS = int(os.getenv("OPTUNA_TRIALS", "12"))

MODEL_RF = "\ub79c\ub364\ud3ec\ub808\uc2a4\ud2b8"
MODEL_XGB = "XGBoost"
MODEL_LASSO = "\ub77c\uc3d8\ud68c\uadc0"
MODEL_RIDGE = "\ub9bf\uc9c0\ud68c\uadc0"
MODEL_HYBRID = "\ud63c\ud569\uc559\uc0c1\ube14"

METHOD_GRID = "\uadf8\ub9ac\ub4dc\uc11c\uce58"
METHOD_OPTUNA = "\uc635\ud29c\ub098"
METHOD_VOTING = "\ubcf4\ud305"
METHOD_BAGGING = "\ubc30\uae45"
METHOD_BOOSTING = "\ubd80\uc2a4\ud305"
METHOD_STACKING = "\uc2a4\ud0dc\ud0b9"


def make_ridge(alpha=1.0):
    return Pipeline([("scaler", RobustScaler()), ("model", Ridge(alpha=alpha))])


def make_lasso(alpha=1.0):
    return Pipeline([("scaler", StandardScaler()), ("model", Lasso(alpha=alpha, max_iter=200000, random_state=42))])


def make_bagging(base_estimator, n_estimators=120):
    try:
        return BaggingRegressor(estimator=base_estimator, n_estimators=n_estimators, random_state=42, n_jobs=-1)
    except TypeError:
        return BaggingRegressor(base_estimator=base_estimator, n_estimators=n_estimators, random_state=42, n_jobs=-1)


def make_adaboost(base_estimator, n_estimators=80):
    try:
        return AdaBoostRegressor(estimator=base_estimator, n_estimators=n_estimators, learning_rate=0.05, random_state=42)
    except TypeError:
        return AdaBoostRegressor(base_estimator=base_estimator, n_estimators=n_estimators, learning_rate=0.05, random_state=42)


def regression_metrics(y_true, y_pred) -> dict:
    y_pred = np.maximum(np.asarray(y_pred, dtype=float), 0)
    y_true_arr = np.asarray(y_true, dtype=float)
    safe = np.where(y_true_arr == 0, np.nan, y_true_arr)
    return {
        "mae": mean_absolute_error(y_true_arr, y_pred),
        "rmse": mean_squared_error(y_true_arr, y_pred) ** 0.5,
        "mape": np.nanmean(np.abs((y_true_arr - y_pred) / safe)) * 100,
        "r2": r2_score(y_true_arr, y_pred),
    }


def optuna_split(train_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    opt_train = train_df[train_df["date"] < "2024-01-01"].copy()
    opt_valid = train_df[train_df["date"] >= "2024-01-01"].copy()
    if opt_train.empty or opt_valid.empty:
        midpoint = max(1, int(len(train_df) * 0.8))
        opt_train = train_df.iloc[:midpoint].copy()
        opt_valid = train_df.iloc[midpoint:].copy()
    return opt_train, opt_valid


def create_model_specs(train_df: pd.DataFrame) -> List[Tuple[str, str, object]]:
    if optuna is not None:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    opt_train, opt_valid = optuna_split(train_df)
    model_specs = []

    if RUN_GRID_SEARCH:
        model_specs.append((MODEL_RF, METHOD_GRID, GridSearchCV(
            RandomForestRegressor(random_state=42, n_jobs=-1),
            {"n_estimators": [200, 500], "max_depth": [8, 12], "min_samples_leaf": [1, 2]},
            scoring="neg_mean_absolute_error", cv=3, n_jobs=-1,
        )))

    if RUN_OPTUNA and optuna is not None:
        def rf_objective(trial):
            model = RandomForestRegressor(
                n_estimators=trial.suggest_int("n_estimators", 200, 500, step=100),
                max_depth=trial.suggest_int("max_depth", 6, 16),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 4),
                max_features=trial.suggest_categorical("max_features", ["sqrt", 0.7, 1.0]),
                random_state=42, n_jobs=-1,
            )
            model.fit(opt_train[FEATURE_COLUMNS], opt_train["qty"])
            return mean_absolute_error(opt_valid["qty"], model.predict(opt_valid[FEATURE_COLUMNS]))
        study = optuna.create_study(direction="minimize")
        study.optimize(rf_objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)
        model_specs.append((MODEL_RF, METHOD_OPTUNA, RandomForestRegressor(**study.best_params, random_state=42, n_jobs=-1)))

    model_specs.append((MODEL_RF, METHOD_BAGGING, RandomForestRegressor(n_estimators=500, max_depth=12, min_samples_leaf=2, random_state=42, n_jobs=-1)))
    rf_vote = [
        ("rf1", RandomForestRegressor(n_estimators=250, max_depth=10, min_samples_leaf=1, random_state=41, n_jobs=-1)),
        ("rf2", RandomForestRegressor(n_estimators=350, max_depth=14, min_samples_leaf=2, random_state=42, n_jobs=-1)),
        ("ridge", make_ridge(1.0)),
    ]
    if XGBRegressor is not None:
        rf_vote.append(("xgb", XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, objective="reg:squarederror", random_state=42, n_jobs=-1, verbosity=0)))
    model_specs.append((MODEL_RF, METHOD_VOTING, VotingRegressor(estimators=rf_vote, n_jobs=-1)))
    if XGBRegressor is not None:
        model_specs.append((MODEL_RF, METHOD_BOOSTING, XGBRegressor(n_estimators=500, learning_rate=0.04, max_depth=4, subsample=0.9, colsample_bytree=0.9, objective="reg:squarederror", random_state=42, n_jobs=-1, verbosity=0)))
    rf_stack = [("rf", RandomForestRegressor(n_estimators=500, max_depth=12, min_samples_leaf=2, random_state=42, n_jobs=-1)), ("ridge", make_ridge(1.0))]
    if XGBRegressor is not None:
        rf_stack.append(("xgb", XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, objective="reg:squarederror", random_state=42, n_jobs=-1, verbosity=0)))
    model_specs.append((MODEL_RF, METHOD_STACKING, StackingRegressor(estimators=rf_stack, final_estimator=Ridge(alpha=1.0), cv=3, n_jobs=-1)))

    if XGBRegressor is not None:
        if RUN_GRID_SEARCH:
            model_specs.append((MODEL_XGB, METHOD_GRID, GridSearchCV(
                XGBRegressor(objective="reg:squarederror", random_state=42, n_jobs=-1, verbosity=0),
                {"n_estimators": [200, 500], "learning_rate": [0.03, 0.05], "max_depth": [3, 4], "subsample": [0.7, 0.9], "colsample_bytree": [0.7, 0.9]},
                scoring="neg_mean_absolute_error", cv=3, n_jobs=-1,
            )))
        if RUN_OPTUNA and optuna is not None:
            def xgb_objective(trial):
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 150, 400, step=50),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                    "max_depth": trial.suggest_int("max_depth", 3, 6),
                    "subsample": trial.suggest_float("subsample", 0.6, 0.95),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.95),
                    "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 25.0),
                    "objective": "reg:squarederror", "random_state": 42, "n_jobs": -1, "verbosity": 0,
                }
                model = XGBRegressor(**params)
                model.fit(opt_train[FEATURE_COLUMNS], opt_train["qty"])
                return mean_absolute_error(opt_valid["qty"], model.predict(opt_valid[FEATURE_COLUMNS]))
            study = optuna.create_study(direction="minimize")
            study.optimize(xgb_objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)
            params = dict(study.best_params)
            params.update({"objective": "reg:squarederror", "random_state": 42, "n_jobs": -1, "verbosity": 0})
            model_specs.append((MODEL_XGB, METHOD_OPTUNA, XGBRegressor(**params)))
        model_specs.append((MODEL_XGB, METHOD_BOOSTING, XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=4, subsample=0.7, colsample_bytree=0.7, reg_lambda=15.0, objective="reg:squarederror", random_state=42, n_jobs=-1, verbosity=0)))
        model_specs.append((MODEL_XGB, METHOD_BAGGING, make_bagging(XGBRegressor(n_estimators=250, learning_rate=0.05, max_depth=4, objective="reg:squarederror", random_state=42, n_jobs=-1, verbosity=0), n_estimators=60)))
        model_specs.append((MODEL_XGB, METHOD_VOTING, VotingRegressor(estimators=[
            ("xgb_a", XGBRegressor(n_estimators=250, learning_rate=0.05, max_depth=3, objective="reg:squarederror", random_state=41, n_jobs=-1, verbosity=0)),
            ("xgb_b", XGBRegressor(n_estimators=350, learning_rate=0.03, max_depth=4, objective="reg:squarederror", random_state=42, n_jobs=-1, verbosity=0)),
            ("rf", RandomForestRegressor(n_estimators=500, max_depth=12, min_samples_leaf=2, random_state=42, n_jobs=-1)),
        ], n_jobs=-1)))
        model_specs.append((MODEL_XGB, METHOD_STACKING, StackingRegressor(estimators=[
            ("xgb", XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, objective="reg:squarederror", random_state=42, n_jobs=-1, verbosity=0)),
            ("rf", RandomForestRegressor(n_estimators=500, max_depth=12, min_samples_leaf=2, random_state=42, n_jobs=-1)),
            ("ridge", make_ridge(1.0)),
        ], final_estimator=Ridge(alpha=1.0), cv=3, n_jobs=-1)))

        hybrid_xgb = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=4, subsample=0.7, colsample_bytree=0.7, reg_lambda=15.0, objective="reg:squarederror", random_state=42, n_jobs=-1, verbosity=0)
        hybrid_rf = RandomForestRegressor(n_estimators=500, max_depth=12, min_samples_leaf=2, random_state=42, n_jobs=-1)
        model_specs.append((MODEL_HYBRID, "Ridge+RandomForest+XGBoost", StackingRegressor(estimators=[("ridge", make_ridge(1.0)), ("rf", hybrid_rf), ("xgb", hybrid_xgb)], final_estimator=Ridge(alpha=1.0), cv=3, n_jobs=-1)))
        model_specs.append((MODEL_HYBRID, "Lasso+Ridge+XGBoost", StackingRegressor(estimators=[("lasso", make_lasso(1.0)), ("ridge", make_ridge(1.0)), ("xgb", hybrid_xgb)], final_estimator=Ridge(alpha=1.0), cv=3, n_jobs=-1)))
        model_specs.append((MODEL_HYBRID, "Ridge+Lasso+RandomForest+XGBoost", StackingRegressor(estimators=[("ridge", make_ridge(1.0)), ("lasso", make_lasso(1.0)), ("rf", hybrid_rf), ("xgb", hybrid_xgb)], final_estimator=Ridge(alpha=1.0), cv=3, n_jobs=-1)))

    ridge_alphas = np.logspace(-4, 4, 80)
    lasso_alphas = np.logspace(-2, 7, 70)
    model_specs.append((MODEL_RIDGE, METHOD_GRID, GridSearchCV(make_ridge(), {"model__alpha": ridge_alphas}, scoring="neg_mean_absolute_error", cv=3, n_jobs=-1)))
    if RUN_OPTUNA and optuna is not None:
        def ridge_objective(trial):
            model = make_ridge(trial.suggest_float("alpha", 1e-4, 1e4, log=True))
            model.fit(opt_train[FEATURE_COLUMNS], opt_train["qty"])
            return mean_absolute_error(opt_valid["qty"], model.predict(opt_valid[FEATURE_COLUMNS]))
        study = optuna.create_study(direction="minimize")
        study.optimize(ridge_objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)
        model_specs.append((MODEL_RIDGE, METHOD_OPTUNA, make_ridge(study.best_params["alpha"])))
    model_specs.append((MODEL_RIDGE, METHOD_BAGGING, make_bagging(make_ridge(1.0))))
    model_specs.append((MODEL_RIDGE, METHOD_VOTING, VotingRegressor(estimators=[("ridge_a", make_ridge(0.1)), ("ridge_b", make_ridge(1.0)), ("ridge_c", make_ridge(10.0))], n_jobs=-1)))
    model_specs.append((MODEL_RIDGE, METHOD_BOOSTING, make_adaboost(Ridge(alpha=1.0))))
    model_specs.append((MODEL_RIDGE, METHOD_STACKING, StackingRegressor(estimators=[("ridge_a", make_ridge(0.1)), ("ridge_b", make_ridge(1.0)), ("ridge_c", make_ridge(10.0))], final_estimator=Ridge(alpha=1.0), cv=3, n_jobs=-1)))

    model_specs.append((MODEL_LASSO, METHOD_GRID, GridSearchCV(make_lasso(), {"model__alpha": lasso_alphas}, scoring="neg_mean_absolute_error", cv=3, n_jobs=-1)))
    if RUN_OPTUNA and optuna is not None:
        def lasso_objective(trial):
            model = make_lasso(trial.suggest_float("alpha", 1e-2, 1e7, log=True))
            model.fit(opt_train[FEATURE_COLUMNS], opt_train["qty"])
            return mean_absolute_error(opt_valid["qty"], model.predict(opt_valid[FEATURE_COLUMNS]))
        study = optuna.create_study(direction="minimize")
        study.optimize(lasso_objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)
        model_specs.append((MODEL_LASSO, METHOD_OPTUNA, make_lasso(study.best_params["alpha"])))
    model_specs.append((MODEL_LASSO, METHOD_BAGGING, make_bagging(make_lasso(1.0))))
    model_specs.append((MODEL_LASSO, METHOD_VOTING, VotingRegressor(estimators=[("lasso_a", make_lasso(0.1)), ("lasso_b", make_lasso(1.0)), ("lasso_c", make_lasso(10.0))], n_jobs=-1)))
    model_specs.append((MODEL_LASSO, METHOD_BOOSTING, make_adaboost(Lasso(alpha=1.0, max_iter=200000, random_state=42))))
    model_specs.append((MODEL_LASSO, METHOD_STACKING, StackingRegressor(estimators=[("lasso_a", make_lasso(0.1)), ("lasso_b", make_lasso(1.0)), ("lasso_c", make_lasso(10.0))], final_estimator=Ridge(alpha=1.0), cv=3, n_jobs=-1)))
    return model_specs


def train_and_validate(df: pd.DataFrame):
    train_df = df[df["date"].dt.year < 2025].copy()
    test_df = df[df["date"].dt.year == 2025].copy()
    if train_df.empty or test_df.empty:
        raise ValueError("2025\ub144 \uac80\uc99d\uc744 \ud558\ub824\uba74 2025\ub144 \uc774\uc804 \ud559\uc2b5 \ub370\uc774\ud130\uc640 2025\ub144 \ub370\uc774\ud130\uac00 \ubaa8\ub450 \ud544\uc694\ud569\ub2c8\ub2e4.")
    trained_models = {}
    metrics_rows = []
    pred_frames = []
    specs = create_model_specs(train_df)
    for model_type, method_name, estimator in specs:
        key = f"{model_type} / {method_name}"
        print(f"Training {key}...")
        try:
            estimator.fit(train_df[FEATURE_COLUMNS], train_df["qty"])
            pred = np.maximum(estimator.predict(test_df[FEATURE_COLUMNS]), 0)
        except Exception as exc:
            print(f"Skipped {key}: {exc}")
            continue
        result = test_df[["date", "province", "district", "drug_type", "qty"]].copy()
        result["predicted_qty"] = pred
        result[MODEL_COL] = model_type
        result[METHOD_COL] = method_name
        pred_frames.append(result)
        trained_models[key] = estimator
        for drug_type, part in result.groupby("drug_type"):
            metrics_rows.append({MODEL_COL: model_type, METHOD_COL: method_name, "drug_type": drug_type, **regression_metrics(part["qty"], part["predicted_qty"])})
    metrics = pd.DataFrame(metrics_rows)
    predictions = pd.concat(pred_frames, ignore_index=True)
    best_row = metrics.sort_values("mape").iloc[0]
    best_key = f"{best_row[MODEL_COL]} / {best_row[METHOD_COL]}"
    return trained_models, best_key, metrics, predictions


def build_future_weather(weather: pd.DataFrame, months: pd.DatetimeIndex) -> pd.DataFrame:
    monthly_weather = weather.copy()
    monthly_weather["month"] = monthly_weather["date"].dt.month
    defaults = monthly_weather.groupby("month")[["avg_temp", "max_temp", "min_temp", "rainfall", "max_rainfall", "avg_wind", "max_wind"]].mean()
    rows = []
    for date in months:
        row = defaults.loc[date.month].to_dict()
        row["date"] = date
        rows.append(row)
    return pd.DataFrame(rows)


def forecast_2026(df: pd.DataFrame, encoder: OrdinalEncoder, trained_models: Dict[str, object]) -> pd.DataFrame:
    weather = load_weather()
    months = pd.date_range("2026-01-01", "2026-12-01", freq="MS")
    future_weather = build_future_weather(weather, months)
    history = df[["date", "province", "district", "drug_type", "qty"]].copy()
    rows = []
    keys = history[["province", "district", "drug_type"]].drop_duplicates().sort_values(["drug_type", "district"])
    for key, model in trained_models.items():
        model_type, method_name = key.split(" / ", 1)
        rolling = history.copy()
        for date in months:
            base = keys.copy()
            base["date"] = date
            base = base.merge(future_weather, on="date", how="left")
            encoded = encoder.transform(base[["drug_type", "district"]])
            base["drug_type_code"] = encoded[:, 0]
            base["district_code"] = encoded[:, 1]
            combined = pd.concat([rolling, base.assign(qty=np.nan)], ignore_index=True, sort=False)
            combined = add_features(combined)
            future_rows = combined[combined["date"] == date].copy()
            future_rows[FEATURE_COLUMNS] = future_rows[FEATURE_COLUMNS].fillna(0)
            pred = np.maximum(model.predict(future_rows[FEATURE_COLUMNS]), 0)
            future_rows["qty"] = pred
            rolling = pd.concat([rolling, future_rows[["date", "province", "district", "drug_type", "qty"]]], ignore_index=True)
            out = future_rows[["date", "province", "district", "drug_type"]].copy()
            out[COL_PRED] = pred
            out[MODEL_COL] = model_type
            out[METHOD_COL] = method_name
            rows.append(out)
    forecast = pd.concat(rows, ignore_index=True)
    forecast[COL_DATE] = forecast["date"].dt.strftime("%Y-%m")
    forecast = forecast.rename(columns={"drug_type": COL_DRUG, "district": COL_DISTRICT})
    return forecast[[COL_DATE, "province", COL_DISTRICT, COL_DRUG, COL_PRED, MODEL_COL, METHOD_COL]]


def save_outputs(trained_models, best_key, metrics, predictions, forecast, encoder):
    metrics.to_csv(OUTPUT_DIR / "baseline_models_2025_validation_metrics.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(OUTPUT_DIR / "baseline_models_2025_test_predictions.csv", index=False, encoding="utf-8-sig")
    best_model, best_method = best_key.split(" / ", 1)
    best_forecast = forecast[(forecast[MODEL_COL] == best_model) & (forecast[METHOD_COL] == best_method)].copy()
    best_forecast.to_csv(OUTPUT_DIR / "baseline_best_model_2026_demand_forecast.csv", index=False, encoding="utf-8-sig")
    forecast.to_csv(OUTPUT_DIR / "model_forecast_2026_all_methods.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(OUTPUT_DIR / "model_comparison_2025_metrics.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(OUTPUT_DIR / "model_comparison_2025_predictions.csv", index=False, encoding="utf-8-sig")
    best_forecast.to_csv(OUTPUT_DIR / "random_forest_2026_demand_forecast.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(OUTPUT_DIR / "random_forest_2025_validation_metrics.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(OUTPUT_DIR / "random_forest_2025_test_predictions.csv", index=False, encoding="utf-8-sig")
    payload = {
        "models": trained_models,
        "selected_model_key": best_key,
        "encoder": encoder,
        "feature_columns": FEATURE_COLUMNS,
        "drug_files": DRUG_FILES,
        "data_dir": str(DATA_DIR),
        "model_column": MODEL_COL,
        "method_column": METHOD_COL,
    }
    with open(OUTPUT_DIR / "drug_demand_models.pkl", "wb") as f:
        pickle.dump(payload, f)

def main():
    print(f"DATA_DIR={DATA_DIR}")
    print(f"OUTPUT_DIR={OUTPUT_DIR}")
    df, encoder = prepare_training_data()
    trained_models, best_key, metrics, predictions = train_and_validate(df)
    forecast = forecast_2026(df, encoder, trained_models)
    save_outputs(trained_models, best_key, metrics, predictions, forecast, encoder)
    print(f"\uc120\ud0dd\ub41c \uae30\uc900 \ubaa8\ub378: {best_key}")
    print(f"\uc800\uc7a5 \uc644\ub8cc: {OUTPUT_DIR / 'drug_demand_models.pkl'}")


if __name__ == "__main__":
    main()
