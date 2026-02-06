import os
import json
from typing import Dict, List, Tuple

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

MODEL_PATH = os.path.join(os.path.dirname(__file__), "trained_model.pkl")

FEATURE_COLUMNS: List[str] = [
    "sunlight_hours",
    "safety",
    "sleep_duration_hours",
    "screen_time_minutes",
    "physical_activity_minutes",
    "daily_goal_progression",
    "hour",
    "weekday",
]

DATA_PATH = os.path.join(os.path.dirname(__file__), "generated_sample_data.json")


def _load_training_dataframe() -> pd.DataFrame:
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        sample_data = json.load(f)
    df = pd.DataFrame(sample_data)
    df["target"] = df["mental_state"]
    return df


def prepare_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    X = df[FEATURE_COLUMNS]
    y = df["target"]
    return X, y


def train_model(X: pd.DataFrame, y: pd.Series) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    return model


def retrain_model(print_metrics: bool = True) -> RandomForestRegressor:
    df = _load_training_dataframe()
    X, y = prepare_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = train_model(X_train, y_train)

    if print_metrics:
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        print(f"Model retrained and saved to {MODEL_PATH}")
        print(f"Validation MAE: {mae:.2f}  |  R²: {r2:.2f}")
    return model


def load_model() -> RandomForestRegressor:
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return retrain_model(print_metrics=True)


def _sanitize_input(input_dict: Dict) -> Dict:
    safe = {}
    for k in FEATURE_COLUMNS:
        v = input_dict.get(k, 0)
        try:
            safe[k] = float(v)
        except (TypeError, ValueError):
            safe[k] = 0.0
    return safe


def get_mental_state(input_data: Dict) -> int:
    model = load_model()
    safe = _sanitize_input(input_data)

    input_df = pd.DataFrame([safe])[FEATURE_COLUMNS]
    pred = float(model.predict(input_df)[0])

    pred = max(0.0, min(100.0, pred))
    return int(round(pred))
