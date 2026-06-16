import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier

from app.config import DATASET_PATH, MODEL_PATH
from app.ml.features import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)


RANDOM_STATE = 42


def load_dataset(path: Path = DATASET_PATH) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_excel(path, header=2)
    df = df.dropna(how="all")
    missing = [column for column in FEATURE_COLUMNS + [TARGET_COLUMN] if column not in df.columns]
    if missing:
        raise ValueError(f"В датасете не найдены колонки: {missing}")

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].astype(str)

    for column in NUMERIC_FEATURES:
        X[column] = pd.to_numeric(X[column], errors="raise")
    for column in CATEGORICAL_FEATURES:
        X[column] = X[column].astype(str)

    return X, y


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("categorical", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CATEGORICAL_FEATURES),
            ("numeric", "passthrough", NUMERIC_FEATURES),
        ],
        remainder="drop",
    )


def build_candidates() -> dict[str, Pipeline]:
    base_preprocessor = make_preprocessor()
    return {
        "Decision Tree": Pipeline(
            steps=[
                ("preprocess", base_preprocessor),
                ("classifier", DecisionTreeClassifier(max_depth=4, criterion="gini", class_weight="balanced", random_state=RANDOM_STATE)),
            ]
        ),
        "Logistic Regression": Pipeline(
            steps=[
                ("preprocess", make_preprocessor()),
                ("scale", MinMaxScaler()),
                ("classifier", LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000, random_state=RANDOM_STATE)),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("preprocess", make_preprocessor()),
                ("classifier", RandomForestClassifier(n_estimators=100, max_depth=5, class_weight="balanced", random_state=RANDOM_STATE)),
            ]
        ),
    }


def evaluate_models(X: pd.DataFrame, y: pd.Series) -> dict[str, dict[str, float]]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scorers = {
        "accuracy": make_scorer(accuracy_score),
        "macro_f1": make_scorer(f1_score, average="macro"),
        "weighted_f1": make_scorer(f1_score, average="weighted"),
    }

    results = {}
    for name, model in build_candidates().items():
        scores = cross_validate(model, X, y, cv=cv, scoring=scorers)
        results[name] = {
            "accuracy": round(float(scores["test_accuracy"].mean()), 4),
            "macro_f1": round(float(scores["test_macro_f1"].mean()), 4),
            "weighted_f1": round(float(scores["test_weighted_f1"].mean()), 4),
        }
    return results


def train_and_save() -> dict:
    X, y = load_dataset()
    metrics = evaluate_models(X, y)
    best_name = max(metrics, key=lambda name: metrics[name]["macro_f1"])
    best_model = build_candidates()[best_name]
    best_model.fit(X, y)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": best_model,
        "model_name": best_name,
        "metrics": metrics,
        "classes": sorted(y.unique().tolist()),
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
    }
    joblib.dump(bundle, MODEL_PATH)
    return bundle


if __name__ == "__main__":
    model_bundle = train_and_save()
    print(f"Модель сохранена: {MODEL_PATH}")
    print(f"Лучшая модель: {model_bundle['model_name']}")
    for name, values in model_bundle["metrics"].items():
        print(
            f"{name}: accuracy={values['accuracy']:.4f}, "
            f"macro_f1={values['macro_f1']:.4f}, weighted_f1={values['weighted_f1']:.4f}"
        )
