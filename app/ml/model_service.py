from functools import lru_cache

import joblib
import pandas as pd

from app.config import MODEL_PATH
from app.ml.features import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES


class ModelNotTrainedError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def load_model_bundle() -> dict:
    if not MODEL_PATH.exists():
        raise ModelNotTrainedError(
            "Модель не найдена. Сначала запустите: python scripts/train_model.py"
        )
    return joblib.load(MODEL_PATH)


def normalize_payload(payload: dict) -> dict:
    normalized = {}
    for feature in FEATURE_COLUMNS:
        value = payload.get(feature)
        if feature in NUMERIC_FEATURES:
            normalized[feature] = float(value)
        elif feature in CATEGORICAL_FEATURES:
            normalized[feature] = str(value)
        else:
            normalized[feature] = value
    return normalized


def get_top_features(model, limit: int = 8) -> list[dict]:
    classifier = model.named_steps.get("classifier") if hasattr(model, "named_steps") else model
    if not hasattr(classifier, "feature_importances_"):
        return []

    importances = classifier.feature_importances_
    names = CATEGORICAL_FEATURES + NUMERIC_FEATURES
    pairs = sorted(zip(names, importances), key=lambda item: item[1], reverse=True)
    return [
        {"name": name, "importance": round(float(importance), 4)}
        for name, importance in pairs[:limit]
    ]


def predict_risk(payload: dict) -> dict:
    bundle = load_model_bundle()
    model = bundle["model"]
    classes = list(bundle["classes"])
    normalized = normalize_payload(payload)
    frame = pd.DataFrame([normalized], columns=FEATURE_COLUMNS)

    predicted_class = model.predict(frame)[0]
    probabilities = {}
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(frame)[0]
        probabilities = {
            cls: round(float(probability), 4)
            for cls, probability in zip(model.classes_, proba)
        }

    return {
        "prediction": predicted_class,
        "probabilities": probabilities,
        "classes": classes,
        "model_name": bundle["model_name"],
        "metrics": bundle["metrics"],
        "top_features": get_top_features(model),
    }
