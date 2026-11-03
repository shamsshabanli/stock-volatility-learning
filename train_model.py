"""Train and evaluate one time-ordered XGBoost volatility classifier."""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from xgboost import XGBClassifier


PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "model_dataset.csv"
MODEL_DIR = PROJECT_DIR / "models"
MODEL_PATH = MODEL_DIR / "volatility_xgb.json"
PREDICTIONS_PATH = PROJECT_DIR / "model_predictions.csv"
METRICS_PATH = PROJECT_DIR / "model_metrics.json"

LABEL_TO_ID = {"low": 0, "medium": 1, "high": 2}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}
NUMERIC_FEATURES = [
    "daily_return",
    "rolling_volatility",
    "close_vs_ma20",
    "RSI",
    "volume_change",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    data = pd.read_csv(DATA_PATH)
    required = {
        "symbol",
        "Date",
        "target_date",
        "split",
        "volatility_band",
        "next_day_rolling_volatility",
        "Close",
        "moving_avg_20",
        "Volume",
        *NUMERIC_FEATURES[:2],
        "RSI",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"{DATA_PATH.name} eksik sütun(lar) içeriyor: {sorted(missing)}")

    # Bu özellikler sadece tahmin gününün ve önceki günlerin verilerini kullanır.
    data["close_vs_ma20"] = data["Close"] / data["moving_avg_20"] - 1
    data["volume_change"] = data.groupby("symbol", sort=False)["Volume"].pct_change()

    numeric = data[NUMERIC_FEATURES].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    symbol_features = pd.get_dummies(data["symbol"], prefix="symbol", dtype=int)
    features = pd.concat([numeric, symbol_features], axis=1)
    labels = data["volatility_band"].map(LABEL_TO_ID)

    usable = (
        features.notna().all(axis=1)
        & labels.notna()
        & data["split"].isin(["train", "test"])
    )
    data = data.loc[usable].copy()
    features = features.loc[usable].copy()
    labels = labels.loc[usable].astype(int)

    train_mask = data["split"] == "train"
    test_mask = data["split"] == "test"
    if not train_mask.any() or not test_mask.any():
        raise ValueError("Eğitim veya test verisi boş; model eğitilemiyor")
    if set(labels.loc[train_mask].unique()) != set(LABEL_TO_ID.values()):
        raise ValueError("Eğitim verisinde low, medium ve high sınıflarının tümü bulunmalı")

    x_train = features.loc[train_mask]
    y_train = labels.loc[train_mask]
    x_test = features.loc[test_mask]
    y_test = labels.loc[test_mask]
    test_data = data.loc[test_mask].copy()

    logger.info("Eğitim satırı: %s; test satırı: %s", len(x_train), len(x_test))
    logger.info("Özellik sütunu sayısı: %s", len(features.columns))

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=42,
        n_jobs=2,
    )
    model.fit(x_train, y_train, verbose=False)
    predicted_ids = model.predict(x_test).astype(int)

    accuracy = accuracy_score(y_test, predicted_ids)
    balanced_accuracy = balanced_accuracy_score(y_test, predicted_ids)
    macro_f1 = f1_score(
        y_test,
        predicted_ids,
        labels=[0, 1, 2],
        average="macro",
        zero_division=0,
    )
    logger.info("Test accuracy: %.3f", accuracy)
    logger.info("Test balanced accuracy: %.3f", balanced_accuracy)
    logger.info("Test macro F1: %.3f", macro_f1)
    logger.info(
        "Test classification report:\n%s",
        classification_report(
            y_test,
            predicted_ids,
            labels=[0, 1, 2],
            target_names=["low", "medium", "high"],
            zero_division=0,
        ),
    )
    logger.info(
        "Confusion matrix (rows=actual, columns=predicted; low, medium, high):\n%s",
        confusion_matrix(y_test, predicted_ids, labels=[0, 1, 2]),
    )

    prediction_output = test_data[
        ["symbol", "Date", "target_date", "next_day_rolling_volatility", "volatility_band"]
    ].copy()
    prediction_output = prediction_output.rename(columns={"volatility_band": "actual_band"})
    prediction_output["predicted_band"] = [ID_TO_LABEL[value] for value in predicted_ids]
    prediction_output["correct"] = (
        prediction_output["actual_band"] == prediction_output["predicted_band"]
    )
    prediction_output.to_csv(PREDICTIONS_PATH, index=False)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(MODEL_PATH)

    per_symbol = {}
    for symbol, group in prediction_output.groupby("symbol", sort=True):
        actual_ids = group["actual_band"].map(LABEL_TO_ID)
        predicted_symbol_ids = group["predicted_band"].map(LABEL_TO_ID)
        per_symbol[symbol] = {
            "test_rows": int(len(group)),
            "accuracy": float(accuracy_score(actual_ids, predicted_symbol_ids)),
            "macro_f1": float(
                f1_score(
                    actual_ids,
                    predicted_symbol_ids,
                    labels=[0, 1, 2],
                    average="macro",
                    zero_division=0,
                )
            ),
        }

    metadata = {
        "model": "XGBoost XGBClassifier",
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "train_target_date_end": str(data.loc[train_mask, "target_date"].max()),
        "test_target_date_start": str(data.loc[test_mask, "target_date"].min()),
        "features": list(features.columns),
        "label_to_id": LABEL_TO_ID,
        "overall": {
            "accuracy": float(accuracy),
            "balanced_accuracy": float(balanced_accuracy),
            "macro_f1": float(macro_f1),
        },
        "per_symbol": per_symbol,
    }
    METRICS_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    logger.info("Model kaydedildi: %s", MODEL_PATH)
    logger.info("Test tahminleri kaydedildi: %s", PREDICTIONS_PATH.name)
    logger.info("Metrikler kaydedildi: %s", METRICS_PATH.name)


if __name__ == "__main__":
    main()
