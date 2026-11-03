from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from xgboost import XGBClassifier


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "volatility_xgb.json"

SYMBOLS = ["AAPL", "AMZN", "JNJ", "JPM", "KO", "NVDA", "TSLA", "XOM"]
FEATURE_COLUMNS = [
    "daily_return",
    "rolling_volatility",
    "close_vs_ma20",
    "RSI",
    "volume_change",
    *[f"symbol_{symbol}" for symbol in SYMBOLS],
]
ID_TO_LABEL = {0: "low", 1: "medium", 2: "high"}

model = XGBClassifier()
model.load_model(str(MODEL_PATH))

app = FastAPI(title="Stock Volatility API")


class PredictionRequest(BaseModel):
    symbol: Literal["AAPL", "AMZN", "JNJ", "JPM", "KO", "NVDA", "TSLA", "XOM"]
    daily_return: float
    rolling_volatility: float = Field(ge=0)
    close_vs_ma20: float
    RSI: float = Field(ge=0, le=100)
    volume_change: float


@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": True}


@app.post("/predict")
def predict_volatility(request: PredictionRequest):
    values = request.model_dump()

    row = {
        "daily_return": values["daily_return"],
        "rolling_volatility": values["rolling_volatility"],
        "close_vs_ma20": values["close_vs_ma20"],
        "RSI": values["RSI"],
        "volume_change": values["volume_change"],
    }

    for symbol in SYMBOLS:
        row[f"symbol_{symbol}"] = int(values["symbol"] == symbol)

    features = pd.DataFrame([row], columns=FEATURE_COLUMNS)
    predicted_id = int(model.predict(features)[0])

    return {
        "symbol": values["symbol"],
        "predicted_volatility_band": ID_TO_LABEL[predicted_id],
    }

@app.get("/predict-latest/{symbol}")
def predict_latest(symbol: str):
    symbol = symbol.upper()

    if symbol not in SYMBOLS:
        raise HTTPException(
            status_code=404,
            detail=f"Desteklenmeyen sembol: {symbol}",
        )

    csv_path = PROJECT_DIR / f"{symbol}_transformed.csv"

    if not csv_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"{symbol} için dönüştürülmüş CSV bulunamadı.",
        )

    data = pd.read_csv(csv_path)

    if data.empty:
        raise HTTPException(
            status_code=422,
            detail=f"{symbol} CSV dosyası boş.",
        )

    data["volume_change"] = data["Volume"].pct_change()
    latest = data.iloc[-1]

    values = {
        "daily_return": latest["daily_return"],
        "rolling_volatility": latest["rolling_volatility"],
        "close_vs_ma20": latest["Close"] / latest["moving_avg_20"] - 1,
        "RSI": latest["RSI"],
        "volume_change": latest["volume_change"],
    }

    if any(pd.isna(value) for value in values.values()):
        raise HTTPException(
            status_code=422,
            detail="CSV'nin son satırında tahmin için gereken göstergeler eksik.",
        )

    row = dict(values)
    for listed_symbol in SYMBOLS:
        row[f"symbol_{listed_symbol}"] = int(symbol == listed_symbol)

    features = pd.DataFrame([row], columns=FEATURE_COLUMNS)
    predicted_id = int(model.predict(features)[0])

    return {
        "symbol": symbol,
        "data_date": str(latest["Date"]),
        "predicted_volatility_band": ID_TO_LABEL[predicted_id],
    }