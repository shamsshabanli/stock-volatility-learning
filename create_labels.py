"""Create next-trading-day volatility bands without training a model."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
TICKERS = ["AAPL", "NVDA", "TSLA", "JPM", "XOM", "JNJ", "AMZN", "KO"]
TRAIN_FRACTION = 0.80
OUTPUT_PATH = PROJECT_DIR / "model_dataset.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def make_next_day_targets(ticker: str) -> pd.DataFrame:
    """Attach the next available trading day's rolling volatility to each row."""
    input_path = PROJECT_DIR / f"{ticker}_transformed.csv"
    df = pd.read_csv(input_path)

    required_columns = {"Date", "rolling_volatility"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"{input_path.name} eksik sütun(lar) içeriyor: {sorted(missing)}")
    if df.empty:
        raise ValueError(f"{input_path.name} boş; etiket üretilemiyor")

    df["Date"] = pd.to_datetime(df["Date"], errors="raise")
    df = df.sort_values("Date").reset_index(drop=True)
    df["symbol"] = ticker

    # Shift(-1), sonraki SATIRIN değerini alır; hafta sonu/tatil günlerini atlar.
    df["target_date"] = df["Date"].shift(-1)
    df["next_day_rolling_volatility"] = df["rolling_volatility"].shift(-1)
    return df


def main() -> None:
    frames = []
    for ticker in TICKERS:
        try:
            frame = make_next_day_targets(ticker)
            frames.append(frame)
            logger.info("%s: sonraki işlem günü hedefleri hazır", ticker)
        except Exception:
            logger.exception("%s: hedef verisi hazırlanamadı", ticker)
            raise

    dataset = pd.concat(frames, ignore_index=True)
    dataset = dataset.dropna(subset=["target_date", "next_day_rolling_volatility"])
    dataset = dataset.sort_values(["target_date", "symbol"]).reset_index(drop=True)

    # Zaman sırasını koru: eski tarihler eğitim, yeni tarihler değerlendirme.
    target_dates = np.sort(dataset["target_date"].unique())
    train_date_count = int(len(target_dates) * TRAIN_FRACTION)
    if train_date_count < 1 or train_date_count >= len(target_dates):
        raise ValueError("Eğitim/değerlendirme dönemlerini ayırmak için yeterli tarih yok")

    last_train_target_date = target_dates[train_date_count - 1]
    train_mask = dataset["target_date"] <= last_train_target_date
    dataset["split"] = np.where(train_mask, "train", "test")

    # Eşikleri her hisse için yalnızca eski eğitim döneminden hesapla.
    dataset["volatility_band"] = pd.NA
    for ticker in TICKERS:
        ticker_train_mask = (dataset["symbol"] == ticker) & train_mask
        train_values = dataset.loc[ticker_train_mask, "next_day_rolling_volatility"]
        if train_values.empty:
            raise ValueError(f"{ticker} için eğitim döneminde hedef volatilite yok")

        low_cutoff = train_values.quantile(1 / 3)
        high_cutoff = train_values.quantile(2 / 3)
        ticker_mask = dataset["symbol"] == ticker
        values = dataset.loc[ticker_mask, "next_day_rolling_volatility"]

        dataset.loc[ticker_mask, "volatility_band"] = np.select(
            [values <= low_cutoff, values <= high_cutoff],
            ["low", "medium"],
            default="high",
        )
        logger.info(
            "%s: eğitim eşikleri low<=%.6f, medium<=%.6f",
            ticker,
            low_cutoff,
            high_cutoff,
        )

    dataset["target_date"] = dataset["target_date"].dt.strftime("%Y-%m-%d")
    dataset["Date"] = dataset["Date"].dt.strftime("%Y-%m-%d")
    dataset.to_csv(OUTPUT_PATH, index=False)

    logger.info(
        "Etiketli veri kaydedildi: %s (%s satır; %s eğitim, %s değerlendirme)",
        OUTPUT_PATH.name,
        len(dataset),
        int((dataset["split"] == "train").sum()),
        int((dataset["split"] == "test").sum()),
    )
    logger.info(
        "Sınıf dağılımı:\n%s",
        dataset.groupby(["symbol", "split", "volatility_band"], observed=True)
        .size()
        .to_string(),
    )


if __name__ == "__main__":
    main()
