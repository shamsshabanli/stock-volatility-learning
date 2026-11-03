import logging
from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
TICKERS = ["AAPL", "NVDA", "TSLA", "JPM", "XOM", "JNJ", "AMZN", "KO"]
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

for ticker in TICKERS:
    try:
        input_path = PROJECT_DIR / f"{ticker}.csv"
        output_path = PROJECT_DIR / f"{ticker}_transformed.csv"
        logger.info("%s: dönüşüm başladı", ticker)
        df = pd.read_csv(input_path)
        if df.empty:
            raise ValueError(f"{input_path.name} boş; dönüştürülecek satır yok")

        df["daily_return"] = df["Close"].pct_change()
        df["rolling_volatility"] = df["daily_return"].rolling(window=20).std()
        df["moving_avg_20"] = df["Close"].rolling(window=20).mean()
        df["gain"] = np.where(df["daily_return"] > 0, df["daily_return"], 0)
        df["loss"] = np.where(df["daily_return"] < 0, df["daily_return"].abs(), 0)
        df["avg_gain"] = df["gain"].rolling(window=14).mean()
        df["avg_loss"] = df["loss"].rolling(window=14).mean()
        df["RS"] = df["avg_gain"] / df["avg_loss"]
        df["RSI"] = 100 - 100 / (1 + df["RS"])
        df.to_csv(output_path, index=False)
        logger.info("%s: dönüşüm tamamlandı; %s satır", ticker, len(df))
    except Exception:
        logger.exception("%s: dönüşüm başarısız", ticker)
        raise


