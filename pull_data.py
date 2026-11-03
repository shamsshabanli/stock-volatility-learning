from datetime import date, timedelta
import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

PROJECT_DIR = Path(__file__).resolve().parent
TICKERS = ["AAPL", "NVDA", "TSLA", "JPM", "XOM", "JNJ", "AMZN", "KO"]
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Bitiş tarihi hariç tutulur; böylece henüz kapanmamış bugünkü barı almayız.
end_date = date.today().isoformat()

for ticker in TICKERS:
    try:
        csv_path = PROJECT_DIR / f"{ticker}.csv"

        if csv_path.exists():
            old_data = pd.read_csv(csv_path)
            old_data["Date"] = old_data["Date"].astype(str).str[:10]

            # Son kayıt gününden başla; arada eksik gün varsa onları da tamamla.
            last_date = pd.to_datetime(old_data["Date"]).max().date()
            start_date = last_date.isoformat()
        else:
            old_data = pd.DataFrame()
            start_date = (date.today() - timedelta(days=730)).isoformat()

        logger.info("%s: %s tarihinden veri çekiliyor", ticker, start_date)
        new_data = yf.Ticker(ticker).history(
            start=start_date,
            end=end_date,
            auto_adjust=True,
        )

        if new_data.empty:
            if old_data.empty:
                raise ValueError(f"{ticker} için indirilen veri boş ve eski CSV yok")
            logger.info("%s: yeni veri yok; mevcut CSV korunuyor", ticker)
            continue

        # Tarihleri borsanın yerel işlem gününe göre YYYY-MM-DD biçimine getir.
        new_data.index = new_data.index.strftime("%Y-%m-%d")
        new_data.index.name = "Date"
        new_data = new_data.reset_index()

        if old_data.empty:
            combined_data = new_data
        else:
            combined_data = pd.concat([old_data, new_data], ignore_index=True)
            combined_data["Date"] = combined_data["Date"].astype(str).str[:10]
            combined_data = combined_data.drop_duplicates(
                subset=["Date"],
                keep="last",
            )
            combined_data = combined_data.sort_values("Date")

        combined_data.to_csv(csv_path, index=False)
        logger.info("%s: CSV güncellendi; toplam %s satır", ticker, len(combined_data))
    except Exception:
        logger.exception("%s: veri indirme/güncelleme başarısız", ticker)
        raise
