"""Load transformed stock CSVs into PostgreSQL, updating existing dates safely."""

import csv
import logging
import os
from datetime import date
from pathlib import Path

import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import execute_values


PROJECT_DIR = Path(__file__).resolve().parent
TICKERS = ["AAPL", "NVDA", "TSLA", "JPM", "XOM", "JNJ", "AMZN", "KO"]
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def as_number(value, cast=float):
    """Convert a CSV cell to a number; blank indicator cells become SQL NULL."""
    if value is None or value == "":
        return None
    return cast(float(value))


def read_rows(ticker):
    path = PROJECT_DIR / f"{ticker}_transformed.csv"
    with path.open(newline="", encoding="utf-8") as csvfile:
        for row in csv.DictReader(csvfile):
            # The source CSV dates are YYYY-MM-DD. PostgreSQL stores them as DATE.
            day = date.fromisoformat(row["Date"][:10])
            yield row, day


def main():
    load_dotenv(PROJECT_DIR / ".env")
    required = ["DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD"]
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise RuntimeError(f".env içinde eksik ayar(lar): {', '.join(missing)}")

    try:
        connection = psycopg2.connect(
            host=os.environ["DB_HOST"],
            port=os.environ["DB_PORT"],
            dbname=os.environ["DB_NAME"],
            user=os.environ["DB_USER"],
            password=os.environ["DB_PASSWORD"],
        )
        logger.info("PostgreSQL bağlantısı kuruldu")
    except psycopg2.Error:
        logger.exception("PostgreSQL bağlantısı başarısız; bağlantı ayarlarını kontrol et")
        raise

    daily_sql = """
        INSERT INTO daily_prices
            (symbol, date, open, high, low, close, volume, dividends, stock_splits)
        VALUES %s
        ON CONFLICT (symbol, date) DO UPDATE SET
            open = EXCLUDED.open,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            close = EXCLUDED.close,
            volume = EXCLUDED.volume,
            dividends = EXCLUDED.dividends,
            stock_splits = EXCLUDED.stock_splits
    """
    indicators_sql = """
        INSERT INTO indicators
            (symbol, date, daily_return, rolling_volatility, moving_avg_20)
        VALUES %s
        ON CONFLICT (symbol, date) DO UPDATE SET
            daily_return = EXCLUDED.daily_return,
            rolling_volatility = EXCLUDED.rolling_volatility,
            moving_avg_20 = EXCLUDED.moving_avg_20
    """

    try:
        with connection:
            with connection.cursor() as cursor:
                for ticker in TICKERS:
                    logger.info("%s: veritabanına yükleme başladı", ticker)
                    rows = list(read_rows(ticker))
                    if not rows:
                        raise ValueError(f"{ticker}_transformed.csv boş; yüklenecek satır yok")
                    daily_values = []
                    indicator_values = []
                    for row, day in rows:
                        daily_values.append(
                            (
                                ticker,
                                day,
                                as_number(row["Open"]),
                                as_number(row["High"]),
                                as_number(row["Low"]),
                                as_number(row["Close"]),
                                as_number(row["Volume"], int),
                                as_number(row["Dividends"]),
                                as_number(row["Stock Splits"]),
                            )
                        )
                        indicator_values.append(
                            (
                                ticker,
                                day,
                                as_number(row["daily_return"]),
                                as_number(row["rolling_volatility"]),
                                as_number(row["moving_avg_20"]),
                            )
                        )

                    execute_values(cursor, daily_sql, daily_values, page_size=500)
                    execute_values(cursor, indicators_sql, indicator_values, page_size=500)
                    logger.info("%s: %s satır yüklendi/güncellendi", ticker, len(rows))
        logger.info("Tüm ticker verileri veritabanına yüklendi")
    except Exception:
        logger.exception("Veritabanına yükleme başarısız; işlem geri alındı")
        raise
    finally:
        connection.close()


if __name__ == "__main__":
    main()
