import yfinance as yf

tickers = ["AAPL", "NVDA", "TSLA", "JPM", "XOM", "JNJ", "AMZN", "KO"]

for ticker in tickers:
    data = yf.Ticker(ticker).history(period="2y")
    data.to_csv(f"{ticker}.csv")
    print(f"{ticker} tamamlandı")
