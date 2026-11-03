import json
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import pandas as pd
import streamlit as st


PROJECT_DIR = Path(__file__).resolve().parent
SYMBOLS = ["AAPL", "NVDA", "TSLA", "JPM", "XOM", "JNJ", "AMZN", "KO"]
API_URL = "http://127.0.0.1:8001"

st.set_page_config(
    page_title="Stock Volatility Dashboard",
    page_icon="📈",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_stock_history(symbol):
    """Read the selected stock's historical data from its CSV file."""
    file_path = PROJECT_DIR / f"{symbol}_transformed.csv"

    if not file_path.exists():
        return pd.DataFrame()

    data = pd.read_csv(file_path)

    if "Date" in data.columns:
        data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
        data = data.dropna(subset=["Date"]).sort_values("Date")

    return data


@st.cache_data(show_spinner=False)
def load_model_metrics():
    """Read the latest model evaluation metrics copied from EC2."""
    file_path = PROJECT_DIR / "model_metrics_ec2.json"

    if not file_path.exists():
        return {}

    return json.loads(file_path.read_text(encoding="utf-8"))


@st.cache_data(ttl=60, show_spinner=False)
def get_latest_prediction(symbol):
    """Ask the EC2 API for the selected stock's latest prediction."""
    url = f"{API_URL}/predict-latest/{symbol}"

    with urlopen(url, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


st.title("Stock Volatility Dashboard")
st.write(
    "Explore the latest volatility prediction, historical trends, "
    "and model performance for the selected stock."
)
st.info(
    "Educational project only. A volatility prediction is a risk signal, "
    "not a standalone investment decision."
)

symbol = st.sidebar.selectbox("Select a stock", SYMBOLS)

stock_data = load_stock_history(symbol)
metrics = load_model_metrics()


# Latest prediction from the EC2 API
st.subheader("Current model prediction")

try:
    prediction = get_latest_prediction(symbol)

    st.metric(
        "Next trading day volatility",
        prediction["predicted_volatility_band"].title(),
    )
    st.caption(f"Data date used: {prediction['data_date']}")

except (URLError, TimeoutError, KeyError, json.JSONDecodeError) as error:
    st.error(
        "Could not reach the prediction API. "
        "Check that the SSH tunnel and EC2 API are running."
    )
    st.caption(str(error))


# Overall model performance
st.subheader("Model performance on test data")

if metrics:
    overall = metrics["overall"]

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Accuracy", f"{overall['accuracy']:.1%}")
    metric_col2.metric("Balanced accuracy", f"{overall['balanced_accuracy']:.1%}")
    metric_col3.metric("Macro F1", f"{overall['macro_f1']:.1%}")

    st.caption(
        f"Calculated using {metrics['test_rows']} test observations."
    )

    # Performance for the selected stock
    ticker_metrics = metrics["per_symbol"].get(symbol)

    if ticker_metrics:
        st.subheader(f"Test performance for {symbol}")

        ticker_col1, ticker_col2 = st.columns(2)
        ticker_col1.metric(
            "Ticker accuracy",
            f"{ticker_metrics['accuracy']:.1%}",
        )
        ticker_col2.metric(
            "Ticker Macro F1",
            f"{ticker_metrics['macro_f1']:.1%}",
        )
else:
    st.warning("Model metrics file not found: model_metrics_ec2.json")


# Historical charts
st.subheader(f"{symbol} historical trends")

if stock_data.empty:
    st.warning(f"Historical data file for {symbol} is missing or empty.")
else:
    price_columns = [
        column
        for column in ["Close", "moving_avg_20"]
        if column in stock_data.columns
    ]

    if price_columns:
        price_history = stock_data.set_index("Date")[price_columns]
        price_history = price_history.rename(
            columns={
                "Close": "Closing price",
                "moving_avg_20": "20-day moving average",
            }
        )

        st.write("Closing price and 20-day moving average")
        st.line_chart(price_history)

    if "rolling_volatility" in stock_data.columns:
        volatility_history = stock_data.set_index("Date")[["rolling_volatility"]]
        volatility_history = volatility_history.rename(
            columns={"rolling_volatility": "Historical volatility"}
        )

        st.write("Realized historical volatility")
        st.line_chart(volatility_history)

    latest_csv_date = stock_data["Date"].max().date()
    st.caption(
        f"Historical chart data is from the local CSV; its latest date is "
        f"{latest_csv_date}. The current prediction comes from the EC2 API."
    )


# Past test examples where the model was incorrect
st.subheader(f"Misclassified test examples for {symbol}")

predictions_path = PROJECT_DIR / "model_predictions.csv"

if predictions_path.exists():
    predictions = pd.read_csv(predictions_path)

    incorrect_examples = predictions[
        (predictions["symbol"] == symbol)
        & (~predictions["correct"])
    ].tail(10).copy()

    if incorrect_examples.empty:
        st.success(f"No incorrect test predictions found for {symbol}.")
    else:
        incorrect_examples["result"] = "Incorrect"

        incorrect_examples = incorrect_examples.rename(
            columns={
                "target_date": "Target date",
                "actual_band": "Actual band",
                "predicted_band": "Predicted band",
                "result": "Result",
            }
        )

        st.dataframe(
            incorrect_examples[
                ["Target date", "Actual band", "Predicted band", "Result"]
            ],
            hide_index=True,
        )
else:
    st.warning("Test predictions file not found: model_predictions.csv")


# A fixed AAPL example from the historical test period
st.subheader("AAPL case study: underestimating risk")

st.markdown("""
- **Problem:** Predict the next trading day's volatility band.
- **Approach:** Use historical market indicators and an XGBoost classifier.
- **What went wrong:** On June 25, 2026, actual volatility was high, but the model predicted medium.
- **What an analyst can do:** Review the market conditions and input indicators around the missed day. Treat the model as a risk signal to investigate, not as a standalone decision.
""")