"""
Machine-learning layer for the FastAPI backend.

Trains one Random Forest per stock from the price history in SQLite, caches it,
and produces forecasts. Same simple model as model/stock_predictor.py.
"""
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error

from db import get_db

FEATURES = ["Open", "High", "Low", "Volume", "MA7", "MA21", "Lag1"]
TARGET = "Close"

_cache = {}  # keyed by ticker


def load_prices(ticker):
    conn = get_db()
    rows = conn.execute(
        "SELECT date, open, high, low, close, volume FROM prices WHERE ticker=? ORDER BY date",
        (ticker,),
    ).fetchall()
    conn.close()
    df = pd.DataFrame([tuple(r) for r in rows],
                      columns=["Date", "Open", "High", "Low", "Close", "Volume"])
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def add_features(df):
    df = df.copy()
    df["MA7"]  = df["Close"].rolling(7).mean()
    df["MA21"] = df["Close"].rolling(21).mean()
    df["Lag1"] = df["Close"].shift(1)
    return df.dropna().reset_index(drop=True)


def get_model(ticker):
    """Train (once) and cache the model + test-set results for a ticker."""
    if ticker not in _cache:
        df = add_features(load_prices(ticker))
        split = int(len(df) * 0.8)
        train_df, test_df = df.iloc[:split], df.iloc[split:]
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(train_df[FEATURES], train_df[TARGET])
        y_pred = model.predict(test_df[FEATURES])
        mape = mean_absolute_percentage_error(test_df[TARGET], y_pred)
        _cache[ticker] = {
            "model": model, "df": df, "test_df": test_df,
            "y_pred": y_pred, "accuracy": round((1 - mape) * 100, 2),
        }
    return _cache[ticker]


def forecast(ticker, days=30):
    cache = get_model(ticker)
    model, df = cache["model"], cache["df"]
    last = df.iloc[-1].copy()
    date = df["Date"].iloc[-1]
    out = []
    for _ in range(days):
        # Advance to the next trading day (skip Sat/Sun — markets are closed).
        date = date + pd.tseries.offsets.BusinessDay(1)
        row = pd.DataFrame([last[FEATURES].values], columns=FEATURES)
        price = round(float(model.predict(row)[0]), 2)
        out.append({"date": date.strftime("%Y-%m-%d"), "price": price})
        last["Lag1"] = last["Close"] = last["Open"] = price
    return out


def prediction_payload(ticker, days=30):
    cache = get_model(ticker)
    test_df = cache["test_df"].tail(60)
    y_pred = cache["y_pred"][-60:]
    return {
        "ticker": ticker,
        "accuracy": cache["accuracy"],
        "future": forecast(ticker, days),
        "test_dates": test_df["Date"].dt.strftime("%Y-%m-%d").tolist(),
        "test_actual": [round(v, 2) for v in test_df[TARGET].tolist()],
        "test_predicted": [round(float(v), 2) for v in y_pred],
    }


def latest_quote(ticker):
    conn = get_db()
    rows = conn.execute(
        "SELECT close FROM prices WHERE ticker=? ORDER BY date DESC LIMIT 2", (ticker,),
    ).fetchall()
    conn.close()
    if not rows:
        return None, None
    price = round(rows[0]["close"], 2)
    change = round((rows[0]["close"] - rows[1]["close"]) / rows[1]["close"] * 100, 2) if len(rows) > 1 else 0.0
    return price, change


def history(ticker, days=365):
    # True calendar window: keep rows within the last `days` days of the latest date.
    df = load_prices(ticker)
    if not df.empty:
        cutoff = df["Date"].iloc[-1] - pd.Timedelta(days=days)
        df = df[df["Date"] >= cutoff]
    return {
        "ticker": ticker,
        "dates": df["Date"].dt.strftime("%Y-%m-%d").tolist(),
        "close": df["Close"].tolist(),
        "volume": df["Volume"].tolist(),
    }


def summary(ticker):
    df = load_prices(ticker)
    latest, prev = df.iloc[-1], df.iloc[-2]
    change = round(latest["Close"] - prev["Close"], 2)
    return {
        "ticker": ticker,
        "latest_close": round(latest["Close"], 2),
        "change": change,
        "pct_change": round(change / prev["Close"] * 100, 2),
        "high_52w": round(df.tail(252)["High"].max(), 2),
        "low_52w": round(df.tail(252)["Low"].min(), 2),
        "avg_volume": int(df.tail(30)["Volume"].mean()),
        "last_date": latest["Date"].strftime("%Y-%m-%d"),
    }
