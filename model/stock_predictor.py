"""
stock_predictor.py — Simple stock price prediction model for Invock Investments.

Ye model stock ki purani prices parhta hai, un se seekhta hai, apni accuracy
batata hai, aur agle kuch din ki closing price predict karta hai.

Usage:
    python stock_predictor.py                  # OGDC from the built-in data
    python stock_predictor.py HBL              # a different built-in stock
    python stock_predictor.py LUCK 60          # built-in stock + number of days
    python stock_predictor.py my_data.csv      # your own CSV file
    python stock_predictor.py my_data.csv 60   # your CSV + number of days

Apni CSV file mein ye columns hone chahiye: Date, Open, High, Low, Close, Volume
"""
import os
import sys

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error

# Built-in data file ka path (rasta).
DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/psx_stocks.csv")

# Columns the model learns from (FEATURES), and the value it predicts (TARGET).
FEATURES = ["Open", "High", "Low", "Volume", "MA7", "MA21", "Lag1"]
TARGET = "Close"


def load_from_dataset(ticker):
    """Built-in data se aik stock uske ticker (naam) se nikaal kar parho."""
    df = pd.read_csv(DATA_PATH)
    df = df[df["Ticker"] == ticker].copy()
    if df.empty:
        # No data for this ticker -> stop with an error.
        raise ValueError(f"No data found for '{ticker}'")
    df["Date"] = pd.to_datetime(df["Date"], format="mixed")
    # Purani date pehle aaye is tarah sort karo.
    return df.sort_values("Date").reset_index(drop=True)


def load_from_csv(path):
    """Read a single stock's history from your own CSV file."""
    df = pd.read_csv(path)
    # Check karo ke zaroori columns mojood hain.
    needed = ["Date", "Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing these columns: {', '.join(missing)}")
    df["Date"] = pd.to_datetime(df["Date"], format="mixed")
    return df.sort_values("Date").reset_index(drop=True)


def add_features(df):
    """Kuch simple signals add karo jin se model seekh sakay."""
    df = df.copy()
    df["MA7"]  = df["Close"].rolling(7).mean()    # 7-day average price
    df["MA21"] = df["Close"].rolling(21).mean()   # 21-day average price
    df["Lag1"] = df["Close"].shift(1)             # yesterday's close
    # Shuru ki wo rows hata do jin ki average khali hai.
    df = df.dropna().reset_index(drop=True)
    return df


def train(df):
    """Train on the older 80% of data, test on the most recent 20%."""
    split = int(len(df) * 0.8)
    train_df, test_df = df.iloc[:split], df.iloc[split:]

    # Random Forest = bohat saare decision trees jo mil kar faisla karte hain.
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(train_df[FEATURES], train_df[TARGET])

    # Accuracy = how close predictions are to the real prices on the test set.
    preds = model.predict(test_df[FEATURES])
    mape = mean_absolute_percentage_error(test_df[TARGET], preds)
    accuracy = (1 - mape) * 100
    return model, accuracy


def forecast(model, df, days):
    """Agle `days` din ki closing price predict karo."""
    last = df.iloc[-1].copy()
    last_date = df["Date"].iloc[-1]
    results = []
    for i in range(1, days + 1):
        # Predict one day, then feed that prediction back in for the next day.
        row = pd.DataFrame([last[FEATURES].values], columns=FEATURES)
        price = round(float(model.predict(row)[0]), 2)
        results.append({"date": (last_date + pd.Timedelta(days=i)).strftime("%Y-%m-%d"),
                        "price": price})
        last["Lag1"] = price
        last["Close"] = price
        last["Open"] = price
    return results


def run(source="OGDC", days=30):
    # Agar input .csv par khatam ho to file samjho; warna ye aik ticker hai.
    if source.lower().endswith(".csv"):
        df = load_from_csv(source)
        label = os.path.basename(source)
    else:
        df = load_from_dataset(source.upper())
        label = source.upper()

    # Add features, then train the model.
    df = add_features(df)
    model, accuracy = train(df)

    # Results screen par dikhao.
    print(f"\nStock: {label}")
    print(f"Days of history: {len(df)}")
    print(f"Last close: PKR {df['Close'].iloc[-1]:.2f}")
    print(f"Model accuracy: {accuracy:.2f}%")
    print(f"\nNext {days} days:")
    for p in forecast(model, df, days):
        print(f"  {p['date']}   PKR {p['price']:.2f}")


if __name__ == "__main__":
    # Read the inputs typed in the command line (stock/file and days).
    source = sys.argv[1] if len(sys.argv) > 1 else "OGDC"
    days = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    run(source, days)
