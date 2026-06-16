# Stock Prediction Model

The machine-learning part of **Invock Investments**. It predicts the future
closing price of a Pakistan Stock Exchange stock from its past prices.

## How it works

1. **Load** a stock's price history from `../data/psx_stocks.csv`.
2. **Add features** — a few simple signals the model learns from:
   - `MA7` — 7-day average price
   - `MA21` — 21-day average price
   - `Lag1` — yesterday's close
   - plus the day's Open, High, Low and Volume
3. **Train** a Random Forest on the older 80% of the data.
4. **Test** it on the most recent 20% and report the accuracy.
5. **Forecast** the next N days of closing prices.

## Run it

```bash
pip install -r requirements.txt
python stock_predictor.py            # OGDC, 30-day forecast
python stock_predictor.py HBL        # a different stock
python stock_predictor.py LUCK 60    # stock + number of days
```

## Note

Accuracy is high because the model uses the same day's Open/High/Low to help
predict that day's Close. It is meant for medium- to long-term guidance, not
exact day-trading.
