import json
import joblib
import numpy as np
import pandas as pd

# =========================================================
# LOAD MODELS
# =========================================================

direction_model = joblib.load(
    "hybrid_direction_model.pkl"
)

price_model = joblib.load(
    "hybrid_price_model.pkl"
)

encoder = joblib.load(
    "hybrid_stock_encoder.pkl"
)

# =========================================================
# FILES
# =========================================================

STOCK_FILE = (
    "Companies/paytm_with_quarterly_results.json"
)

INDEX_FOLDER = "Indexes"

# =========================================================
# LOAD STOCK DATA
# =========================================================

with open(STOCK_FILE, "r", encoding="utf-8") as f:
    dataset = json.load(f)

# =========================================================
# LOAD INDEX DATA
# =========================================================

import glob

index_data = {}

index_files = glob.glob(f"{INDEX_FOLDER}/*.json")

for file in index_files:

    index_name = (
        file.split("\\")[-1]
        .replace("_historical_data.json", "")
    )

    with open(file, "r", encoding="utf-8") as f:
        idx_dataset = json.load(f)

    lookup = {}

    for day in idx_dataset["data"]:

        for candle in day["candles"]:

            lookup[
                candle["timestamp"]
            ] = candle

    index_data[index_name] = lookup

# =========================================================
# STOCK NAME
# =========================================================

stock_name = (
    STOCK_FILE.split("/")[-1]
    .replace("_with_quarterly_results.json", "")
)

stock_encoded = encoder.transform(
    [stock_name]
)[0]

print(f"\nTesting Stock: {stock_name}")

# =========================================================
# LATEST DAY
# =========================================================

latest_day = dataset["data"][-1]

candles = latest_day["candles"]

quarterly = latest_day.get(
    "latest_quarterly_result",
    {}
)

# =========================================================
# CLEAN NUMBERS
# =========================================================

def clean_number(value):

    try:
        return float(
            str(value)
            .replace(",", "")
            .replace("%", "")
        )
    except:
        return 0

sales = clean_number(
    quarterly.get("Sales", 0)
)

profit = clean_number(
    quarterly.get("Net Profit", 0)
)

opm = clean_number(
    quarterly.get("OPM %", 0)
)

# =========================================================
# USE SECOND LAST CANDLE
# =========================================================

i = len(candles) - 2

current = candles[i]

timestamp = current["timestamp"]

# =========================================================
# STOCK FEATURES
# =========================================================

open_price = current["open"]
high_price = current["high"]
low_price = current["low"]
close_price = current["close"]
volume = current["volume"]

# =========================================================
# MOVING AVERAGES
# =========================================================

prev_closes = [
    candles[i - j]["close"]
    for j in range(1, 21)
]

ma5 = np.mean(prev_closes[:5])
ma10 = np.mean(prev_closes[:10])
ma20 = np.mean(prev_closes[:20])

std20 = np.std(prev_closes)

momentum = (
    close_price
    - candles[i - 5]["close"]
)

# =========================================================
# RETURNS
# =========================================================

return_1 = (
    (close_price - candles[i - 1]["close"])
    / candles[i - 1]["close"]
)

return_5 = (
    (close_price - candles[i - 5]["close"])
    / candles[i - 5]["close"]
)

# =========================================================
# VOLUME
# =========================================================

volume_ma5 = np.mean([
    candles[i - j]["volume"]
    for j in range(1, 6)
])

relative_volume = (
    volume / volume_ma5
    if volume_ma5 > 0 else 1
)

# =========================================================
# RSI
# =========================================================

gains = []
losses = []

for j in range(i - 14, i):

    diff = (
        candles[j + 1]["close"]
        - candles[j]["close"]
    )

    if diff >= 0:
        gains.append(diff)
    else:
        losses.append(abs(diff))

avg_gain = np.mean(gains) if gains else 0.0001
avg_loss = np.mean(losses) if losses else 0.0001

rs = avg_gain / avg_loss

rsi = 100 - (100 / (1 + rs))

# =========================================================
# INDEX FEATURES
# =========================================================

nifty_return = 0
banknifty_return = 0
niftyit_return = 0
giftnifty_return = 0

# ---------------------------------------------------------

if timestamp in index_data.get("NIFTY", {}):

    c = index_data["NIFTY"][timestamp]

    nifty_return = (
        (c["close"] - c["open"])
        / c["open"]
    )

# ---------------------------------------------------------

if timestamp in index_data.get("NIFTYIT", {}):

    c = index_data["NIFTYIT"][timestamp]

    niftyit_return = (
        (c["close"] - c["open"])
        / c["open"]
    )

# ---------------------------------------------------------

if timestamp in index_data.get(
    "NIFTY_NEXT_50", {}
):

    c = index_data["NIFTY_NEXT_50"][timestamp]

    banknifty_return = (
        (c["close"] - c["open"])
        / c["open"]
    )

# ---------------------------------------------------------

if timestamp in index_data.get(
    "GIFTNIFTY", {}
):

    c = index_data["GIFTNIFTY"][timestamp]

    giftnifty_return = (
        (c["close"] - c["open"])
        / c["open"]
    )

# =========================================================
# FEATURE VECTOR
# =========================================================

features = pd.DataFrame([{

    "open": open_price,
    "high": high_price,
    "low": low_price,
    "close": close_price,

    "return_1": return_1,
    "return_5": return_5,

    "ma5": ma5,
    "ma10": ma10,
    "ma20": ma20,

    "rsi": rsi,

    "momentum": momentum,

    "std20": std20,

    "volume": volume,
    "relative_volume": relative_volume,

    "nifty_return": nifty_return,
    "banknifty_return": banknifty_return,
    "niftyit_return": niftyit_return,
    "giftnifty_return": giftnifty_return,

    "sales": sales,
    "profit": profit,
    "opm": opm,

    "stock_encoded": stock_encoded
}])

# =========================================================
# PREDICTIONS
# =========================================================

direction_prediction = (
    direction_model.predict(features)[0]
)

direction_probs = (
    direction_model.predict_proba(features)[0]
)

price_prediction = (
    price_model.predict(features)[0]
)

# =========================================================
# REAL NEXT PRICE
# =========================================================

actual_next_close = None

if i + 1 < len(candles):

    actual_next_close = candles[i + 1]["close"]

# =========================================================
# OUTPUT
# =========================================================

print("\n===================================")

print(f"Current Price : {close_price:.2f}")

# ---------------------------------------------------------

if direction_prediction == 1:
    print("\nDirection Prediction : UP")
else:
    print("\nDirection Prediction : DOWN")

print(
    f"UP Probability   : "
    f"{direction_probs[1]*100:.2f}%"
)

print(
    f"DOWN Probability : "
    f"{direction_probs[0]*100:.2f}%"
)

# ---------------------------------------------------------

print(
    f"\nPredicted Next Price : "
    f"{price_prediction:.2f}"
)

# ---------------------------------------------------------

if actual_next_close is not None:

    print(
        f"Actual Next Price    : "
        f"{actual_next_close:.2f}"
    )

    error = abs(
        price_prediction
        - actual_next_close
    )

    print(
        f"Prediction Error     : "
        f"{error:.2f}"
    )

    real_direction = (
        "UP"
        if actual_next_close > close_price
        else "DOWN"
    )

    predicted_direction = (
        "UP"
        if price_prediction > close_price
        else "DOWN"
    )

    print(
        f"\nReal Direction       : "
        f"{real_direction}"
    )

    print(
        f"Predicted Direction  : "
        f"{predicted_direction}"
    )

print("===================================")