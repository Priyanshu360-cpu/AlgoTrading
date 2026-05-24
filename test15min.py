# =========================================================
# TEST HYBRID 15-MIN MODELS
# =========================================================
#
# Predicts:
# 1. Future direction (~1 hour)
# 2. Future price
#
# Uses:
# - 15-minute candles
# - indexes
# - quarterly results
#
# =========================================================

import json
import glob
import joblib
import numpy as np
import pandas as pd

# =========================================================
# LOAD MODELS
# =========================================================

direction_model = joblib.load(
    "hybrid_15min_direction.pkl"
)

price_model = joblib.load(
    "hybrid_15min_price.pkl"
)

encoder = joblib.load(
    "hybrid_15min_encoder.pkl"
)

# =========================================================
# CONFIG
# =========================================================

STOCK_FILE = (
    "Companies/paytm_with_quarterly_results.json"
)

INDEX_FOLDER = "Indexes"

# =========================================================
# CONVERT 1-MIN → 15-MIN
# =========================================================

def convert_to_15min(candles):

    if len(candles) < 15:
        return []

    df = pd.DataFrame(candles)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    df.set_index("timestamp", inplace=True)

    resampled = pd.DataFrame()

    resampled["open"] = (
        df["open"]
        .resample("15min")
        .first()
    )

    resampled["high"] = (
        df["high"]
        .resample("15min")
        .max()
    )

    resampled["low"] = (
        df["low"]
        .resample("15min")
        .min()
    )

    resampled["close"] = (
        df["close"]
        .resample("15min")
        .last()
    )

    resampled["volume"] = (
        df["volume"]
        .resample("15min")
        .sum()
    )

    resampled.dropna(inplace=True)

    output = []

    for timestamp, row in resampled.iterrows():

        output.append({

            "timestamp": timestamp.isoformat(),

            "open": float(row["open"]),

            "high": float(row["high"]),

            "low": float(row["low"]),

            "close": float(row["close"]),

            "volume": float(row["volume"])
        })

    return output

# =========================================================
# LOAD STOCK DATA
# =========================================================

with open(STOCK_FILE, "r", encoding="utf-8") as f:
    dataset = json.load(f)

# =========================================================
# LOAD INDEX DATA
# =========================================================

print("Loading index data...")

index_data = {}

index_files = glob.glob(
    f"{INDEX_FOLDER}/*.json"
)

for file in index_files:

    index_name = (
        file.split("\\")[-1]
        .replace("_historical_data.json", "")
    )

    with open(file, "r", encoding="utf-8") as f:
        idx_dataset = json.load(f)

    lookup = {}

    for day in idx_dataset["data"]:

        candles_15m = convert_to_15min(
            day["candles"]
        )

        for candle in candles_15m:

            lookup[
                candle["timestamp"]
            ] = candle

    index_data[index_name] = lookup

# =========================================================
# STOCK NAME
# =========================================================

stock_name = (
    STOCK_FILE.split("/")[-1]
    .replace(
        "_with_quarterly_results.json",
        ""
    )
)

stock_encoded = encoder.transform(
    [stock_name]
)[0]

print(f"\nTesting Stock: {stock_name}")

# =========================================================
# GET LATEST DAY
# =========================================================

latest_day = dataset["data"][-1]

candles_15m = convert_to_15min(
    latest_day["candles"]
)

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
# USE LATEST VALID CANDLE
# =========================================================

i = len(candles_15m) - 5

current = candles_15m[i]

timestamp = current["timestamp"]

# =========================================================
# PRICE DATA
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

    candles_15m[i - j]["close"]

    for j in range(1, 21)
]

ma5 = np.mean(prev_closes[:5])

ma10 = np.mean(prev_closes[:10])

ma20 = np.mean(prev_closes[:20])

std20 = np.std(prev_closes)

# =========================================================
# MOMENTUM
# =========================================================

momentum = (
    close_price
    - candles_15m[i - 5]["close"]
)

# =========================================================
# RETURNS
# =========================================================

return_1 = (
    (close_price
    - candles_15m[i - 1]["close"])
    / candles_15m[i - 1]["close"]
)

return_5 = (
    (close_price
    - candles_15m[i - 5]["close"])
    / candles_15m[i - 5]["close"]
)

# =========================================================
# VOLUME
# =========================================================

volume_ma5 = np.mean([

    candles_15m[i - j]["volume"]

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

        candles_15m[j + 1]["close"]

        - candles_15m[j]["close"]
    )

    if diff >= 0:
        gains.append(diff)
    else:
        losses.append(abs(diff))

avg_gain = (
    np.mean(gains)
    if gains else 0.0001
)

avg_loss = (
    np.mean(losses)
    if losses else 0.0001
)

rs = avg_gain / avg_loss

rsi = 100 - (100 / (1 + rs))

# =========================================================
# INDEX FEATURES
# =========================================================

nifty_return = 0

niftyit_return = 0

next50_return = 0

gift_return = 0

# ---------------------------------------------------------

if timestamp in index_data.get(
    "NIFTY", {}
):

    c = index_data["NIFTY"][
        timestamp
    ]

    nifty_return = (
        (c["close"] - c["open"])
        / c["open"]
    )

# ---------------------------------------------------------

if timestamp in index_data.get(
    "NIFTYIT", {}
):

    c = index_data["NIFTYIT"][
        timestamp
    ]

    niftyit_return = (
        (c["close"] - c["open"])
        / c["open"]
    )

# ---------------------------------------------------------

if timestamp in index_data.get(
    "NIFTY_NEXT_50", {}
):

    c = index_data[
        "NIFTY_NEXT_50"
    ][timestamp]

    next50_return = (
        (c["close"] - c["open"])
        / c["open"]
    )

# ---------------------------------------------------------

if timestamp in index_data.get(
    "GIFTNIFTY", {}
):

    c = index_data["GIFTNIFTY"][
        timestamp
    ]

    gift_return = (
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
    "niftyit_return": niftyit_return,
    "next50_return": next50_return,
    "gift_return": gift_return,

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
# REAL FUTURE PRICE
# =========================================================

future_index = min(
    i + 4,
    len(candles_15m) - 1
)

future_close = candles_15m[
    future_index
]["close"]

# =========================================================
# OUTPUT
# =========================================================

print("\n===================================")

print(
    f"Current Price : "
    f"{close_price:.2f}"
)

# ---------------------------------------------------------

if direction_prediction == 1:

    print(
        "\nPrediction : "
        "BULLISH (NEXT ~1 HOUR)"
    )

else:

    print(
        "\nPrediction : "
        "BEARISH (NEXT ~1 HOUR)"
    )

# ---------------------------------------------------------

print(
    f"\nUP Probability   : "
    f"{direction_probs[1]*100:.2f}%"
)

print(
    f"DOWN Probability : "
    f"{direction_probs[0]*100:.2f}%"
)

# ---------------------------------------------------------

print(
    f"\nPredicted Future Price : "
    f"{price_prediction:.2f}"
)

print(
    f"Actual Future Price    : "
    f"{future_close:.2f}"
)

# ---------------------------------------------------------

error = abs(
    price_prediction
    - future_close
)

print(
    f"Prediction Error       : "
    f"{error:.2f}"
)

# =========================================================
# DIRECTION CHECK
# =========================================================

real_direction = (
    "UP"
    if future_close > close_price
    else "DOWN"
)

predicted_direction = (
    "UP"
    if price_prediction > close_price
    else "DOWN"
)

print(
    f"\nReal Direction      : "
    f"{real_direction}"
)

print(
    f"Predicted Direction : "
    f"{predicted_direction}"
)

print("===================================")