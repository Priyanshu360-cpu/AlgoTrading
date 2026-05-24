import json
import joblib
import numpy as np
import pandas as pd

# =========================================================
# LOAD MODEL
# =========================================================

model = joblib.load("advanced_stock_model.pkl")

encoder = joblib.load("stock_label_encoder.pkl")

# =========================================================
# LOAD ONE STOCK FILE
# =========================================================

FILE = "Companies/paytm_with_quarterly_results.json"

with open(FILE, "r", encoding="utf-8") as f:
    dataset = json.load(f)

# =========================================================
# STOCK NAME
# =========================================================

stock_name = (
    FILE.split("/")[-1]
    .replace("_with_quarterly_results.json", "")
)

stock_encoded = encoder.transform([stock_name])[0]

print(f"Testing stock: {stock_name}")

# =========================================================
# GET LATEST DAY
# =========================================================

latest_day = dataset["data"][-1]

candles = latest_day["candles"]

quarterly = latest_day.get("latest_quarterly_result", {})

# =========================================================
# CLEAN NUMBER
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

sales = clean_number(quarterly.get("Sales", 0))
profit = clean_number(quarterly.get("Net Profit", 0))
opm = clean_number(quarterly.get("OPM %", 0))

# =========================================================
# USE LATEST CANDLE
# =========================================================

i = len(candles) - 2

current = candles[i]

open_price = current["open"]
high_price = current["high"]
low_price = current["low"]
close_price = current["close"]
volume = current["volume"]

# =========================================================
# FEATURES
# =========================================================

prev_closes = [
    candles[i - j]["close"]
    for j in range(1, 21)
]

ma5 = np.mean(prev_closes[:5])
ma10 = np.mean(prev_closes[:10])
ma20 = np.mean(prev_closes[:20])

std20 = np.std(prev_closes)

momentum = close_price - candles[i - 5]["close"]

candle_range = high_price - low_price

body_size = abs(close_price - open_price)

volume_ma5 = np.mean([
    candles[i - j]["volume"]
    for j in range(1, 6)
])

relative_volume = (
    volume / volume_ma5
    if volume_ma5 > 0 else 1
)

return_1 = (
    (close_price - candles[i - 1]["close"])
    / candles[i - 1]["close"]
)

return_5 = (
    (close_price - candles[i - 5]["close"])
    / candles[i - 5]["close"]
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
# BUILD FEATURE VECTOR
# =========================================================

features = pd.DataFrame([{

    "open": open_price,
    "high": high_price,
    "low": low_price,
    "close": close_price,

    "volume": volume,
    "relative_volume": relative_volume,

    "ma5": ma5,
    "ma10": ma10,
    "ma20": ma20,

    "std20": std20,

    "momentum": momentum,

    "return_1": return_1,
    "return_5": return_5,

    "range": candle_range,
    "body_size": body_size,

    "rsi": rsi,

    "sales": sales,
    "profit": profit,
    "opm": opm,

    "stock_name_encoded": stock_encoded
}])

# =========================================================
# PREDICTION
# =========================================================

prediction = model.predict(features)[0]

probability = model.predict_proba(features)[0]

up_probability = probability[1] * 100
down_probability = probability[0] * 100

# =========================================================
# OUTPUT
# =========================================================

print("\n===================================")

print(f"Current Close Price: {close_price}")

if prediction == 1:
    print("\nPrediction: NEXT CANDLE UP")
else:
    print("\nPrediction: NEXT CANDLE DOWN")

print(f"\nUP Probability: {up_probability:.2f}%")
print(f"DOWN Probability: {down_probability:.2f}%")

print("===================================")