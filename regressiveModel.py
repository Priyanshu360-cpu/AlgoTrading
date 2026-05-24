# =========================================================
# STOCK PRICE REGRESSION MODEL
# =========================================================
#
# Predicts:
# NEXT CANDLE CLOSE PRICE
#
# Uses:
# - Multiple stocks
# - Technical indicators
# - Quarterly results
#
# Output:
# Exact next predicted price
#
# =========================================================

import json
import glob
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor

# =========================================================
# CONFIG
# =========================================================

DATA_FOLDER = "Companies"

MODEL_OUTPUT = "stock_price_regression_model.pkl"

# =========================================================
# LOAD DATA
# =========================================================

all_rows = []

json_files = glob.glob(f"{DATA_FOLDER}/*.json")

print(f"Found {len(json_files)} stock files")

# =========================================================
# PROCESS EACH STOCK
# =========================================================

for file in json_files:

    stock_name = (
        file.split("\\")[-1]
        .replace("_with_quarterly_results.json", "")
    )

    print(f"\nProcessing: {stock_name}")

    with open(file, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # =====================================================
    # EACH DAY
    # =====================================================

    for day in dataset["data"]:

        candles = day.get("candles", [])

        quarterly = day.get("latest_quarterly_result", {})

        if len(candles) < 30:
            continue

        # =================================================
        # CLEAN NUMBERS
        # =================================================

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

        # =================================================
        # CANDLE LOOP
        # =================================================

        for i in range(20, len(candles) - 1):

            current = candles[i]
            next_candle = candles[i + 1]

            # =================================================
            # CURRENT CANDLE
            # =================================================

            open_price = current["open"]
            high_price = current["high"]
            low_price = current["low"]
            close_price = current["close"]
            volume = current["volume"]

            # =================================================
            # TARGET
            # =================================================

            next_close = next_candle["close"]

            # =================================================
            # TECHNICAL FEATURES
            # =================================================

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

            # =================================================
            # RSI
            # =================================================

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

            # =================================================
            # FINAL ROW
            # =================================================

            row = {

                # STOCK
                "stock_name": stock_name,

                # PRICE
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,

                # VOLUME
                "volume": volume,
                "relative_volume": relative_volume,

                # MA
                "ma5": ma5,
                "ma10": ma10,
                "ma20": ma20,

                # VOLATILITY
                "std20": std20,

                # MOMENTUM
                "momentum": momentum,

                # RETURNS
                "return_1": return_1,
                "return_5": return_5,

                # CANDLE
                "range": candle_range,
                "body_size": body_size,

                # RSI
                "rsi": rsi,

                # FUNDAMENTALS
                "sales": sales,
                "profit": profit,
                "opm": opm,

                # TARGET
                "target": next_close
            }

            all_rows.append(row)

# =========================================================
# DATAFRAME
# =========================================================

df = pd.DataFrame(all_rows)

print("\nDataset Shape:")
print(df.shape)

# =========================================================
# ENCODE STOCK NAMES
# =========================================================

encoder = LabelEncoder()

df["stock_name_encoded"] = encoder.fit_transform(
    df["stock_name"]
)

df.drop(columns=["stock_name"], inplace=True)

# =========================================================
# FEATURES / TARGET
# =========================================================

X = df.drop(columns=["target"])

y = df["target"]

# =========================================================
# TIME-SAFE SPLIT
# =========================================================

split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# =========================================================
# MODEL
# =========================================================

print("\nTraining regression model...")

model = XGBRegressor(
    n_estimators=500,
    max_depth=10,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)

# =========================================================
# PREDICTIONS
# =========================================================

predictions = model.predict(X_test)

# =========================================================
# METRICS
# =========================================================

mae = mean_absolute_error(y_test, predictions)

mse = mean_squared_error(y_test, predictions)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, predictions)

print("\n===================================")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")
print("===================================")

# =========================================================
# SAMPLE PREDICTIONS
# =========================================================

print("\nSample Predictions:\n")

for i in range(10):

    actual = y_test.iloc[i]

    predicted = predictions[i]

    print(
        f"Actual: {actual:.2f} | "
        f"Predicted: {predicted:.2f}"
    )

# =========================================================
# FEATURE IMPORTANCE
# =========================================================

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop Features:\n")
print(importance.head(20))

# =========================================================
# SAVE
# =========================================================

joblib.dump(model, MODEL_OUTPUT)

joblib.dump(
    encoder,
    "stock_regression_encoder.pkl"
)

print(f"\nSaved model: {MODEL_OUTPUT}")