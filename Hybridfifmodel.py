# =========================================================
# HYBRID 15-MIN STOCK + INDEX AI MODEL
# =========================================================
#
# FEATURES:
# - Converts 1-minute → 15-minute candles
# - Uses STOCK candles
# - Uses INDEX candles
# - Uses quarterly results
#
# PREDICTS:
# 1. Future direction (next ~1 hour)
# 2. Future price
#
# =========================================================

import json
import glob
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import (
    XGBClassifier,
    XGBRegressor
)

# =========================================================
# CONFIG
# =========================================================

COMPANIES_FOLDER = "Companies"
INDICES_FOLDER = "Indexes"

CLASSIFICATION_MODEL = "hybrid_15min_direction.pkl"
REGRESSION_MODEL = "hybrid_15min_price.pkl"

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

    # OHLCV aggregation
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
# LOAD INDEX DATA
# =========================================================

print("Loading index data...")

index_data = {}

index_files = glob.glob(
    f"{INDICES_FOLDER}/*.json"
)

for file in index_files:

    index_name = (
        file.split("\\")[-1]
        .replace("_historical_data.json", "")
    )

    print(f"Loaded Index: {index_name}")

    with open(file, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    candle_lookup = {}

    for day in dataset["data"]:

        candles_15m = convert_to_15min(
            day["candles"]
        )

        for candle in candles_15m:

            candle_lookup[
                candle["timestamp"]
            ] = candle

    index_data[index_name] = candle_lookup

# =========================================================
# FEATURE ENGINEERING
# =========================================================

all_rows = []

company_files = glob.glob(
    f"{COMPANIES_FOLDER}/*.json"
)

print(
    f"\nFound {len(company_files)} company files"
)

# =========================================================
# PROCESS EACH STOCK
# =========================================================

for file in company_files:

    stock_name = (
        file.split("\\")[-1]
        .replace(
            "_with_quarterly_results.json",
            ""
        )
    )

    print(f"\nProcessing: {stock_name}")

    with open(file, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # =====================================================
    # EACH DAY
    # =====================================================

    for day in dataset["data"]:

        candles_15m = convert_to_15min(
            day["candles"]
        )

        quarterly = day.get(
            "latest_quarterly_result",
            {}
        )

        if len(candles_15m) < 30:
            continue

        # =================================================
        # CLEAN QUARTERLY DATA
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

        sales = clean_number(
            quarterly.get("Sales", 0)
        )

        profit = clean_number(
            quarterly.get("Net Profit", 0)
        )

        opm = clean_number(
            quarterly.get("OPM %", 0)
        )

        # =================================================
        # MAIN LOOP
        # =================================================

        for i in range(20, len(candles_15m) - 4):

            current = candles_15m[i]

            # =================================================
            # FUTURE TARGET
            # =================================================

            future_index = min(
                i + 4,
                len(candles_15m) - 1
            )

            future_candle = candles_15m[
                future_index
            ]

            timestamp = current["timestamp"]

            # =================================================
            # CURRENT PRICE
            # =================================================

            open_price = current["open"]

            high_price = current["high"]

            low_price = current["low"]

            close_price = current["close"]

            volume = current["volume"]

            # =================================================
            # FUTURE PRICE
            # =================================================

            future_close = future_candle["close"]

            future_return = (
                (future_close - close_price)
                / close_price
            )

            # =================================================
            # DIRECTION LABEL
            # =================================================

            direction_target = (
                1 if future_return > 0.003
                else 0
            )

            # =================================================
            # MOVING AVERAGES
            # =================================================

            prev_closes = [

                candles_15m[i - j]["close"]

                for j in range(1, 21)
            ]

            ma5 = np.mean(prev_closes[:5])

            ma10 = np.mean(prev_closes[:10])

            ma20 = np.mean(prev_closes[:20])

            std20 = np.std(prev_closes)

            # =================================================
            # MOMENTUM
            # =================================================

            momentum = (
                close_price
                - candles_15m[i - 5]["close"]
            )

            # =================================================
            # RETURNS
            # =================================================

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

            # =================================================
            # VOLUME
            # =================================================

            volume_ma5 = np.mean([

                candles_15m[i - j]["volume"]

                for j in range(1, 6)
            ])

            relative_volume = (
                volume / volume_ma5
                if volume_ma5 > 0 else 1
            )

            # =================================================
            # RSI
            # =================================================

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

            # =================================================
            # INDEX FEATURES
            # =================================================

            nifty_return = 0

            niftyit_return = 0

            next50_return = 0

            gift_return = 0

            # -------------------------------------------------

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

            # -------------------------------------------------

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

            # -------------------------------------------------

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

            # -------------------------------------------------

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

            # =================================================
            # FINAL FEATURE ROW
            # =================================================

            row = {

                # STOCK
                "stock_name": stock_name,

                # PRICE
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,

                # RETURNS
                "return_1": return_1,
                "return_5": return_5,

                # MOVING AVERAGES
                "ma5": ma5,
                "ma10": ma10,
                "ma20": ma20,

                # RSI
                "rsi": rsi,

                # MOMENTUM
                "momentum": momentum,

                # VOLATILITY
                "std20": std20,

                # VOLUME
                "volume": volume,
                "relative_volume": relative_volume,

                # INDEXES
                "nifty_return": nifty_return,
                "niftyit_return": niftyit_return,
                "next50_return": next50_return,
                "gift_return": gift_return,

                # FUNDAMENTALS
                "sales": sales,
                "profit": profit,
                "opm": opm,

                # TARGETS
                "direction_target": direction_target,

                "price_target": future_close
            }

            all_rows.append(row)

# =========================================================
# CREATE DATAFRAME
# =========================================================

df = pd.DataFrame(all_rows)

print("\nDataset Shape:")
print(df.shape)

# =========================================================
# ENCODE STOCK NAMES
# =========================================================

encoder = LabelEncoder()

df["stock_encoded"] = encoder.fit_transform(
    df["stock_name"]
)

df.drop(columns=["stock_name"], inplace=True)

# =========================================================
# FEATURES / TARGETS
# =========================================================

X = df.drop(columns=[
    "direction_target",
    "price_target"
])

y_class = df["direction_target"]

y_reg = df["price_target"]

# =========================================================
# TIME SPLIT
# =========================================================

split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]

X_test = X.iloc[split_index:]

y_class_train = y_class.iloc[:split_index]

y_class_test = y_class.iloc[split_index:]

y_reg_train = y_reg.iloc[:split_index]

y_reg_test = y_reg.iloc[split_index:]

# =========================================================
# CLASSIFICATION MODEL
# =========================================================

print("\nTraining direction model...")

classifier = XGBClassifier(

    n_estimators=400,

    max_depth=8,

    learning_rate=0.03,

    subsample=0.8,

    colsample_bytree=0.8,

    random_state=42,

    eval_metric="logloss"
)

classifier.fit(
    X_train,
    y_class_train
)

# =========================================================
# CLASSIFICATION RESULTS
# =========================================================

class_predictions = classifier.predict(
    X_test
)

accuracy = accuracy_score(
    y_class_test,
    class_predictions
)

print("\n===================================")

print(
    f"Direction Accuracy: "
    f"{accuracy*100:.2f}%"
)

print("===================================\n")

print(
    classification_report(
        y_class_test,
        class_predictions
    )
)

# =========================================================
# REGRESSION MODEL
# =========================================================

print("\nTraining regression model...")

regressor = XGBRegressor(

    n_estimators=500,

    max_depth=8,

    learning_rate=0.03,

    subsample=0.8,

    colsample_bytree=0.8,

    random_state=42
)

regressor.fit(
    X_train,
    y_reg_train
)

# =========================================================
# REGRESSION RESULTS
# =========================================================

reg_predictions = regressor.predict(
    X_test
)

mae = mean_absolute_error(
    y_reg_test,
    reg_predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_reg_test,
        reg_predictions
    )
)

r2 = r2_score(
    y_reg_test,
    reg_predictions
)

print("\n===================================")

print(f"MAE  : {mae:.4f}")

print(f"RMSE : {rmse:.4f}")

print(f"R²   : {r2:.4f}")

print("===================================")

# =========================================================
# SAVE MODELS
# =========================================================

joblib.dump(
    classifier,
    CLASSIFICATION_MODEL
)

joblib.dump(
    regressor,
    REGRESSION_MODEL
)

joblib.dump(
    encoder,
    "hybrid_15min_encoder.pkl"
)

print("\n===================================")
print("Saved 15-minute models successfully")
print("===================================")