# =========================================================
# HYBRID STOCK + INDEX AI MODEL
# =========================================================
#
# FEATURES:
# - Uses STOCK candles
# - Uses INDEX candles
# - Uses quarterly results
# - Predicts:
#       1. Direction (UP/DOWN)
#       2. Exact next price
#
# MODELS:
# - XGBClassifier
# - XGBRegressor
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

CLASSIFICATION_MODEL = "hybrid_direction_model.pkl"
REGRESSION_MODEL = "hybrid_price_model.pkl"

# =========================================================
# LOAD INDEX DATA
# =========================================================

print("Loading index data...")

index_data = {}

index_files = glob.glob(f"{INDICES_FOLDER}/*.json")

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

        for candle in day["candles"]:

            candle_lookup[
                candle["timestamp"]
            ] = candle

    index_data[index_name] = candle_lookup

# =========================================================
# FEATURE ENGINEERING
# =========================================================

all_rows = []

company_files = glob.glob(f"{COMPANIES_FOLDER}/*.json")

print(f"\nFound {len(company_files)} company files")

# =========================================================
# PROCESS EACH STOCK
# =========================================================

for file in company_files:

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

        quarterly = day.get(
            "latest_quarterly_result",
            {}
        )

        if len(candles) < 30:
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
        # CANDLE LOOP
        # =================================================

        for i in range(20, len(candles) - 1):

            current = candles[i]
            next_candle = candles[i + 1]

            timestamp = current["timestamp"]

            # =================================================
            # STOCK FEATURES
            # =================================================

            open_price = current["open"]
            high_price = current["high"]
            low_price = current["low"]
            close_price = current["close"]
            volume = current["volume"]

            # =================================================
            # MOVING AVERAGES
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

            # =================================================
            # RETURNS
            # =================================================

            return_1 = (
                (close_price - candles[i - 1]["close"])
                / candles[i - 1]["close"]
            )

            return_5 = (
                (close_price - candles[i - 5]["close"])
                / candles[i - 5]["close"]
            )

            # =================================================
            # VOLUME
            # =================================================

            volume_ma5 = np.mean([
                candles[i - j]["volume"]
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
                    candles[j + 1]["close"]
                    - candles[j]["close"]
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
            banknifty_return = 0
            niftyit_return = 0
            giftnifty_return = 0

            # -------------------------------------------------

            if timestamp in index_data.get("NIFTY", {}):

                nifty_candle = index_data[
                    "NIFTY"
                ][timestamp]

                nifty_return = (
                    (nifty_candle["close"]
                    - nifty_candle["open"])
                    / nifty_candle["open"]
                )

            # -------------------------------------------------

            if timestamp in index_data.get("NIFTYIT", {}):

                niftyit_candle = index_data[
                    "NIFTYIT"
                ][timestamp]

                niftyit_return = (
                    (niftyit_candle["close"]
                    - niftyit_candle["open"])
                    / niftyit_candle["open"]
                )

            # -------------------------------------------------

            if timestamp in index_data.get(
                "NIFTY_NEXT_50", {}
            ):

                next50_candle = index_data[
                    "NIFTY_NEXT_50"
                ][timestamp]

                banknifty_return = (
                    (next50_candle["close"]
                    - next50_candle["open"])
                    / next50_candle["open"]
                )

            # -------------------------------------------------

            if timestamp in index_data.get(
                "GIFTNIFTY", {}
            ):

                gift_candle = index_data[
                    "GIFTNIFTY"
                ][timestamp]

                giftnifty_return = (
                    (gift_candle["close"]
                    - gift_candle["open"])
                    / gift_candle["open"]
                )

            # =================================================
            # TARGETS
            # =================================================

            next_close = next_candle["close"]

            future_return = (
                (next_close - close_price)
                / close_price
            )

            direction_target = (
                1 if future_return > 0 else 0
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

                # MA
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
                "banknifty_return": banknifty_return,
                "niftyit_return": niftyit_return,
                "giftnifty_return": giftnifty_return,

                # FUNDAMENTALS
                "sales": sales,
                "profit": profit,
                "opm": opm,

                # TARGETS
                "direction_target": direction_target,
                "price_target": next_close
            }

            all_rows.append(row)

# =========================================================
# DATAFRAME
# =========================================================

df = pd.DataFrame(all_rows)

print("\nDataset Shape:")
print(df.shape)

# =========================================================
# ENCODE STOCK NAME
# =========================================================

encoder = LabelEncoder()

df["stock_encoded"] = encoder.fit_transform(
    df["stock_name"]
)

df.drop(columns=["stock_name"], inplace=True)

# =========================================================
# FEATURES
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

# FEATURES
X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

# CLASSIFICATION TARGET
y_class_train = y_class.iloc[:split_index]
y_class_test = y_class.iloc[split_index:]

# REGRESSION TARGET
y_reg_train = y_reg.iloc[:split_index]
y_reg_test = y_reg.iloc[split_index:]

# =========================================================
# CLASSIFICATION MODEL
# =========================================================

print("\nTraining direction model...")

classifier = XGBClassifier(
    n_estimators=400,
    max_depth=10,
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
# CLASSIFICATION EVALUATION
# =========================================================

class_predictions = classifier.predict(X_test)

accuracy = accuracy_score(
    y_class_test,
    class_predictions
)

print("\n===================================")
print(f"Direction Accuracy: {accuracy*100:.2f}%")
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
    max_depth=10,
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
# REGRESSION EVALUATION
# =========================================================

reg_predictions = regressor.predict(X_test)

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
    "hybrid_stock_encoder.pkl"
)

print("\nSaved all models successfully")