import os
import json
import requests
import time
from datetime import datetime

# =========================================================
# CONFIG
# =========================================================

COMPANIES_FOLDER = "Companies"

# ADD TOKEN IF REQUIRED
ACCESS_TOKEN = ""

HEADERS = {
    "Accept": "application/json",
    "Content-Type": "application/json"
}

# Uncomment if API requires auth
# HEADERS["Authorization"] = f"Bearer {ACCESS_TOKEN}"

# =========================================================
# DATE RANGE
# =========================================================

FROM_DATE = "2026-05-14"
TO_DATE = "2026-05-15"

INTERVAL = "1minute"

# =========================================================
# DOWNLOAD FUNCTION
# =========================================================

def fetch_new_day_data(instrument_key):

    url = (
        f"https://api.upstox.com/v2/historical-candle/"
        f"{instrument_key}/{INTERVAL}/{TO_DATE}/{FROM_DATE}"
    )

    response = requests.get(
        url,
        headers=HEADERS,
        timeout=30
    )

    if response.status_code != 200:

        print(f"API Failed: {response.status_code}")

        return []

    result = response.json()

    candles = result.get("data", {}).get("candles", [])

    # Reverse oldest -> newest
    candles.reverse()

    structured = []

    for candle in candles:

        structured.append({
            "timestamp": candle[0],
            "open": candle[1],
            "high": candle[2],
            "low": candle[3],
            "close": candle[4],
            "volume": candle[5],
            "oi": candle[6]
        })

    return structured

# =========================================================
# PROCESS ALL FILES
# =========================================================

files = [
    f for f in os.listdir(COMPANIES_FOLDER)
    if f.endswith(".json")
]

print(f"Found {len(files)} company files")

for file in files:

    full_path = os.path.join(COMPANIES_FOLDER, file)

    print(f"\nProcessing: {file}")

    with open(full_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # =====================================================
    # GET INSTRUMENT KEY
    # =====================================================

    instrument_key = dataset.get("instrument_key")

    if not instrument_key:

        print("No instrument_key found. Skipping.")
        continue

    print(f"Instrument: {instrument_key}")

    # =====================================================
    # FETCH NEW DAY
    # =====================================================

    candles = fetch_new_day_data(instrument_key)

    if len(candles) == 0:

        print("No candles found")
        continue

    print(f"Downloaded {len(candles)} candles")

    # =====================================================
    # CHECK DUPLICATE DATE
    # =====================================================

    already_exists = False

    for day in dataset["data"]:

        if day["date"] == TO_DATE:
            already_exists = True
            break

    if already_exists:

        print("Date already exists. Skipping append.")
        continue

    # =====================================================
    # COPY PREVIOUS QUARTERLY RESULT
    # =====================================================

    latest_quarter = None

    if len(dataset["data"]) > 0:

        last_day = dataset["data"][-1]

        latest_quarter = last_day.get(
            "latest_quarterly_result",
            {}
        )

    # =====================================================
    # CREATE NEW DAY OBJECT
    # =====================================================

    new_day = {
        "date": TO_DATE,
        "candles": candles
    }

    if latest_quarter:
        new_day["latest_quarterly_result"] = latest_quarter

    # =====================================================
    # APPEND
    # =====================================================

    # dataset["data"].append(new_day)

    # =====================================================
    # SAVE FILE
    # =====================================================

    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(
            dataset,
            f,
            indent=4,
            ensure_ascii=False
        )

    print("Updated file successfully")

    time.sleep(0.5)

print("\n===================================")
print("All company files updated")
print("===================================")