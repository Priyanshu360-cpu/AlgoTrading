import requests
import json
import time
import os
from datetime import datetime, timedelta

# ============================================
# STOCKS TO DOWNLOAD
# ============================================

STOCKS = {
    # "ANTHEM_BIOSCIENCES": "NSE_EQ|INE0CZ201020",
    # "PRICOL": "NSE_EQ|INE726V01018",
    # "BOROSIL_RENEWABLES": "NSE_EQ|INE666D01022",
    # "POWERICA": "NSE_EQ|INE921L01032",
    # "CARRARO":"NSE_EQ|INE0V7W01012",
    # "ANTELOPUS":"NSE_EQ|INE818A01017",
    # "ANAND RATHI":"NSE_EQ|INE549H01021",
    # "VARROC":"NSE_EQ|INE665L01035",
    # "RISHABH_INSTRUMENTS":"NSE_EQ|INE0N2P01017",
    # "RANE":"NSE_EQ|INE050H01012",
    # "MONTECARLO":"NSE_EQ|INE423A01024",
    # "VMART":"NSE_EQ|INE665J01013",
    # "ARVIND_SMARTSPACES":"NSE_EQ|INE034S01021",
    # "MINDACORP":"NSE_EQ|INE842C01021",
    # "TTK_PRESTIGE":"NSE_EQ|INE690A01028",
    # "FRACTAL_ANALYTICS":"NSE_EQ|INE212S01015",
    # "GLOBUS_SPIRITS":"NSE_EQ|INE615I01010",
    # "STOVEKRAFT":"NSE_EQ|INE00IN01015",
    # "MANGALAM_CEMENT":"NSE_EQ|INE094A01015",
    # "GRAPHITE_INDIA":"NSE_EQ|INE371A01025",
    "STAR_HEALTH_& AL_INS":"NSE_EQ|INE575P01011",
    "PSP_PROJECTS":"NSE_EQ|INE488V01015",
    "PVR_INOX":"NSE_EQ|INE191H01014",
    "SHEELA_FOAM":"NSE_EQ|INE916U01025",
    "THE_RAMCO_CEMENTS":"NSE_EQ|INE331A01037",
    "DIVGI_TORQTRANSFER":"NSE_EQ|INE753U01022",
    "SOLARA_ACTIVE":"NSE_EQ|INE624Z01016",
    "GE_POWER":"NSE_EQ|INE878A01011",
    "HINDUSTAN_UNILEVER":"NSE_EQ|INE030A01027"


}

START_DATE = datetime(2024, 1, 1)
END_DATE   = datetime(2026, 5, 14)

INTERVAL = "1minute"

OUTPUT_DIR = "stock_data"

ACCESS_TOKEN = ""

# ============================================
# API CONFIG
# ============================================

BASE_URL = "https://api.upstox.com/v2/historical-candle"

headers = {
    "Accept": "application/json",
    "Content-Type": "application/json"
}

# Uncomment if required
# headers["Authorization"] = f"Bearer {ACCESS_TOKEN}"

# ============================================
# HELPERS
# ============================================

def is_weekday(date_obj):
    return date_obj.weekday() < 5

def format_date(date_obj):
    return date_obj.strftime("%Y-%m-%d")

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================
# DOWNLOAD FUNCTION
# ============================================

def download_stock(symbol, instrument_key):

    print(f"\n{'='*50}")
    print(f"Downloading {symbol}")
    print(f"{'='*50}")

    dataset = {
        "symbol": symbol,
        "instrument_key": instrument_key,
        "interval": INTERVAL,
        "generated_at": datetime.now().isoformat(),
        "data": []
    }

    current_date = START_DATE

    while current_date <= END_DATE:

        if not is_weekday(current_date):
            current_date += timedelta(days=1)
            continue

        to_date = format_date(current_date)
        from_date = format_date(current_date - timedelta(days=1))

        url = (
            f"{BASE_URL}/"
            f"{instrument_key}/"
            f"{INTERVAL}/"
            f"{to_date}/"
            f"{from_date}"
        )

        print(f"{symbol} -> {to_date}")

        try:
            response = requests.get(
                url,
                headers=headers,
                timeout=30
            )

            if response.status_code != 200:
                print(f"Failed: {response.status_code}")
                current_date += timedelta(days=1)
                continue

            result = response.json()

            candles = result.get(
                "data",
                {}
            ).get(
                "candles",
                []
            )

            candles.reverse()

            if not candles:
                current_date += timedelta(days=1)
                continue

            day_data = {
                "date": to_date,
                "candles": []
            }

            for candle in candles:

                day_data["candles"].append({
                    "timestamp": candle[0],
                    "open": candle[1],
                    "high": candle[2],
                    "low": candle[3],
                    "close": candle[4],
                    "volume": candle[5],
                    "oi": candle[6]
                })

            dataset["data"].append(day_data)

            print(
                f"Saved {len(candles)} candles"
            )

            time.sleep(0.5)

        except Exception as e:
            print("Error:", e)

        current_date += timedelta(days=1)

    # Save individual stock file
    filename = os.path.join(
        OUTPUT_DIR,
        f"{symbol}.json"
    )

    with open(
        filename,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            dataset,
            f,
            indent=4
        )

    print(f"\nSaved: {filename}")

# ============================================
# MAIN LOOP
# ============================================

for symbol, instrument_key in STOCKS.items():
    download_stock(symbol, instrument_key)

print("\nAll downloads completed.")