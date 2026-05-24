import requests
import json
import time
import os
from datetime import datetime, timedelta

# ============================================
# STOCKS TO DOWNLOAD
# ============================================

STOCKS = {
   
    "AXISBANK":"NSE_EQ|INE238A01034",
    "NESTLE_INDIA":"NSE_EQ|INE239A01024",
    "ICICIBANK":"NSE_EQ|INE090A01021",
    "JSWSTEEL":"NSE_EQ|INE019A01038",
    "BAJAJFINSV":"NSE_EQ|INE918I01026",
    "HINDALCO":"NSE_EQ|INE038A01020",
    "ADANIGREEN":"NSE_EQ|INE364U01010",
    "CHOLAFIN":"NSE_EQ|INE121A01024",
    "MFSL":"NSE_EQ|INE180A01020",
    "ASTRAL":"NSE_EQ|INE006I01046",
    "DALBHARAT":"NSE_EQ|INE00R701025",
    "SHRIRAMFIN":"NSE_EQ|INE721A01047",
    "NILE":"NSE_EQ|INE445D01013",
    "VELJAN":"NSE_EQ|INE753K01015",
    "PROV":"NSE_EQ|INE0ONE01016",
    "HAVELLS":"NSE_EQ|INE176B01034",
    "NBIFIN":"NSE_EQ|INE365I01020",
    "WELINV":"NSE_EQ|INE389K01018",
    "AMBICAAGAR":"NSE_EQ|INE792B01012",
    "GKW":"NSE_EQ|INE528A01020",
    "GRPLTD":"NSE_EQ|INE137I01015",
    "JINDALPOLY":"NSE_EQ|IN3320220087",
    "NITTAGELA":"NSE_EQ|INE265B01019",
    "VENUSREM":"NSE_EQ|INE411B01019",
    "ANONDITA":"NSE_EQ|INE0VTV01012",
    "NIBE":"NSE_EQ|INE149O01018",
    "DSSL":"NSE_EQ|INE417B01040",
    "XPROINDIA":"NSE_EQ|INE445C01015",
    "SASKEN":"NSE_EQ|INE231F01020",
    "DODLA":"NSE_EQ|INE021O01019"


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