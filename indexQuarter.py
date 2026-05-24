import json
import requests
import time
from bs4 import BeautifulSoup
from datetime import datetime

# =========================================================
# CONFIG
# =========================================================

INPUT_JSON_FILE = "SBIN_NSE.json"
OUTPUT_JSON_FILE = "SBIN_with_quarterly_results.json"

# Example:
# TCS
# INFY
# ZOMATO
# RELIANCE
COMPANY_NAME = "SBIN"

# Screener company URL
SCREENER_URL = f"https://www.screener.in/company/{COMPANY_NAME}/consolidated/"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

# =========================================================
# LOAD STOCK DATA
# =========================================================

with open(INPUT_JSON_FILE, "r", encoding="utf-8") as f:
    dataset = json.load(f)

# =========================================================
# SCRAPE QUARTERLY RESULTS
# =========================================================

print("Downloading quarterly results...")

response = requests.get(SCREENER_URL, headers=HEADERS)

if response.status_code != 200:
    raise Exception(f"Failed to fetch Screener page: {response.status_code}")

soup = BeautifulSoup(response.text, "html.parser")

# =========================================================
# FIND QUARTERLY TABLE
# =========================================================

quarterly_section = None

tables = soup.find_all("table")

for table in tables:

    previous_text = table.get_text(" ", strip=True).lower()

    if "quarters" in previous_text or "sales" in previous_text:
        quarterly_section = table
        break

if quarterly_section is None:
    raise Exception("Quarterly table not found")

# =========================================================
# PARSE TABLE
# =========================================================

rows = quarterly_section.find_all("tr")

headers_row = rows[0]

quarters = []

for th in headers_row.find_all(["th", "td"])[1:]:
    quarter_name = th.get_text(strip=True)

    if quarter_name:
        quarters.append(quarter_name)

quarterly_data = {}

for row in rows[1:]:

    cols = row.find_all(["td", "th"])

    if len(cols) < 2:
        continue

    metric_name = cols[0].get_text(strip=True)

    values = []

    for col in cols[1:]:
        values.append(col.get_text(strip=True))

    quarterly_data[metric_name] = values

# =========================================================
# BUILD STRUCTURED QUARTER OBJECTS
# =========================================================

structured_quarters = []

for i, quarter in enumerate(quarters):

    quarter_info = {
        "quarter": quarter
    }

    for metric_name, values in quarterly_data.items():

        if i < len(values):
            quarter_info[metric_name] = values[i]

    structured_quarters.append(quarter_info)

# =========================================================
# CONVERT QUARTER LABELS TO DATES
# =========================================================

def quarter_to_date(quarter_label):

    # Example:
    # Mar 2024
    # Jun 2024

    try:
        return datetime.strptime(quarter_label, "%b %Y")
    except:
        return None

# Sort quarters chronologically
valid_quarters = []

for q in structured_quarters:

    q_date = quarter_to_date(q["quarter"])

    if q_date:
        q["quarter_date"] = q_date
        valid_quarters.append(q)

valid_quarters.sort(key=lambda x: x["quarter_date"])

# =========================================================
# APPEND PREVIOUS QUARTER RESULT TO EACH DAY
# =========================================================

print("Appending quarterly results to daily candles...")

for day in dataset["data"]:

    day_date = datetime.strptime(day["date"], "%Y-%m-%d")

    applicable_quarter = None

    for q in valid_quarters:

        if q["quarter_date"] <= day_date:
            applicable_quarter = q
        else:
            break

    if applicable_quarter:

        clean_quarter = applicable_quarter.copy()

        del clean_quarter["quarter_date"]

        day["latest_quarterly_result"] = clean_quarter

# =========================================================
# SAVE FINAL FILE
# =========================================================

with open(OUTPUT_JSON_FILE, "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=4, ensure_ascii=False)

print("\n====================================")
print("Finished.")
print(f"Saved: {OUTPUT_JSON_FILE}")
print("====================================")