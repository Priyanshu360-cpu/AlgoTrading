import json
import requests
import os
from bs4 import BeautifulSoup
from datetime import datetime

# =========================================================
# CONFIG
# =========================================================

INPUT_DIR = "stock_data"
OUTPUT_DIR = "stock_data_with_quarters"

os.makedirs(OUTPUT_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

# =========================================================
# HELPERS
# =========================================================

def quarter_to_date(quarter_label):
    try:
        return datetime.strptime(quarter_label, "%b %Y")
    except:
        return None

# =========================================================
# DOWNLOAD QUARTERLY RESULTS
# =========================================================

def get_quarterly_results(company_name):

    url = (
        f"https://www.screener.in/company/"
        f"{company_name.replace('_','')}/consolidated/"
    )

    print(f"Fetching quarterly data for {company_name}")

    response = requests.get(
        url,
        headers=HEADERS,
        timeout=30
    )

    if response.status_code != 200:
        print(f"Failed: {company_name}")
        return []

    soup = BeautifulSoup(
        response.text,
        "html.parser"
    )

    quarterly_table = None

    for table in soup.find_all("table"):

        text = table.get_text(
            " ",
            strip=True
        ).lower()

        if "quarters" in text or "sales" in text:
            quarterly_table = table
            break

    if quarterly_table is None:
        print(
            f"Quarter table not found for "
            f"{company_name}"
        )
        return []

    rows = quarterly_table.find_all("tr")

    if len(rows) == 0:
        return []

    quarters = []

    for th in rows[0].find_all(
        ["th", "td"]
    )[1:]:

        quarter_name = th.get_text(
            strip=True
        )

        if quarter_name:
            quarters.append(
                quarter_name
            )

    quarterly_data = {}

    for row in rows[1:]:

        cols = row.find_all(
            ["th", "td"]
        )

        if len(cols) < 2:
            continue

        metric_name = cols[0].get_text(
            strip=True
        )

        quarterly_data[
            metric_name
        ] = [
            c.get_text(strip=True)
            for c in cols[1:]
        ]

    structured_quarters = []

    for i, quarter in enumerate(quarters):

        quarter_info = {
            "quarter": quarter
        }

        for metric, values in quarterly_data.items():

            if i < len(values):
                quarter_info[metric] = values[i]

        q_date = quarter_to_date(
            quarter
        )

        if q_date:
            quarter_info[
                "quarter_date"
            ] = q_date

            structured_quarters.append(
                quarter_info
            )

    structured_quarters.sort(
        key=lambda x: x["quarter_date"]
    )

    return structured_quarters

# =========================================================
# PROCESS ONE STOCK FILE
# =========================================================

def process_stock_file(filepath):

    symbol = os.path.splitext(
        os.path.basename(filepath)
    )[0]

    print(
        f"\nProcessing {symbol}"
    )

    with open(
        filepath,
        "r",
        encoding="utf-8"
    ) as f:

        dataset = json.load(f)

    quarters = get_quarterly_results(
        symbol
    )

    if len(quarters) == 0:
        print(
            f"No quarterly data "
            f"for {symbol}"
        )
        return

    for day in dataset["data"]:

        day_date = datetime.strptime(
            day["date"],
            "%Y-%m-%d"
        )

        applicable = None

        for q in quarters:

            if q["quarter_date"] <= day_date:
                applicable = q
            else:
                break

        if applicable:

            quarter_copy = {
                k: v
                for k, v in applicable.items()
                if k != "quarter_date"
            }

            day[
                "latest_quarterly_result"
            ] = quarter_copy

    output_file = os.path.join(
        OUTPUT_DIR,
        f"{symbol}_with_quarters.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            dataset,
            f,
            indent=4,
            ensure_ascii=False
        )

    print(
        f"Saved -> {output_file}"
    )

# =========================================================
# MAIN
# =========================================================

json_files = [
    os.path.join(INPUT_DIR, f)
    for f in os.listdir(INPUT_DIR)
    if f.endswith(".json")
]

print(
    f"Found {len(json_files)} stock files"
)

for file in json_files:
    process_stock_file(file)

print("\nDone.")