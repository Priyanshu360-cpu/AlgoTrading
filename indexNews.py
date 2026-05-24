import json
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote

# =========================================================
# CONFIG
# =========================================================

INPUT_JSON_FILE = "HappiestMinds_historical_data.json"
OUTPUT_JSON_FILE = "HappiestMinds_historical_with_news.json"

COMPANY_NAME = "HappiestMinds"

MAX_NEWS_PER_DAY = 5

# =========================================================
# LOAD EXISTING DATA
# =========================================================

with open(INPUT_JSON_FILE, "r", encoding="utf-8") as f:
    dataset = json.load(f)

# =========================================================
# HEADERS
# =========================================================

headers = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

# =========================================================
# GOOGLE NEWS SCRAPER
# =========================================================

def fetch_google_news(company_name, date_str):

    # Google query
    query = f"{company_name} stock news {date_str}"

    # Google News Search URL
    url = f"https://www.google.com/search?q={quote(query)}&tbm=nws"

    try:

        response = requests.get(url, headers=headers, timeout=30)

        if response.status_code != 200:
            print(f"Google blocked request: {response.status_code}")
            return []

        soup = BeautifulSoup(response.text, "html.parser")

        news_items = []

        # Google news result blocks
        results = soup.select("div.SoaBEf")

        for result in results[:MAX_NEWS_PER_DAY]:

            try:

                title_tag = result.select_one("div.MBeuO")

                source_tag = result.select_one(".NUnG9d span")

                snippet_tag = result.select_one(".GI74Re")

                link_tag = result.find("a")

                date_tag = result.select_one("span.LfVVr")

                news = {
                    "title": title_tag.text.strip() if title_tag else "",
                    "source": source_tag.text.strip() if source_tag else "",
                    "snippet": snippet_tag.text.strip() if snippet_tag else "",
                    "date": date_tag.text.strip() if date_tag else "",
                    "link": (
                        "https://www.google.com" + link_tag["href"]
                        if link_tag and link_tag.get("href")
                        else ""
                    )
                }

                # Skip empty news
                if news["title"] != "":
                    news_items.append(news)

            except Exception as e:
                print(f"Parse error: {e}")

        return news_items

    except Exception as e:
        print(f"Request error: {e}")
        return []

# =========================================================
# APPEND NEWS TO EACH DAY
# =========================================================

total_days = len(dataset["data"])

for idx, day_data in enumerate(dataset["data"], start=1):

    date_str = day_data["date"]

    print(f"\n[{idx}/{total_days}] Fetching news for {date_str}")

    news = fetch_google_news(COMPANY_NAME, date_str)

    if len(news) > 0:

        day_data["news"] = news

        print(f"Added {len(news)} news articles")

    else:
        print("No news found")

    # Delay to avoid blocking
    time.sleep(3)

# =========================================================
# SAVE UPDATED DATASET
# =========================================================

with open(OUTPUT_JSON_FILE, "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=4, ensure_ascii=False)

print("\n======================================")
print("Finished.")
print(f"Saved file: {OUTPUT_JSON_FILE}")
print("======================================")