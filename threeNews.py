# =========================================================
# RSS NEWS + SENTIMENT PIPELINE
# =========================================================
#
# Fetches:
# - National News
# - International News
# - Stock News
#
# Uses:
# - Google News RSS
# - Sentiment Analysis
#
# No Google blocking issue
#
# =========================================================

import json
import glob
import time
import requests
import feedparser

from urllib.parse import quote

from vaderSentiment.vaderSentiment import (
    SentimentIntensityAnalyzer
)

# =========================================================
# CONFIG
# =========================================================

COMPANIES_FOLDER = "stock_data"

MAX_NEWS = 5

# =========================================================
# SENTIMENT ANALYZER
# =========================================================

analyzer = SentimentIntensityAnalyzer()

# =========================================================
# FETCH RSS NEWS
# =========================================================

def fetch_rss_news(query):

    rss_url = (
        "https://news.google.com/rss/search?q="
        f"{quote(query)}"
    )

    try:

        feed = feedparser.parse(rss_url)

        news_items = []

        for entry in feed.entries[:MAX_NEWS]:

            title = entry.title

            link = entry.link

            summary = ""

            if hasattr(entry, "summary"):

                summary = entry.summary

            combined_text = (
                title + " " + summary
            )

            # =================================================
            # SENTIMENT SCORE
            # =================================================

            sentiment_score = (
                analyzer.polarity_scores(
                    combined_text
                )["compound"]
            )

            # =================================================
            # SENTIMENT LABEL
            # =================================================

            if sentiment_score >= 0.2:

                sentiment = "POSITIVE"

            elif sentiment_score <= -0.2:

                sentiment = "NEGATIVE"

            else:

                sentiment = "NEUTRAL"

            # =================================================
            # MARKET IMPACT
            # =================================================

            if abs(sentiment_score) >= 0.5:

                impact = "HIGH"

            elif abs(sentiment_score) >= 0.2:

                impact = "MEDIUM"

            else:

                impact = "LOW"

            # =================================================
            # NEWS OBJECT
            # =================================================

            news = {

                "title": title,

                "summary": summary,

                "link": link,

                "sentiment_score": sentiment_score,

                "sentiment": sentiment,

                "market_impact": impact
            }
            print(f"Fetched news: {title} | Sentiment: {sentiment} | Impact: {impact}")
            news_items.append(news)

        return news_items

    except Exception as e:

        print(f"RSS Error: {e}")

        return []

# =========================================================
# PROCESS COMPANY FILES
# =========================================================

company_files = glob.glob(
    f"{COMPANIES_FOLDER}/*.json"
)

print(
    f"Found {len(company_files)} company files"
)

# =========================================================
# MAIN LOOP
# =========================================================

for file in company_files:

    print(f"\nProcessing: {file}")

    stock_name = (
        file.split("\\")[-1]
        .replace(
            "_with_quarterly_results.json",
            ""
        )
    )

    with open(file, "r", encoding="utf-8") as f:

        dataset = json.load(f)

    total_days = len(dataset["data"])

    # =====================================================
    # EACH DAY
    # =====================================================

    for idx, day in enumerate(dataset["data"]):

        date_str = day["date"]

        print(
            f"\n[{idx+1}/{total_days}] "
            f"{stock_name.replace('.json', '')} | {date_str}"
        )

        # =================================================
        # NATIONAL NEWS
        # =================================================

        national_query = (
            f"Indian stock market {date_str}"
        )

        national_news = fetch_rss_news(
            national_query
        )

        print(
            f"National News: "
            f"{len(national_news)}"
        )

        # =================================================
        # INTERNATIONAL NEWS
        # =================================================

        international_query = (
            f"US stock market global economy "
            f"{date_str}"
        )

        international_news = fetch_rss_news(
            international_query
        )

        print(
            f"International News: "
            f"{len(international_news)}"
        )

        # =================================================
        # STOCK SPECIFIC NEWS
        # =================================================

        stock_query = (
            f"{stock_name.replace('.json', '')} stock "
            f"{date_str}"
        )

        stock_news = fetch_rss_news(
            stock_query
        )

        print(
            f"Stock News: "
            f"{len(stock_news)}"
            f"({stock_news[0]['title'] if len(stock_news) > 0 else 'No news'})"
        )

        # =================================================
        # AGGREGATE SENTIMENT
        # =================================================

        all_news = (

            national_news

            + international_news

            + stock_news
        )

        if len(all_news) > 0:

            avg_sentiment = sum(

                x["sentiment_score"]

                for x in all_news

            ) / len(all_news)

        else:

            avg_sentiment = 0

        # =================================================
        # MARKET SIGNAL
        # =================================================

        if avg_sentiment >= 0.2:

            market_signal = "BULLISH"

        elif avg_sentiment <= -0.2:

            market_signal = "BEARISH"

        else:

            market_signal = "NEUTRAL"

        # =================================================
        # FINAL APPEND
        # =================================================

        day["news_analysis"] = {

            "aggregate_sentiment":
                avg_sentiment,

            "market_signal":
                market_signal,

            "national_news":
                national_news,

            "international_news":
                international_news,

            "stock_news":
                stock_news
        }

        # =================================================
        # RATE LIMIT
        # =================================================

        time.sleep(1)

    # =====================================================
    # SAVE FILE
    # =====================================================

    with open(file, "w", encoding="utf-8") as f:

        json.dump(
            dataset,
            f,
            indent=4,
            ensure_ascii=False
        )

    print(f"\nUpdated: {file}")

# =========================================================
# DONE
# =========================================================

print("\n===================================")
print("Finished news enrichment")
print("===================================")