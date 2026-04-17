"""
Financial Asset Sentiment Analyzer
====================================
Fetches market data and news for a given financial asset, runs VADER-based
sentiment analysis on recent headlines, and produces price and sentiment
charts alongside analyst recommendation data.

Supported asset classes: Indices, Cryptocurrencies, Energies, Metals.
Individual US equities are also supported via a direct ticker lookup.

Dependencies:
    yfinance, matplotlib, vaderSentiment, feedparser
"""

import feedparser
import matplotlib.pyplot as plt
import yfinance as yf

from datetime import datetime, timedelta
from urllib.parse import quote
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# =============================================================================
# Asset Class Definitions
# =============================================================================

ASSET_CLASSES = {
    "Indices": {
        "S&P 500 E-Mini":    "ES=F",
        "Nasdaq 100 E-Mini": "NQ=F",
        "Dow Futures E-Mini":"YM=F",
    },
    "Cryptocurrencies": {
        "Bitcoin":      "BTC-USD",
        "Ether":        "ETH-USD",
        "Tether USDT":  "USDT-USD",
        "XRP":          "XRP-USD",
        "Solana":       "SOL-USD",
    },
    "Energies": {
        "Crude Oil WTI":  "CL=F",
        "Crude Oil Brent":"BZ=F",
        "Gasoline RBOB":  "RB=F",
        "Natural Gas":    "NG=F",
        "Ethanol Chicago":"AC=F",
    },
    "Metals": {
        "Gold":         "GC=F",
        "Silver":       "SI=F",
        "Platinum":     "PL=F",
        "Palladium":    "PA=F",
        "Copper E-mini":"QC=F",
        "Aluminum":     "ALI=F",
    },
}


# =============================================================================
# Market Data Retrieval
# =============================================================================

def get_current_price(ticker_obj):
    """
    Return the most recent market price for the given ticker.

    Tries the 'regularMarketPrice' field from the info dict first; falls back
    to the last closing price from a 1-day history request.

    Returns:
        float or None
    """
    try:
        price = ticker_obj.info.get("regularMarketPrice")
        if price is None:
            hist = ticker_obj.history(period="1d")
            if not hist.empty:
                price = hist["Close"].iloc[-1]
        return price
    except Exception:
        return None


def get_target_price(ticker_obj):
    """
    Return the consensus analyst target price for a stock ticker.

    Returns:
        float or None
    """
    try:
        return ticker_obj.info.get("targetMeanPrice")
    except Exception:
        return None


def get_price_history(ticker_obj, months=6):
    """
    Fetch daily closing prices for the last *months* calendar months.

    Returns:
        pandas.Series or None
    """
    end   = datetime.now()
    start = end - timedelta(days=30 * months)
    hist  = ticker_obj.history(start=start, end=end, interval="1d")
    return hist["Close"] if not hist.empty else None


def get_recommendations(ticker_obj):
    """
    Return the most recent analyst recommendation breakdown for a stock.

    Returns:
        dict with keys period, strongBuy, buy, hold, sell, strongSell, or None
    """
    try:
        rec_df = ticker_obj.recommendations
        if rec_df is None or rec_df.empty:
            return None
        latest = rec_df.iloc[-1]
        return {
            "period":    latest.get("period",    ""),
            "strongBuy": int(latest.get("strongBuy", 0)),
            "buy":       int(latest.get("buy",       0)),
            "hold":      int(latest.get("hold",      0)),
            "sell":      int(latest.get("sell",      0)),
            "strongSell":int(latest.get("strongSell",0)),
        }
    except Exception:
        return None


# =============================================================================
# News Retrieval
# =============================================================================

def get_yfinance_news(ticker_obj, limit=15):
    """
    Pull recent news articles from Yahoo Finance for the given ticker.

    Returns:
        list of dicts with keys: headline, time, publisher, url
    """
    try:
        news = ticker_obj.news
        if not news:
            return []
        articles = []
        for item in news[:limit]:
            articles.append({
                "headline":  item.get("title", ""),
                "time":      item.get("providerPublishTime", None),
                "publisher": item.get("publisher", "Yahoo Finance"),
                "url":       item.get("link", ""),
            })
        return articles
    except Exception:
        return []


def fetch_rss_news(query, num_articles=10):
    """
    Retrieve articles from the Google News RSS feed for the given query string.

    Returns:
        list of dicts with keys: headline, time, publisher, url
    """
    rss_url = (
        f"https://news.google.com/rss/search"
        f"?q={quote(query)}&hl=en-US&gl=US&ceid=US:en"
    )
    feed     = feedparser.parse(rss_url)
    articles = []
    for entry in feed.entries[:num_articles]:
        articles.append({
            "headline":  entry.title,
            "time":      None,
            "publisher": entry.get("source", {}).get("title", "Google News"),
            "url":       entry.link,
        })
    return articles


def fetch_market_news(asset_name, ticker_symbol):
    """
    Aggregate news articles from Yahoo Finance and Google News RSS.

    Deduplication is performed on the headline string so the same story
    appearing in both sources is only counted once.

    Returns:
        list of article dicts
    """
    all_articles   = []
    seen_headlines = set()

    # Yahoo Finance articles
    ticker_obj  = yf.Ticker(ticker_symbol)
    yf_articles = get_yfinance_news(ticker_obj, limit=10)
    for art in yf_articles:
        headline = art["headline"].strip()
        if headline and headline not in seen_headlines:
            seen_headlines.add(headline)
            all_articles.append(art)

    # Google News RSS queries
    queries = [
        asset_name,
        f"{asset_name} market",
        f"{asset_name} price",
        f"{asset_name} news",
        f"{asset_name} analysis",
        f"{asset_name} forecast",
    ]
    queries = list(dict.fromkeys(q for q in queries if q))

    for query in queries:
        for art in fetch_rss_news(query, num_articles=5):
            headline = art["headline"].strip()
            if headline and headline not in seen_headlines:
                seen_headlines.add(headline)
                all_articles.append(art)

    return all_articles


# =============================================================================
# Sentiment Analysis
# =============================================================================

def analyze_sentiment_vader(articles):
    """
    Run VADER sentiment scoring on the headline of each article.

    Each article dict is mutated in place to include 'sentiment_score'
    (float) and 'sentiment_label' (Positive / Neutral / Negative).

    Returns:
        tuple: (overall_score, positive_count, neutral_count, negative_count)
    """
    analyzer = SentimentIntensityAnalyzer()
    scores   = []

    for art in articles:
        score = analyzer.polarity_scores(art["headline"])["compound"]
        scores.append(score)
        art["sentiment_score"] = score
        if score >= 0.05:
            art["sentiment_label"] = "Positive"
        elif score <= -0.05:
            art["sentiment_label"] = "Negative"
        else:
            art["sentiment_label"] = "Neutral"

    if not scores:
        return 0.0, 0, 0, 0

    overall  = sum(scores) / len(scores)
    positive = sum(1 for s in scores if s >=  0.05)
    negative = sum(1 for s in scores if s <= -0.05)
    neutral  = len(scores) - positive - negative

    return overall, positive, neutral, negative


def print_sentiment_summary(articles, total_analyzed):
    """Print a breakdown of Positive / Neutral / Negative article counts."""
    tally = {"Positive": 0, "Neutral": 0, "Negative": 0}
    for art in articles:
        tally[art["sentiment_label"]] += 1

    print("\n--- Market Sentiment Summary ---")
    print(f"Total articles analyzed: {total_analyzed}")
    for label, count in tally.items():
        pct = (count / total_analyzed * 100) if total_analyzed > 0 else 0
        print(f"  {label:<10}: {count:>3}  ({pct:.1f}%)")


# =============================================================================
# Charting / Visualisation
# =============================================================================

def plot_price_history(price_series, name):
    """Render a line chart of the last 6 months of daily closing prices."""
    if price_series is None or price_series.empty:
        print("No price data available for chart.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(price_series.index, price_series.values, color="steelblue", linewidth=2)
    ax.set_title(f"{name} — Last 6 Months Closing Price")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()
    plt.show()


def plot_sentiment_indicator(score):
    """Render a horizontal bar indicator showing the overall VADER score."""
    if score >= 0.05:
        color, label = "green", "Positive"
    elif score <= -0.05:
        color, label = "firebrick", "Negative"
    else:
        color, label = "slategray", "Neutral"

    fig, ax = plt.subplots(figsize=(6, 1.5))
    ax.barh([0], score, color=color, height=0.5, align="center")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-0.5, 0.5)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(f"Overall Sentiment Score: {score:.3f}  ({label})")
    ax.set_xlabel("VADER Compound Score")
    ax.set_yticks([])
    fig.tight_layout()
    plt.show()


def plot_sentiment_distribution(pos, neu, neg):
    """Render a bar chart of Positive / Neutral / Negative article counts."""
    labels = ["Positive", "Neutral", "Negative"]
    counts = [pos, neu, neg]
    colors = ["green", "slategray", "firebrick"]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, counts, color=colors)
    ax.set_title("Market Sentiment Distribution")
    ax.set_ylabel("Number of Articles")
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            str(count),
            ha="center", va="bottom",
        )
    fig.tight_layout()
    plt.show()


def plot_recommendations(rec, name):
    """Render a bar chart of analyst buy / hold / sell recommendations."""
    if not rec:
        return

    labels = ["Strong Buy", "Buy", "Hold", "Sell", "Strong Sell"]
    values = [rec["strongBuy"], rec["buy"], rec["hold"], rec["sell"], rec["strongSell"]]
    colors = ["darkgreen", "lightgreen", "gold", "salmon", "darkred"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=colors)
    ax.set_title(f"{name} — Analyst Recommendations ({rec['period']})")
    ax.set_ylabel("Number of Analysts")
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            str(val),
            ha="center", va="bottom",
        )
    fig.tight_layout()
    plt.show()


# =============================================================================
# CLI Menu Helpers
# =============================================================================

def select_asset_class():
    """
    Prompt the user to choose an asset class then a specific asset within it.

    Returns:
        tuple: (asset_name, ticker_symbol) or (None, None) on invalid input
    """
    classes = list(ASSET_CLASSES.keys())
    print()
    for i, cls in enumerate(classes, 1):
        print(f"  {i}. {cls}")

    cls_input = input("\nSelect asset class (number): ").strip()
    try:
        idx = int(cls_input) - 1
        if not 0 <= idx < len(classes):
            raise ValueError
        asset_class = classes[idx]
    except ValueError:
        print("Invalid selection.")
        return None, None

    assets      = ASSET_CLASSES[asset_class]
    asset_names = list(assets.keys())
    print(f"\n--- {asset_class} ---")
    for i, name in enumerate(asset_names, 1):
        print(f"  {i}. {name}")

    asset_input = input("\nSelect asset (number): ").strip()
    try:
        a_idx = int(asset_input) - 1
        if not 0 <= a_idx < len(asset_names):
            raise ValueError
        name   = asset_names[a_idx]
        ticker = assets[name]
        return name, ticker
    except ValueError:
        print("Invalid selection.")
        return None, None


# =============================================================================
# Analysis Pipeline
# =============================================================================

def run_analysis(selected_name, ticker_symbol, is_stock=False):
    """
    Execute the full analysis pipeline for the given asset.

    Steps:
        1. Current price (and analyst target for stocks)
        2. 6-month price history chart
        3. News aggregation and VADER sentiment analysis
        4. Analyst recommendation chart (stocks only)
    """
    print(f"\nAnalyzing {selected_name}  (ticker: {ticker_symbol})\n")
    print("-" * 50)

    ticker_obj = yf.Ticker(ticker_symbol)

    # --- Price data ---
    current = get_current_price(ticker_obj)
    if current:
        print(f"Current Price : ${current:,.2f}")
    else:
        print("Current Price : unavailable")

    if is_stock:
        target = get_target_price(ticker_obj)
        if target and current:
            upside = (target / current - 1) * 100
            print(f"Analyst Target: ${target:,.2f}  (implied upside: {upside:+.1f}%)")
        elif target:
            print(f"Analyst Target: ${target:,.2f}")
        else:
            print("Analyst Target: unavailable")

    # --- Price history ---
    print("\nFetching 6-month price history...")
    hist = get_price_history(ticker_obj)
    if hist is not None:
        print(f"  Latest close: ${hist.iloc[-1]:,.2f}")
        plot_price_history(hist, selected_name)
    else:
        print("  No price history data returned.")

    # --- News and sentiment ---
    print("\nFetching latest headlines (Yahoo Finance + Google News RSS)...")
    articles = fetch_market_news(selected_name, ticker_symbol)

    if not articles:
        print("  No articles found.")
    else:
        overall, pos, neu, neg = analyze_sentiment_vader(articles)

        if overall >= 0.05:
            sentiment_label = "POSITIVE"
        elif overall <= -0.05:
            sentiment_label = "NEGATIVE"
        else:
            sentiment_label = "NEUTRAL"

        print(f"\nOverall VADER Sentiment Score : {overall:.3f}  ({sentiment_label})")
        print_sentiment_summary(articles, len(articles))

        print("\nRecent headlines:")
        print("-" * 50)
        for i, art in enumerate(articles[:10], 1):
            print(f"{i:>2}. {art['headline']}")
            print(
                f"    Sentiment : {art['sentiment_label']}"
                f"  (score: {art['sentiment_score']:+.2f})"
                f"  |  {art.get('publisher', 'Unknown')}"
            )
            print(f"    Link      : {art['url']}\n")

        plot_sentiment_indicator(overall)
        plot_sentiment_distribution(pos, neu, neg)

    # --- Analyst recommendations (equities only) ---
    if is_stock:
        print("\nFetching analyst recommendations...")
        rec = get_recommendations(ticker_obj)
        if rec:
            print(f"  Period      : {rec['period']}")
            print(
                f"  Strong Buy  : {rec['strongBuy']}  |  "
                f"Buy: {rec['buy']}  |  "
                f"Hold: {rec['hold']}  |  "
                f"Sell: {rec['sell']}  |  "
                f"Strong Sell: {rec['strongSell']}"
            )
            plot_recommendations(rec, selected_name)
        else:
            print("  No recommendation data available.")

    print("\nAnalysis complete.")
    print("=" * 50)


# =============================================================================
# Entry Point
# =============================================================================

def main():
    """
    Interactive command-line loop for the Financial Asset Sentiment Analyzer.

    Presents a menu that allows the user to analyze a predefined asset class
    or enter a custom US equity ticker. The loop continues until the user
    chooses to exit.
    """
    while True:
        print("\n=== Financial Asset Sentiment Analyzer ===")
        print("  1. Asset classes  (Indices, Crypto, Energies, Metals)")
        print("  2. US equity      (enter any ticker symbol)")
        print("  0. Exit")

        choice = input("\nEnter choice: ").strip()

        if choice == "0":
            print("Exiting.")
            break

        elif choice == "1":
            name, ticker = select_asset_class()
            if name is None:
                continue
            run_analysis(name, ticker, is_stock=False)

        elif choice == "2":
            ticker = input("Enter ticker symbol (e.g. AAPL, MSFT, TSLA): ").strip().upper()
            if not ticker:
                print("No ticker entered.")
                continue
            run_analysis(ticker, ticker, is_stock=True)

        else:
            print("Please enter 0, 1, or 2.")
            continue

        again = input("\nAnalyze another asset? (y/n): ").strip().lower()
        if again != "y":
            print("Exiting.")
            break


if __name__ == "__main__":
    main()
