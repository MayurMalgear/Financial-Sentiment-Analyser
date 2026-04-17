# Financial Asset Sentiment Analyzer

A command-line tool that combines real-time market data with news sentiment analysis to give a quick read on how the market is feeling about a given asset.

It pulls price data from Yahoo Finance, aggregates recent headlines from Yahoo Finance and Google News RSS, scores each headline using VADER sentiment analysis, and produces a set of charts covering price history, sentiment distribution, and analyst recommendations.

---

## Supported Assets

| Category | Examples |
|---|---|
| Indices | S&P 500 E-Mini, Nasdaq 100 E-Mini, Dow Futures E-Mini |
| Cryptocurrencies | Bitcoin, Ether, XRP, Solana |
| Energies | Crude Oil WTI/Brent, Natural Gas, Gasoline RBOB |
| Metals | Gold, Silver, Platinum, Palladium, Copper |
| US Equities | Any valid ticker (AAPL, TSLA, MSFT, etc.) |

---

## Requirements

- Python 3.8 or higher

Install dependencies with:

```bash
pip install yfinance matplotlib vaderSentiment feedparser
```

---

## Usage

```bash
python sentiment_analyzer.py
```

You will be presented with a menu:

```
=== Financial Asset Sentiment Analyzer ===
  1. Asset classes  (Indices, Crypto, Energies, Metals)
  2. US equity      (enter any ticker symbol)
  0. Exit
```

Select an option and follow the prompts. The tool will:

1. Display the current price and, for equities, the analyst consensus target
2. Plot the last 6 months of daily closing prices
3. Fetch and score recent headlines using VADER
4. Print a sentiment summary and the top 10 headlines with individual scores
5. Show a sentiment distribution chart and overall score indicator
6. For equities, display a bar chart of analyst buy / hold / sell ratings

---

## Project Structure

```
sentiment_analyzer.py   # Main script — all logic lives here
README.md
```

---

## How Sentiment Scoring Works

Headlines are scored using [VADER](https://github.com/cjhutto/vaderSentiment) (Valence Aware Dictionary and sEntiment Reasoner), a lexicon-based sentiment analysis tool designed for short social and financial text. Each headline receives a compound score between -1 and +1:

| Score | Label |
|---|---|
| >= 0.05 | Positive |
| <= -0.05 | Negative |
| Between | Neutral |

The overall score is the mean of all headline scores for that asset.

---

## Data Sources

- **Price data** — [Yahoo Finance](https://finance.yahoo.com) via the `yfinance` library
- **News** — Yahoo Finance news feed + Google News RSS (deduplicated by headline)
- **Analyst recommendations** — Yahoo Finance (equities only)

---

## Limitations

- Sentiment is based on headline text only, not full article content
- Google News RSS results can vary by region and query phrasing
- Analyst recommendation data is only available for US-listed equities
- Price data availability depends on Yahoo Finance's coverage for each ticker

---

## License

MIT License. Free to use and modify.
