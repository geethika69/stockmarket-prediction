# 📈 Stock Market Investment Analyzer

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Libraries](https://img.shields.io/badge/Libraries-yfinance%20%7C%20pandas%20%7C%20matplotlib-orange)

A Python-based financial data visualization tool that fetches **real-time stock market data** and generates a comprehensive investment analysis dashboard — helping you decide *which stocks are worth investing in*.

---

## 📊 What It Does

Given a list of stock tickers (e.g. AAPL, MSFT, TSLA), the tool:

1. **Fetches live data** via `yfinance` (no API key required)
2. **Computes financial metrics** for each stock:
   - Annual Return & Volatility
   - Sharpe Ratio (risk-adjusted return)
   - Maximum Drawdown
   - RSI — Relative Strength Index (momentum)
   - 1-Month Price Momentum
3. **Generates a dark-themed dashboard** with 5 visualization panels
4. **Scores each stock** (0–100) and recommends the best pick

---

## 🖼️ Dashboard Preview

| Panel | Description |
|-------|-------------|
| Price + MA | Price trend with 20-day & 50-day moving averages |
| RSI Chart | Momentum indicator (overbought/oversold zones) |
| Normalised Performance | All stocks indexed to 100 for fair comparison |
| Risk vs Return | Scatter plot: volatility vs annual return |
| Correlation Heatmap | How stocks move together (diversification insight) |
| Investment Score | Composite ranking bar chart with top pick |

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/stock-analyzer.git
cd stock-analyzer
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the analyzer

```bash
# Default: AAPL, MSFT, GOOGL, TSLA, AMZN — last 1 year
python stock_analyzer.py

# Custom tickers and period
python stock_analyzer.py --tickers AAPL MSFT INFY TCS.NS --period 2y

# Save dashboard to a custom file
python stock_analyzer.py --output my_analysis.png
```

---

## 📐 Investment Score Formula

The composite **Investment Score (0–100)** is calculated as:

```
Score = (Sharpe Ratio  × 35%)
      + (Annual Return × 25%)
      + (Low Volatility × 20%)
      + (1M Momentum   × 20%)
```

Each metric is **normalised** across the selected stocks before weighting,
so scores are always relative to the current comparison set.

> ⚠️ **Disclaimer**: This tool is for **educational and research purposes only**.
> It is not financial advice. Always do your own due diligence before investing.

---

## 🗂️ Project Structure

```
stock-analyzer/
│
├── stock_analyzer.py   # Main script — fetch, compute, visualise
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## 🧰 Tech Stack

| Library | Purpose |
|---------|---------|
| `yfinance` | Fetch real stock price data from Yahoo Finance |
| `pandas` | Data manipulation and time-series handling |
| `numpy` | Numerical computations (returns, volatility, RSI) |
| `matplotlib` | Multi-panel dashboard visualisation |

---

## 📌 Example Output (Terminal)

```
📥  Downloading data for: AAPL, MSFT, GOOGL, TSLA, AMZN  (1y)
✅  252 trading days loaded.

────────────────────────────────────────────────────────────────────────────────
  INVESTMENT METRICS SUMMARY
────────────────────────────────────────────────────────────────────────────────
       Annual Return %  Volatility %  Sharpe Ratio  Max Drawdown %  RSI (14)  Score
AAPL             22.4          23.1          0.75           -18.2      58.3   71.4
MSFT             18.9          21.8          0.64           -15.6      54.1   65.2
...

⭐  RECOMMENDED STOCK  →  AAPL  (Score: 71.4)
```

---

## 📜 License

MIT © 2024 — Free to use, modify, and distribute.
