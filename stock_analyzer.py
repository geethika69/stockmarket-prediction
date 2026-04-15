"""
Stock Market Investment Analyzer
=================================
Fetches real stock data using yfinance and generates:
  - Price & Moving Average trends
  - RSI (momentum indicator)
  - Volatility comparison
  - Correlation heatmap
  - Investment Score summary

Usage:
    python stock_analyzer.py
    python stock_analyzer.py --tickers AAPL MSFT TSLA --period 1y
"""

import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import yfinance as yf

# ── Default stocks to compare ──────────────────────────────────────────────
DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"]
DEFAULT_PERIOD  = "1y"          # 6mo | 1y | 2y | 5y
RISK_FREE_RATE  = 0.05          # 5 % annual (approx. US T-bill)

# ── Colour palette ─────────────────────────────────────────────────────────
PALETTE = ["#2196F3", "#FF5722", "#4CAF50", "#FF9800", "#9C27B0",
           "#00BCD4", "#E91E63", "#8BC34A"]

plt.rcParams.update({
    "figure.facecolor":  "#0D1117",
    "axes.facecolor":    "#161B22",
    "axes.edgecolor":    "#30363D",
    "axes.labelcolor":   "#C9D1D9",
    "xtick.color":       "#8B949E",
    "ytick.color":       "#8B949E",
    "text.color":        "#C9D1D9",
    "grid.color":        "#21262D",
    "grid.linewidth":    0.6,
    "legend.facecolor":  "#161B22",
    "legend.edgecolor":  "#30363D",
    "font.family":       "monospace",
})


# ═══════════════════════════════════════════════════════════════════════════
# DATA LAYER
# ═══════════════════════════════════════════════════════════════════════════

def fetch_data(tickers: list[str], period: str) -> pd.DataFrame:
    """Download adjusted close prices for all tickers."""
    print(f"\n📥  Downloading data for: {', '.join(tickers)}  ({period})\n")
    raw = yf.download(tickers, period=period, auto_adjust=True, progress=False)
    closes = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    closes = closes.dropna(how="all")
    print(f"✅  {len(closes)} trading days loaded.\n")
    return closes


# ═══════════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════════

def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(window).mean()
    loss  = (-delta.clip(upper=0)).rolling(window).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_metrics(closes: pd.DataFrame) -> pd.DataFrame:
    """Return a summary DataFrame with investment-relevant metrics."""
    daily_ret = closes.pct_change().dropna()
    rows = []
    for ticker in closes.columns:
        s = closes[ticker].dropna()
        dr = daily_ret[ticker].dropna()

        annual_return  = dr.mean() * 252
        annual_vol     = dr.std()  * np.sqrt(252)
        sharpe         = (annual_return - RISK_FREE_RATE) / annual_vol if annual_vol else 0
        max_dd         = ((s / s.cummax()) - 1).min()
        rsi_last       = compute_rsi(s).iloc[-1]

        # Simple momentum: 1-month return
        momentum = (s.iloc[-1] / s.iloc[-21] - 1) if len(s) > 21 else 0

        rows.append({
            "Ticker":          ticker,
            "Annual Return %": round(annual_return * 100, 2),
            "Volatility %":    round(annual_vol    * 100, 2),
            "Sharpe Ratio":    round(sharpe, 2),
            "Max Drawdown %":  round(max_dd * 100, 2),
            "RSI (14)":        round(rsi_last, 1),
            "1M Momentum %":   round(momentum * 100, 2),
        })

    df = pd.DataFrame(rows).set_index("Ticker")

    # ── Investment Score (0–100) ───────────────────────────────────────────
    # Weights: Sharpe 35 | Return 25 | Low-Volatility 20 | Momentum 20
    def norm(col, higher_better=True):
        mn, mx = df[col].min(), df[col].max()
        if mx == mn:
            return pd.Series(50, index=df.index)
        n = (df[col] - mn) / (mx - mn) * 100
        return n if higher_better else 100 - n

    df["Score"] = (
        norm("Sharpe Ratio")    * 0.35 +
        norm("Annual Return %") * 0.25 +
        norm("Volatility %", higher_better=False) * 0.20 +
        norm("1M Momentum %")   * 0.20
    ).round(1)

    return df


# ═══════════════════════════════════════════════════════════════════════════
# VISUALISATION
# ═══════════════════════════════════════════════════════════════════════════

def plot_price_ma(ax, closes, ticker, color):
    """Price line + 20-day & 50-day moving averages."""
    s = closes[ticker].dropna()
    ax.plot(s.index, s.values, color=color, linewidth=1.2, label="Price")
    ax.plot(s.index, s.rolling(20).mean(), color="white",
            linewidth=0.8, linestyle="--", alpha=0.7, label="MA-20")
    ax.plot(s.index, s.rolling(50).mean(), color="#FFD700",
            linewidth=0.8, linestyle="--", alpha=0.7, label="MA-50")
    ax.fill_between(s.index, s.values, s.min(), alpha=0.08, color=color)
    ax.set_title(ticker, fontsize=11, fontweight="bold", color=color)
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.4)
    ax.set_ylabel("Price (USD)", fontsize=8)


def plot_rsi(ax, closes, ticker, color):
    rsi = compute_rsi(closes[ticker].dropna())
    ax.plot(rsi.index, rsi.values, color=color, linewidth=1.0)
    ax.axhline(70, color="#FF5722", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.axhline(30, color="#4CAF50", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.fill_between(rsi.index, rsi.values, 50,
                    where=(rsi > 50), alpha=0.15, color="#4CAF50")
    ax.fill_between(rsi.index, rsi.values, 50,
                    where=(rsi < 50), alpha=0.15, color="#FF5722")
    ax.set_ylim(0, 100)
    ax.set_ylabel("RSI", fontsize=8)
    ax.grid(True, alpha=0.4)
    ax.text(0.01, 0.92, "Overbought (70)", transform=ax.transAxes,
            fontsize=7, color="#FF5722", alpha=0.8)
    ax.text(0.01, 0.04, "Oversold (30)",   transform=ax.transAxes,
            fontsize=7, color="#4CAF50", alpha=0.8)


def build_dashboard(closes: pd.DataFrame, metrics: pd.DataFrame,
                    save_path: str = "dashboard.png"):

    tickers = list(closes.columns)
    n       = len(tickers)
    colors  = PALETTE[:n]

    fig = plt.figure(figsize=(20, 26))
    fig.suptitle("📈  Stock Market Investment Analyzer",
                 fontsize=20, fontweight="bold", color="#58A6FF", y=0.99)

    outer = gridspec.GridSpec(4, 1, figure=fig,
                              hspace=0.45,
                              height_ratios=[n * 2.8, 3, 3, 3.5])

    # ── SECTION 1: Price + RSI per ticker ─────────────────────────────────
    inner1 = gridspec.GridSpecFromSubplotSpec(
        n, 2, subplot_spec=outer[0], hspace=0.6, wspace=0.3)

    for i, (ticker, color) in enumerate(zip(tickers, colors)):
        ax_price = fig.add_subplot(inner1[i, 0])
        ax_rsi   = fig.add_subplot(inner1[i, 1])
        plot_price_ma(ax_price, closes, ticker, color)
        plot_rsi(ax_rsi,   closes, ticker, color)

    # ── SECTION 2: Normalised price comparison ────────────────────────────
    ax_norm = fig.add_subplot(outer[1])
    for ticker, color in zip(tickers, colors):
        s = closes[ticker].dropna()
        ax_norm.plot(s.index, (s / s.iloc[0]) * 100,
                     color=color, linewidth=1.4, label=ticker)
    ax_norm.axhline(100, color="white", linewidth=0.6,
                    linestyle=":", alpha=0.5)
    ax_norm.set_title("Normalised Price Performance (Base = 100)",
                      fontsize=12, fontweight="bold", color="#58A6FF")
    ax_norm.legend(fontsize=9)
    ax_norm.grid(True, alpha=0.4)
    ax_norm.set_ylabel("Indexed Price", fontsize=9)

    # ── SECTION 3: Risk-Return scatter + Volatility bar ───────────────────
    inner3 = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer[2], wspace=0.35)

    # Risk-Return scatter
    ax_rr = fig.add_subplot(inner3[0])
    for ticker, color in zip(tickers, colors):
        x = metrics.loc[ticker, "Volatility %"]
        y = metrics.loc[ticker, "Annual Return %"]
        ax_rr.scatter(x, y, color=color, s=180, zorder=5, edgecolors="white",
                      linewidths=0.5)
        ax_rr.annotate(ticker, (x, y), textcoords="offset points",
                       xytext=(6, 4), fontsize=9, color=color)
    ax_rr.axhline(0, color="white", linewidth=0.5, linestyle="--", alpha=0.5)
    ax_rr.set_xlabel("Annual Volatility %", fontsize=9)
    ax_rr.set_ylabel("Annual Return %",     fontsize=9)
    ax_rr.set_title("Risk vs Return",       fontsize=12,
                    fontweight="bold", color="#58A6FF")
    ax_rr.grid(True, alpha=0.4)

    # Volatility bar chart
    ax_vol = fig.add_subplot(inner3[1])
    vols = metrics["Volatility %"].values
    bars = ax_vol.bar(tickers, vols, color=colors, edgecolor="#30363D",
                      linewidth=0.5)
    ax_vol.bar_label(bars, fmt="%.1f%%", fontsize=8, color="white", padding=3)
    ax_vol.set_title("Annual Volatility Comparison",
                     fontsize=12, fontweight="bold", color="#58A6FF")
    ax_vol.set_ylabel("Volatility %", fontsize=9)
    ax_vol.grid(True, alpha=0.4, axis="y")

    # ── SECTION 4: Correlation heatmap + Investment Score ─────────────────
    inner4 = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer[3], wspace=0.4)

    # Correlation heatmap
    ax_corr = fig.add_subplot(inner4[0])
    corr = closes.pct_change().corr()
    cmap = LinearSegmentedColormap.from_list(
        "rg", ["#FF5722", "#21262D", "#4CAF50"], N=256)
    im = ax_corr.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1,
                        aspect="auto")
    plt.colorbar(im, ax=ax_corr, fraction=0.046, pad=0.04)
    ax_corr.set_xticks(range(n)); ax_corr.set_xticklabels(tickers, fontsize=9)
    ax_corr.set_yticks(range(n)); ax_corr.set_yticklabels(tickers, fontsize=9)
    ax_corr.set_title("Return Correlation Matrix",
                      fontsize=12, fontweight="bold", color="#58A6FF")
    for i in range(n):
        for j in range(n):
            ax_corr.text(j, i, f"{corr.values[i, j]:.2f}",
                         ha="center", va="center", fontsize=8,
                         color="white" if abs(corr.values[i, j]) < 0.7 else "#0D1117")

    # Investment Score bar (horizontal)
    ax_score = fig.add_subplot(inner4[1])
    scores = metrics["Score"].sort_values(ascending=True)
    bar_colors = [PALETTE[tickers.index(t)] for t in scores.index]
    h_bars = ax_score.barh(scores.index, scores.values,
                           color=bar_colors, edgecolor="#30363D",
                           linewidth=0.5, height=0.55)
    ax_score.bar_label(h_bars, fmt="%.1f", fontsize=9,
                       color="white", padding=4)
    ax_score.set_xlim(0, 115)
    ax_score.set_title("Investment Score (0–100)",
                       fontsize=12, fontweight="bold", color="#58A6FF")
    ax_score.set_xlabel("Score", fontsize=9)
    ax_score.grid(True, alpha=0.4, axis="x")

    best = scores.idxmax()
    ax_score.text(0.99, 0.02,
                  f"⭐ Top Pick: {best}",
                  transform=ax_score.transAxes,
                  fontsize=10, color="#FFD700",
                  ha="right", va="bottom", fontweight="bold")

    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"📊  Dashboard saved → {save_path}")
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def print_metrics(metrics: pd.DataFrame):
    sep = "─" * 80
    print(f"\n{sep}")
    print("  INVESTMENT METRICS SUMMARY")
    print(sep)
    print(metrics.to_string())
    print(sep)

    best = metrics["Score"].idxmax()
    print(f"\n⭐  RECOMMENDED STOCK  →  {best}  "
          f"(Score: {metrics.loc[best, 'Score']})")
    print(f"   • Annual Return : {metrics.loc[best, 'Annual Return %']}%")
    print(f"   • Volatility    : {metrics.loc[best, 'Volatility %']}%")
    print(f"   • Sharpe Ratio  : {metrics.loc[best, 'Sharpe Ratio']}")
    print(f"   • Max Drawdown  : {metrics.loc[best, 'Max Drawdown %']}%")
    print(f"   • RSI (14)      : {metrics.loc[best, 'RSI (14)']}")
    print(f"\n⚠️   This is for educational purposes only. "
          "Not financial advice.\n")


def main():
    parser = argparse.ArgumentParser(
        description="Stock Market Investment Analyzer")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS,
                        help="List of stock tickers (e.g. AAPL MSFT)")
    parser.add_argument("--period", default=DEFAULT_PERIOD,
                        help="Data period: 6mo | 1y | 2y | 5y")
    parser.add_argument("--output", default="dashboard.png",
                        help="Output image filename")
    args = parser.parse_args()

    closes  = fetch_data(args.tickers, args.period)
    metrics = compute_metrics(closes)
    print_metrics(metrics)
    build_dashboard(closes, metrics, save_path=args.output)
    plt.show()


if __name__ == "__main__":
    main()
