"""
Stat-Arb Pairs Trading Engine
==============================
A realistic pairs trading system using cointegration + Kalman filter
hedge ratio estimation — the same approach used in quant research desks.

Pipeline:
  1. Fetch price data
  2. Find cointegrated pairs (Engle-Granger)
  3. Estimate dynamic hedge ratio (Kalman filter)
  4. Generate z-score signals
  5. Backtest and report performance
"""

from data.fetch import fetch_prices
from signals.cointegration import find_cointegrated_pairs
from signals.kalman import KalmanHedgeRatio
from signals.zscore import compute_zscore
from backtest.engine import run_backtest
from backtest.report import print_report


# ── Config ────────────────────────────────────────────────────────────────────

START_DATE       = "2020-01-01"
END_DATE         = "2024-01-01"
LOOKBACK_WINDOW  = 60
ENTRY_THRESHOLD  = 2.0
EXIT_THRESHOLD   = 0.5
PVALUE_CUTOFF    = 0.05


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  STAT-ARB PAIRS TRADING ENGINE")
    print("=" * 60)

    print("\nEnter two stock tickers separated by a space (e.g. KO PEP):")
    raw = input("  Tickers: ").upper().split()
    if len(raw) < 2:
        print("  Need at least 2 tickers.")
        return
    TICKERS = raw

    # 1. Fetch price data
    print(f"\n[1/5] Fetching prices for {len(TICKERS)} tickers ({START_DATE} → {END_DATE})...")
    prices = fetch_prices(TICKERS, START_DATE, END_DATE)
    print(f"      {len(prices)} trading days loaded.\n")

    # 2. Find cointegrated pairs
    print("[2/5] Testing for cointegration (Engle-Granger)...")
    pairs = find_cointegrated_pairs(prices, pvalue_cutoff=PVALUE_CUTOFF)
    if not pairs:
        print("      No cointegrated pairs found. Try expanding the ticker list.")
        return
    print(f"      Found {len(pairs)} cointegrated pair(s):\n")
    for (t1, t2, pval) in pairs:
        print(f"      {t1:6s} / {t2:6s}  —  p-value: {pval:.4f}")

    # 3. Run strategy on best pair
    best_pair = pairs[0]
    ticker_a, ticker_b, pval = best_pair
    print(f"\n[3/5] Running Kalman filter hedge ratio on {ticker_a}/{ticker_b}...")
    series_a = prices[ticker_a]
    series_b = prices[ticker_b]

    kalman = KalmanHedgeRatio()
    hedge_ratios, spreads = kalman.fit(series_a, series_b)
    print(f"      Hedge ratio range: {hedge_ratios.min():.3f} → {hedge_ratios.max():.3f}")

    # 4. Compute z-score signals
    print(f"\n[4/5] Computing z-scores (lookback={LOOKBACK_WINDOW}d)...")
    zscores = compute_zscore(spreads, window=LOOKBACK_WINDOW)
    print(f"      Z-score range: {zscores.min():.2f} → {zscores.max():.2f}")

    # 5. Backtest
    print(f"\n[5/5] Backtesting strategy...")
    print(f"      Entry threshold: ±{ENTRY_THRESHOLD}σ  |  Exit threshold: ±{EXIT_THRESHOLD}σ\n")
    results = run_backtest(
        series_a, series_b, hedge_ratios, zscores,
        entry_threshold=ENTRY_THRESHOLD,
        exit_threshold=EXIT_THRESHOLD,
    )

    print_report(ticker_a, ticker_b, results)


if __name__ == "__main__":
    main()