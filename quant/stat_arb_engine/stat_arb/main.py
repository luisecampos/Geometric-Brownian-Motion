"""
Stat-Arb Pairs Trading Engine
==============================
A realistic pairs trading system using cointegration + Kalman filter hedge
ratio estimation, walk-forward pair selection, and vol-targeted position
sizing — the same building blocks used on real quant research desks.

Pipeline:
  1. Fetch price data for the ticker universe
  2. Walk-forward: re-test cointegration on a rolling window, building a
     schedule of which pairs are tradeable in which period (no look-ahead)
  3. For every pair-period: dynamic hedge ratio (Kalman), z-score signal,
     vol-targeted size, and a day-by-day backtest
  4. Combine every pair-period into one portfolio, track correlation between
     pairs' daily PnL
  5. Report portfolio performance and save a tearsheet PNG
"""

import sys

# Windows consoles often default to a legacy codepage (cp1252) that can't
# encode the box-drawing / sigma characters used in the report output below.
# Force UTF-8 stdout so this runs the same on Windows, macOS, and Linux.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from data.fetch import fetch_prices
from signals.walkforward import generate_pair_schedule
from backtest.portfolio import run_portfolio_backtest
from backtest.report import print_portfolio_report
from backtest.tearsheet import plot_tearsheet


# ── Config ────────────────────────────────────────────────────────────────────

START_DATE        = "2018-01-01"
END_DATE           = "2024-01-01"

# Ticker universe walk-forward pair selection draws candidates from, spread
# across a few sectors so there's a realistic chance of finding cointegrated
# pairs each rebalance window (mostly-uncorrelated tickers just won't qualify).
UNIVERSE = [
    # Big banks
    "JPM", "BAC", "WFC", "C",
    # Consumer staples / beverages
    "KO", "PEP",
    # Energy majors
    "XOM", "CVX", "COP",
    # Retail
    "WMT", "TGT", "COST",
    # Airlines
    "DAL", "UAL", "AAL",
]

TRAIN_WINDOW      = 252   # trading days of history used to re-fit cointegration
TEST_WINDOW       = 63    # trading days each pair-selection stays active (~1 quarter)
LOOKBACK_WINDOW   = 60    # z-score rolling window
ENTRY_THRESHOLD   = 2.0
EXIT_THRESHOLD    = 0.5
PVALUE_CUTOFF     = 0.05
TARGET_DAILY_VOL  = 0.01  # per-pair vol target used for position sizing


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  STAT-ARB PORTFOLIO ENGINE  (walk-forward, multi-pair)")
    print("=" * 60)

    print(f"\n[1/4] Fetching prices for {len(UNIVERSE)} tickers ({START_DATE} -> {END_DATE})...")
    prices = fetch_prices(UNIVERSE, START_DATE, END_DATE)
    print(f"      {len(prices)} trading days loaded across {len(prices.columns)} tickers.")

    print(f"\n[2/4] Walk-forward cointegration (train={TRAIN_WINDOW}d, test={TEST_WINDOW}d)...")
    schedule = generate_pair_schedule(
        prices,
        universe=prices.columns.tolist(),
        train_window=TRAIN_WINDOW,
        test_window=TEST_WINDOW,
        pvalue_cutoff=PVALUE_CUTOFF,
    )
    if not schedule:
        print("      No cointegrated pairs found in any window. Try a bigger/different universe.")
        return
    n_distinct = len({(pp.ticker_a, pp.ticker_b) for pp in schedule})
    print(f"      {len(schedule)} pair-period(s) across {n_distinct} distinct pair(s).")

    print(f"\n[3/4] Backtesting every pair-period (vol target={TARGET_DAILY_VOL:.1%}/day)...")
    results = run_portfolio_backtest(
        prices,
        schedule,
        lookback_window=LOOKBACK_WINDOW,
        entry_threshold=ENTRY_THRESHOLD,
        exit_threshold=EXIT_THRESHOLD,
        target_daily_vol=TARGET_DAILY_VOL,
    )

    print_portfolio_report(results)

    print("[4/4] Rendering tearsheet...")
    out_path = plot_tearsheet(results, out_dir="outputs")
    print(f"      Saved to {out_path}")


if __name__ == "__main__":
    main()
