"""
Stat-Arb Pairs Trading Engine
==============================
A realistic pairs trading system using cointegration + Kalman filter hedge
ratio estimation, walk-forward pair selection, and vol-targeted position
sizing — the same building blocks used on real quant research desks.

Pipeline:
  1. Fetch price data for the ticker universe
  2. Walk-forward: re-test cointegration on a rolling window (with a
     multiple-testing correction and a half-life filter), building a
     schedule of which pairs are tradeable in which period (no look-ahead)
  3. For every pair-period: dynamic log-space hedge ratio (Kalman, noise
     fit by MLE on the selection window), z-score signal, vol-targeted
     size, and a day-by-day backtest with stop-loss, returns on capital
  4. Combine every pair-period into one portfolio, track correlation between
     pairs' daily PnL
  5. Report portfolio performance and save a tearsheet PNG
  6. Self-learning layer (learn/): meta-labeler, nested walk-forward tuning
     and Thompson allocation, run next to the baseline on the same
     out-of-sample days, with deflated Sharpe and feature importances
"""

import sys

# Windows consoles often default to a legacy codepage (cp1252) that can't
# encode the box-drawing / sigma characters used in the report output below.
# Force UTF-8 stdout so this runs the same on Windows, macOS, and Linux.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import os

from data.fetch import fetch_prices
from signals.walkforward import generate_pair_schedule
from backtest.portfolio import prepare_pair_periods, run_portfolio_backtest
from backtest.report import print_portfolio_report, print_comparison
from backtest.tearsheet import plot_tearsheet, plot_comparison


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
STOP_THRESHOLD    = 3.5   # |z| beyond this = pair likely broke; exit and wait
# Pair screen. With the self-learning layer on, the z-score rule is the
# high-recall PRIMARY model and the meta-labeler is the precision layer, so
# the screen is deliberately loose: raw p < 0.10, no multiple-testing
# correction, half-life filter kept. BH at 5% leaves 2 pair-periods / 3 trades
# in 2019-2023, nothing to learn from. The coint p-value is a meta-labeler
# feature, so the classifier can still learn to distrust weak pairs.
# For the conservative stand-alone engine use 0.05 + "bh".
PVALUE_CUTOFF     = 0.10  # FDR level under "bh", family-wise level under "bonferroni"
MULTIPLE_TESTING  = None  # "bh" | "bonferroni" | None — 105 pairs/quarter need a correction
MIN_HALF_LIFE     = 5     # days — faster than this is usually noise / bid-ask bounce
MAX_HALF_LIFE     = 60    # days — slower than this won't revert inside a 63d window
SIGNAL            = "rolling"  # "rolling" z-score of the spread | "innovation" = Kalman e/sqrt(S)
TARGET_DAILY_VOL  = 0.0025  # per-pair daily vol target, as a fraction of book capital
                          # (~10 uncorrelated pairs on at once -> ~0.8%/day, ~12%/yr for the book)

SECTORS = {
    "JPM": "banks", "BAC": "banks", "WFC": "banks", "C": "banks",
    "KO": "staples", "PEP": "staples",
    "XOM": "energy", "CVX": "energy", "COP": "energy",
    "WMT": "retail", "TGT": "retail", "COST": "retail",
    "DAL": "airlines", "UAL": "airlines", "AAL": "airlines",
}

# ── Self-learning layer (learn/) ──────────────────────────────────────────────
SELF_LEARNING     = True
MARKET_TICKER     = "SPY"   # regime features only; never traded
EMBARGO_DAYS      = 5       # gap between the last usable label and a rebalance
MIN_TRAIN_EVENTS  = 40      # meta-labeler abstains (baseline size) until this many labels
P_MIN             = 0.5     # skip a signal if P(profitable) is below this...
KELLY_UNIT        = 0.25    # ...or its Kelly fraction <= 0; size = f*/KELLY_UNIT, capped
MAX_SIZE_MULT     = 2.0
HOLD_MULTIPLE     = 2.0     # triple-barrier time limit = HOLD_MULTIPLE x half-life
# Configurations evaluated while building this, beyond the ones main.py runs
# itself: two candidate screens (raw p<0.05 and p<0.10). Counted in the DSR.
EXTRA_TRIALS      = 2
TRADE_LOG_PATH    = os.path.join("outputs", "trade_log.sqlite")   # ".parquet" needs pyarrow


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  STAT-ARB PORTFOLIO ENGINE  (walk-forward, multi-pair)")
    print("=" * 60)

    steps = 5 if SELF_LEARNING else 4

    print(f"\n[1/{steps}] Fetching prices for {len(UNIVERSE)} tickers ({START_DATE} -> {END_DATE})...")
    prices = fetch_prices(UNIVERSE, START_DATE, END_DATE)
    print(f"      {len(prices)} trading days loaded across {len(prices.columns)} tickers.")
    spy = None
    if SELF_LEARNING:
        spy = fetch_prices([MARKET_TICKER], START_DATE, END_DATE)[MARKET_TICKER]
        spy = spy.reindex(prices.index).ffill()

    print(f"\n[2/{steps}] Walk-forward cointegration (train={TRAIN_WINDOW}d, test={TEST_WINDOW}d)...")
    schedule = generate_pair_schedule(
        prices,
        universe=prices.columns.tolist(),
        train_window=TRAIN_WINDOW,
        test_window=TEST_WINDOW,
        pvalue_cutoff=PVALUE_CUTOFF,
        multiple_testing=MULTIPLE_TESTING,
        min_half_life=MIN_HALF_LIFE,
        max_half_life=MAX_HALF_LIFE,
        verbose=True,
    )
    if not schedule:
        print("      No cointegrated pairs found in any window. Try a bigger/different universe.")
        return
    n_distinct = len({(pp.ticker_a, pp.ticker_b) for pp in schedule})
    print(f"      {len(schedule)} pair-period(s) across {n_distinct} distinct pair(s).")

    print(f"\n[3/{steps}] Backtesting every pair-period (vol target={TARGET_DAILY_VOL:.2%}/day)...")
    # Kalman fits are done once and shared by the baseline and every learned variant.
    prepared = prepare_pair_periods(prices, schedule, TARGET_DAILY_VOL, TRAIN_WINDOW)
    results = run_portfolio_backtest(
        prices,
        schedule,
        lookback_window=LOOKBACK_WINDOW,
        entry_threshold=ENTRY_THRESHOLD,
        exit_threshold=EXIT_THRESHOLD,
        stop_threshold=STOP_THRESHOLD,
        target_daily_vol=TARGET_DAILY_VOL,
        signal=SIGNAL,
        train_window=TRAIN_WINDOW,
        prepared=prepared,
    )

    print_portfolio_report(results)

    print(f"[4/{steps}] Rendering tearsheet...")
    out_path = plot_tearsheet(results, out_dir="outputs")
    print(f"      Saved to {out_path}")

    if not SELF_LEARNING:
        return

    # Imported here so the plain engine runs without scikit-learn installed.
    from learn.pipeline import LearnConfig, run_learning_layer
    from learn.trade_log import TradeLog

    print(f"\n[5/{steps}] Self-learning layer: meta-labeling, nested walk-forward tuning, Thompson allocation...")
    cfg = LearnConfig(
        lookback_window=LOOKBACK_WINDOW, entry_threshold=ENTRY_THRESHOLD,
        exit_threshold=EXIT_THRESHOLD, stop_threshold=STOP_THRESHOLD, signal=SIGNAL,
        embargo_days=EMBARGO_DAYS, min_train=MIN_TRAIN_EVENTS, p_min=P_MIN,
        kelly_unit=KELLY_UNIT, max_mult=MAX_SIZE_MULT, hold_multiple=HOLD_MULTIPLE,
        sectors=SECTORS, extra_trials=EXTRA_TRIALS,
    )
    learned = run_learning_layer(prices, schedule, prepared, results, spy=spy, cfg=cfg)

    print()
    print_comparison(learned.rows)
    for kind, imp in learned.importances.items():
        top = ", ".join(f"{f} {v:+.3f}" for f, v in imp.head(5).items())
        print(f"  OOS permutation importance (meta-{kind}, drop in AUC): {top}")
    print()

    log_path = TradeLog(TRADE_LOG_PATH).save(learned.candidates)
    print(f"      Trade log ({len(learned.candidates)} candidate signals) saved to {log_path}")
    cmp_path = plot_comparison(learned.variants, learned.rows, learned.importances,
                               learned.calibration, out_dir="outputs")
    print(f"      Comparison tearsheet saved to {cmp_path}")


if __name__ == "__main__":
    main()
