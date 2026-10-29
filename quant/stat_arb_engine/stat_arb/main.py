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

import numpy as np

from data.fetch import fetch_prices
from data import universe as wide
from signals.cache import default_jobs
from signals.walkforward import generate_pair_schedule
from backtest.portfolio import prepare_pair_periods, run_portfolio_backtest
from backtest.report import print_portfolio_report, print_comparison
from backtest.tearsheet import plot_tearsheet, plot_comparison
from backtest.export import variant_tables, learning_tables, write_tables


# ── Config ────────────────────────────────────────────────────────────────────

START_DATE        = "2018-01-01"
END_DATE           = "2024-01-01"

# Ticker universe walk-forward pair selection draws candidates from.
#   "wide"   : 11 SPDR sector ETFs + ~25 large caps per sector (~290 names
#              after data cleaning), see data/universe.py. Pairs are only
#              tested within a sector (~3,800 candidates per window).
#   "narrow" : the original 15 names in 5 groups, every pair tested.
UNIVERSE_NAME = "wide"

NARROW_SECTORS = {
    "JPM": "banks", "BAC": "banks", "WFC": "banks", "C": "banks",
    "KO": "staples", "PEP": "staples",
    "XOM": "energy", "CVX": "energy", "COP": "energy",
    "WMT": "retail", "TGT": "retail", "COST": "retail",
    "DAL": "airlines", "UAL": "airlines", "AAL": "airlines",
}

if UNIVERSE_NAME == "wide":
    UNIVERSE       = wide.UNIVERSE
    PAIR_SECTORS   = wide.SECTORS      # restricts candidate pairs to one sector
    SECTORS        = wide.INDUSTRIES   # same_sector feature = same industry group
    MAX_MISSING    = 0.01              # a late listing would shorten everyone's history
else:
    UNIVERSE       = list(NARROW_SECTORS)
    PAIR_SECTORS   = None
    SECTORS        = NARROW_SECTORS
    MAX_MISSING    = 0.05

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
MULTIPLE_TESTING  = None  # "bh" | "bonferroni" | None — 3,843 pairs/window (wide) need a correction
MIN_HALF_LIFE     = 5     # days — faster than this is usually noise / bid-ask bounce
MAX_HALF_LIFE     = 60    # days — slower than this won't revert inside a 63d window
SIGNAL            = "rolling"  # "rolling" z-score of the spread | "innovation" = Kalman e/sqrt(S)
TARGET_DAILY_VOL  = 0.0025  # per-pair daily vol target, as a fraction of book capital
                          # (~10 uncorrelated pairs on at once -> ~0.8%/day, ~12%/yr for the book)
# Book-level budget. When set, each period's per-pair target is
# BOOK_DAILY_VOL / sqrt(pairs active that period) instead of TARGET_DAILY_VOL,
# so ~500 concurrent pairs don't run ~7x the vol of 10. The pair count is
# known at period_start, so this is causal. It assumes independent pairs;
# on the wide universe realised book vol still comes out ~3x the budget
# (40% vs ~13%/yr) because pairs share legs and sector factors. Sharpe/DSR
# don't depend on the scale; returns, vol and drawdowns do.
BOOK_DAILY_VOL    = 0.008 if UNIVERSE_NAME == "wide" else None

N_JOBS            = default_jobs()          # processes for coint tests + Kalman fits
CACHE_DIR         = "cache"                 # prices, coint tests, Kalman fits (gitignored)

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
# itself (7 variants + 12 grid points = 19). Counted in the DSR. Each line
# is one full run of 7 variants + 12 grid points:
#   narrow universe, raw p<0.05 screen                                   19
#   narrow universe, raw p<0.10 screen                                   19
#   wide universe, BH-at-10% screen (sensitivity check)                  19
#   oos_test.py on 2024-2026: x=0 with e=1.5 lb=40 and e=2 lb=60          2
# Add 19 here for every further full configuration you try.
EXTRA_TRIALS      = 19 + 19 + 19 + 2
TRADE_LOG_PATH    = os.path.join("outputs", "trade_log.sqlite")   # ".parquet" needs pyarrow
DASHBOARD_DIR     = os.path.join("outputs", "dashboard")            # CSV tables for the results dashboard


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  STAT-ARB PORTFOLIO ENGINE  (walk-forward, multi-pair)")
    print("=" * 60)

    steps = 5 if SELF_LEARNING else 4

    print(f"\n[1/{steps}] Fetching prices for {len(UNIVERSE)} tickers ({START_DATE} -> {END_DATE})...")
    prices = fetch_prices(UNIVERSE, START_DATE, END_DATE, max_missing=MAX_MISSING, cache_dir=CACHE_DIR)
    print(f"      {len(prices)} trading days loaded across {len(prices.columns)} tickers.")
    spy = None
    if SELF_LEARNING:
        spy = fetch_prices([MARKET_TICKER], START_DATE, END_DATE, cache_dir=CACHE_DIR)[MARKET_TICKER]
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
        sectors=PAIR_SECTORS,
        n_jobs=N_JOBS,
        cache_dir=CACHE_DIR,
    )
    if not schedule:
        print("      No cointegrated pairs found in any window. Try a bigger/different universe.")
        return
    n_distinct = len({(pp.ticker_a, pp.ticker_b) for pp in schedule})
    print(f"      {len(schedule)} pair-period(s) across {n_distinct} distinct pair(s).")

    if BOOK_DAILY_VOL:
        n_active = {}
        for pp in schedule:
            n_active[pp.period_start] = n_active.get(pp.period_start, 0) + 1
        target_vol = {T: BOOK_DAILY_VOL / np.sqrt(k) for T, k in n_active.items()}
        vol_msg = (f"book budget {BOOK_DAILY_VOL:.2%}/day split over {min(n_active.values())}-"
                   f"{max(n_active.values())} pairs/period")
    else:
        target_vol = TARGET_DAILY_VOL
        vol_msg = f"vol target={TARGET_DAILY_VOL:.2%}/day per pair"
    print(f"\n[3/{steps}] Backtesting every pair-period ({vol_msg})...")
    # Kalman fits are done once and shared by the baseline and every learned variant.
    prepared = prepare_pair_periods(prices, schedule, target_vol, TRAIN_WINDOW,
                                    n_jobs=N_JOBS, cache_dir=CACHE_DIR)
    results = run_portfolio_backtest(
        prices,
        schedule,
        lookback_window=LOOKBACK_WINDOW,
        entry_threshold=ENTRY_THRESHOLD,
        exit_threshold=EXIT_THRESHOLD,
        stop_threshold=STOP_THRESHOLD,
        signal=SIGNAL,
        train_window=TRAIN_WINDOW,
        prepared=prepared,
    )

    print_portfolio_report(results)

    print(f"[4/{steps}] Rendering tearsheet...")
    out_path = plot_tearsheet(results, out_dir="outputs")
    print(f"      Saved to {out_path}")

    if not SELF_LEARNING:
        paths = write_tables(variant_tables("2019-2023", {"baseline": results}, sectors=PAIR_SECTORS),
                             DASHBOARD_DIR, "insample")
        print(f"      Dashboard tables saved to {os.path.dirname(paths[0])}")
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
    dsr = {r["name"]: r["dsr"] for r in learned.rows}
    tables = {**variant_tables("2019-2023", learned.variants, dsr, sectors=PAIR_SECTORS),
              **learning_tables("2019-2023", learned)}
    paths = write_tables(tables, DASHBOARD_DIR, "insample")
    print(f"      Dashboard tables saved to {os.path.dirname(paths[0])}")


if __name__ == "__main__":
    main()
