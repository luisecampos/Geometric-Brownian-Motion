"""
oos_test.py
-----------
Pre-registered out-of-sample test of "no profit-taking exit" on data the
research never touched (2024-01 to 2026-09).

Written and committed BEFORE any 2024-2026 price was downloaded. Nothing
below may be changed after the run; any variation is a new trial and must be
reported as one.

Hypothesis (from the 2019-2023 wide-universe research, README "Takeaways"):
  Holding every entry until its period ends or the stop hits beats taking
  profit when |z| falls below 0.5. The engine exits on |z| < exit_threshold,
  so exit_threshold = 0 means there is no signal exit at all (it does NOT
  mean "exit at the mean"). In-sample, every x=0 grid point beat every
  x=0.5 point, and nested walk-forward tuning switched to x=0 by 2019-Q3.
  In-sample exit mix for e=2 lb=60: with x=0.5, 8,763 signal exits made
  +4.6 (sum of per-trade PnL, book units) and 3,518 period-end exits lost
  -3.3; with x=0, there are no signal exits and 8,086 period-end exits made
  +1.5.

Frozen setup (identical to main.py's wide default except where listed):
  universe   data/universe.py as committed, same-sector pairs
  screen     raw p < 0.10, half-life 5-60 days, 252d train / 63d test
  sizing     book budget 0.8%/day split as 1/sqrt(active pairs)
  costs      2bp per leg, stop at |z| > 3.5, rolling z-score
  data       2023-01-01 to 2026-09-30 (the first 252 days only select
             pairs; every trading period starts in 2024 or later)
  no learning layer

Configurations (3 run, 2 are candidates, so N = 2 trials on this data):
  PRIMARY    entry 1.5, exit 0, lookback 40: the setting the nested tuner
             chose in every quarter from 2022-Q3 to 2023-Q4, i.e. what it
             would have frozen at the end of the research sample
  SECONDARY  entry 2.0, exit 0, lookback 60: the baseline with only the
             exit changed, which isolates the hypothesis
  REFERENCE  entry 2.0, exit 0.5, lookback 60: the baseline (not a
             candidate; it is what the candidates must beat)

Verdict, decided on PRIMARY with the deflated Sharpe over the 2 candidates:
  CONFIRMED                   Sharpe > 0 and DSR >= 0.95
  SUPPORTED, NOT SIGNIFICANT  Sharpe > 0 and > REFERENCE, DSR < 0.95
  NOT CONFIRMED               anything else
Also reported (secondary evidence, no verdict attached): the annualised
Sharpe and t-stat of the daily PnL difference SECONDARY - REFERENCE, which
tests the exit change alone on the same trades' signals.

In-sample research counts these 2 trials too: main.py's EXTRA_TRIALS goes
up by 2 after this run, whatever the result.
"""

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

import main as M
from data.fetch import fetch_prices
from signals.walkforward import generate_pair_schedule
from backtest.portfolio import prepare_pair_periods, run_portfolio_backtest
from backtest.report import summary_stats
from learn.evaluate import daily_sharpe, deflated_sharpe

START_DATE = "2023-01-01"
END_DATE   = "2026-10-01"     # yfinance end is exclusive: last day 2026-09-30
OOS_START  = pd.Timestamp("2024-01-01")

CONFIGS = {                   # name -> (entry, exit, lookback)
    "PRIMARY e=1.5 x=0 lb=40"    : (1.5, 0.0, 40),
    "SECONDARY e=2 x=0 lb=60"    : (2.0, 0.0, 60),
    "REFERENCE e=2 x=0.5 lb=60"  : (2.0, 0.5, 60),
}
CANDIDATES = ["PRIMARY e=1.5 x=0 lb=40", "SECONDARY e=2 x=0 lb=60"]


def main():
    print(f"Fetching {len(M.UNIVERSE)} tickers {START_DATE} -> {END_DATE}...")
    prices = fetch_prices(M.UNIVERSE, START_DATE, END_DATE, max_missing=M.MAX_MISSING, cache_dir=M.CACHE_DIR)
    print(f"  {len(prices)} days x {len(prices.columns)} tickers, {prices.index[0].date()} -> {prices.index[-1].date()}")

    schedule = generate_pair_schedule(
        prices, prices.columns.tolist(), M.TRAIN_WINDOW, M.TEST_WINDOW,
        M.PVALUE_CUTOFF, M.MULTIPLE_TESTING, M.MIN_HALF_LIFE, M.MAX_HALF_LIFE,
        verbose=True, sectors=M.PAIR_SECTORS, n_jobs=M.N_JOBS, cache_dir=M.CACHE_DIR,
    )
    first = min(pp.period_start for pp in schedule)
    assert first >= OOS_START, f"a trading period starts before 2024: {first.date()}"
    n_active = {}
    for pp in schedule:
        n_active[pp.period_start] = n_active.get(pp.period_start, 0) + 1
    print(f"  {len(schedule)} pair-periods, {len(n_active)} quarters from {first.date()}, "
          f"{min(n_active.values())}-{max(n_active.values())} pairs/quarter")

    target_vol = {T: M.BOOK_DAILY_VOL / np.sqrt(k) for T, k in n_active.items()}
    prepared = prepare_pair_periods(prices, schedule, target_vol, M.TRAIN_WINDOW,
                                    n_jobs=M.N_JOBS, cache_dir=M.CACHE_DIR)

    results = {}
    for name, (entry, exit_, lookback) in CONFIGS.items():
        results[name] = run_portfolio_backtest(
            prices, schedule, lookback_window=lookback, entry_threshold=entry,
            exit_threshold=exit_, stop_threshold=M.STOP_THRESHOLD, signal=M.SIGNAL,
            prepared=prepared,
        )

    trial_sharpes = [daily_sharpe(results[c].daily_pnl) for c in CANDIDATES]
    print()
    print(f"  {'Config':<28} {'Sharpe':>7} {'DSR(N=2)':>9} {'Return':>8} {'Vol':>7} {'MaxDD':>8} {'Trades':>7} {'Win%':>6}")
    rows = {}
    for name, res in results.items():
        st = summary_stats(res)
        d = deflated_sharpe(res.daily_pnl, trial_sharpes, len(CANDIDATES))
        rows[name] = {**st, "dsr": d["dsr"]}
        print(f"  {name:<28} {st['sharpe']:>7.2f} {d['dsr']:>9.2f} {st['total_return']:>8.2%} "
              f"{st['ann_vol']:>7.2%} {st['max_dd']:>8.2%} {st['trades']:>7d} {st['win_rate'] * 100:>5.1f}%")
    print(f"  SR0 over the 2 candidates = {d['sr0_ann']:.2f} (annualised)")

    print("\n  Sharpe by year:")
    for name, res in results.items():
        x = res.daily_pnl
        yrs = {y: round(daily_sharpe(x[x.index.year == y]) * np.sqrt(252), 2) for y in sorted(set(x.index.year))}
        print(f"    {name:<28} {yrs}")

    diff = results["SECONDARY e=2 x=0 lb=60"].daily_pnl - results["REFERENCE e=2 x=0.5 lb=60"].daily_pnl
    sr_diff = daily_sharpe(diff)
    print(f"\n  Exit change alone (SECONDARY - REFERENCE): Sharpe {sr_diff * np.sqrt(252):.2f}, "
          f"t = {sr_diff * np.sqrt(len(diff)):.2f} over {len(diff)} days")

    p, ref = rows["PRIMARY e=1.5 x=0 lb=40"], rows["REFERENCE e=2 x=0.5 lb=60"]
    if p["sharpe"] > 0 and p["dsr"] >= 0.95:
        verdict = "CONFIRMED"
    elif p["sharpe"] > 0 and p["sharpe"] > ref["sharpe"]:
        verdict = "SUPPORTED, NOT SIGNIFICANT"
    else:
        verdict = "NOT CONFIRMED"
    print(f"\n  VERDICT (pre-registered rule on PRIMARY): {verdict}")


if __name__ == "__main__":
    main()
