"""
pair_test.py
------------
Pre-registered test of stricter pair selection on the 2019-2023 research
sample. Its result decides what, if anything, enters the forward holdout
(forward_test.py).

Written and committed BEFORE either method ran on market data. Nothing below
may be changed after the run; any variation is a new trial and must be
reported as one.

Hypothesis (from exit_test.py and the screen's hit counts):
  Changing how a trade closes barely helped: with a time stop, the trades that
  would have been forced out at period end lost about as much anyway. So the
  losses come from entering pairs that don't revert. The raw p<0.10 screen
  finds ~520 pairs a window against ~384 expected by chance, so roughly three
  in four are noise. Selecting fewer, more durable pairs should lose less.

This is NOT an out-of-sample test. The sample has been used for every test in
this project, and the idea came from it. A pass only earns a place in the
forward holdout.

Selection methods (trading is unchanged, see below):
  PERSIST2    the raw p<0.10 + half-life screen, keeping a pair only if it also
              passed in the previous window
  PERSIST4    same, passing in the previous 3 windows too. Windows are 252 days
              stepped by 63, so neighbours share 75% of their data. Passing
              twice is partly mechanical; passing 4 times spans 441 days.
  DISTANCE20  the distance method (Gatev, Goetzmann and Rouwenhorst, 2006):
              rank same-sector pairs by the sum of squared differences of their
              log prices, each rebased to 0 at the start of the 252-day window,
              and take the 20 closest with a half-life of 5-60 days. 20 is the
              paper's number, not searched. No cointegration test.
The baseline is the current screen, unchanged.

Frozen setup: main.py's wide default (universe, same-sector pairs, 252d train
/ 63d test, 0.8%/day book budget split by that config's active pairs, 2bp per
leg, stop 3.5, rolling z-score, log-space Kalman hedge), e=2 x=0.5 lb=60, no
exit rules, no learning layer.

Evaluation window: PERSIST4 can't trade before its 4th window, so EVERY config
including the baseline is measured from that window on (to 2023-12-29). Early
2019, a bad stretch for the baseline, is out for all of them, so the baseline
here won't show its usual -0.18.

Candidates: PERSIST2, PERSIST4, DISTANCE20, so 3 new trials. main.py's
EXTRA_TRIALS already includes them, whatever the result.

Decision, on the candidate with the highest Sharpe:
  ADVANCE         Sharpe > 0 AND t >= 2.4 on the daily PnL difference candidate
                  minus baseline (two-sided 5%, Bonferroni over 3 candidates)
  DO NOT ADVANCE  anything else
The DSR over all trials is reported, with no verdict attached.

The decision is mechanical: this script writes forward_config.json itself. On
ADVANCE it names the candidate; otherwise the candidate is null, and the
holdout stays unspent.
"""

import json
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

import main as M
from data.fetch import fetch_prices
from signals.walkforward import distance_schedule, generate_pair_schedule, persistent_pairs
from backtest.portfolio import prepare_pair_periods, run_portfolio_backtest
from backtest.report import summary_stats
from backtest.export import variant_tables, write_tables
from learn.evaluate import daily_sharpe, deflated_sharpe

MIN_T = 2.4        # two-sided 5%, Bonferroni over the 3 candidates
DISTANCE_TOP_N = 20
FORWARD_CONFIG = "forward_config.json"
CANDIDATES = ["PERSIST2", "PERSIST4", "DISTANCE20"]


def build_schedule(name: str, prices: pd.DataFrame) -> list:
    """The pair schedule of one selection method."""
    if name == "DISTANCE20":
        return distance_schedule(prices, prices.columns.tolist(), M.TRAIN_WINDOW, M.TEST_WINDOW,
                                 DISTANCE_TOP_N, M.MIN_HALF_LIFE, M.MAX_HALF_LIFE, sectors=M.PAIR_SECTORS)
    schedule = generate_pair_schedule(
        prices, prices.columns.tolist(), M.TRAIN_WINDOW, M.TEST_WINDOW,
        M.PVALUE_CUTOFF, M.MULTIPLE_TESTING, M.MIN_HALF_LIFE, M.MAX_HALF_LIFE,
        verbose=False, sectors=M.PAIR_SECTORS, n_jobs=M.N_JOBS, cache_dir=M.CACHE_DIR,
    )
    if name == "PERSIST2":
        return persistent_pairs(schedule, 2)
    if name == "PERSIST4":
        return persistent_pairs(schedule, 4)
    return schedule


def backtest(prices: pd.DataFrame, schedule: list):
    """The frozen trading setup on one schedule, with its own book budget."""
    n_active = {}
    for pp in schedule:
        n_active[pp.period_start] = n_active.get(pp.period_start, 0) + 1
    target_vol = {T: M.BOOK_DAILY_VOL / np.sqrt(k) for T, k in n_active.items()}
    prepared = prepare_pair_periods(prices, schedule, target_vol, M.TRAIN_WINDOW,
                                    n_jobs=M.N_JOBS, cache_dir=M.CACHE_DIR)
    return run_portfolio_backtest(
        prices, schedule, lookback_window=M.LOOKBACK_WINDOW, entry_threshold=M.ENTRY_THRESHOLD,
        exit_threshold=M.EXIT_THRESHOLD, stop_threshold=M.STOP_THRESHOLD, signal=M.SIGNAL,
        prepared=prepared,
    )


def prior_daily_sharpes() -> list[float]:
    """Per-day Sharpes of the earlier in-sample configurations, from their last runs."""
    variants = pd.read_csv(os.path.join(M.DASHBOARD_DIR, "insample_variants.csv"))
    grid = pd.read_csv(os.path.join(M.DASHBOARD_DIR, "insample_grid.csv"))
    exits = pd.read_csv(os.path.join(M.DASHBOARD_DIR, "exitrules_variants.csv"))
    exits = exits[exits["variant"] != "baseline"]
    return list(pd.concat([variants, grid, exits])["sharpe"] / np.sqrt(252))


def main():
    print(f"Fetching {len(M.UNIVERSE)} tickers {M.START_DATE} -> {M.END_DATE}...")
    prices = fetch_prices(M.UNIVERSE, M.START_DATE, M.END_DATE, max_missing=M.MAX_MISSING, cache_dir=M.CACHE_DIR)

    schedules = {name: build_schedule(name, prices) for name in ["baseline"] + CANDIDATES}
    first = min(pp.period_start for pp in schedules["PERSIST4"])
    schedules = {name: [pp for pp in s if pp.period_start >= first] for name, s in schedules.items()}
    print(f"  Evaluated from {first.date()}, {len({pp.period_start for pp in schedules['baseline']})} quarters")
    for name, s in schedules.items():
        per_q = pd.Series([pp.period_start for pp in s]).value_counts()
        print(f"  {name:<11} {len(s):>6} pair-periods, {len({(p.ticker_a, p.ticker_b) for p in s}):>5} distinct pairs, "
              f"{per_q.min()}-{per_q.max()} a quarter")

    results = {name: backtest(prices, s) for name, s in schedules.items()}

    trial_sharpes = prior_daily_sharpes() + [daily_sharpe(results[c].daily_pnl) for c in CANDIDATES]
    n_trials = 19 + M.EXTRA_TRIALS

    print()
    print(f"  {'Config':<11} {'Sharpe':>7} {'DSR':>6} {'Return':>9} {'Vol':>7} {'MaxDD':>9} {'Trades':>7} {'Win%':>6}   t vs base")
    rows, dsr = {}, {}
    base = results["baseline"].daily_pnl
    for name, res in results.items():
        st = summary_stats(res)
        d = deflated_sharpe(res.daily_pnl, trial_sharpes, n_trials)
        diff = res.daily_pnl - base.reindex(res.daily_pnl.index).fillna(0.0)
        t = daily_sharpe(diff) * np.sqrt(len(diff))
        rows[name], dsr[name] = {**st, "t": t}, d["dsr"]
        print(f"  {name:<11} {st['sharpe']:>7.2f} {d['dsr']:>6.2f} {st['total_return']:>9.2%} {st['ann_vol']:>7.2%} "
              f"{st['max_dd']:>9.2%} {st['trades']:>7d} {st['win_rate'] * 100:>5.1f}%   {t:>6.2f}")
    print(f"  DSR over N={n_trials} trials, SR0 = {d['sr0_ann']:.2f} (annualised)")

    print("\n  Sharpe by year:")
    for name, res in results.items():
        x = res.daily_pnl
        yrs = {int(y): round(float(daily_sharpe(x[x.index.year == y]) * np.sqrt(252)), 2) for y in sorted(set(x.index.year))}
        print(f"    {name:<11} {yrs}")

    tables = variant_tables("Pair selection 2019-2023", results, dsr, sectors=M.PAIR_SECTORS)
    print("\n  PnL by exit reason:")
    print(tables["exits"].drop(columns="sample").to_string(index=False))
    write_tables(tables, M.DASHBOARD_DIR, "pairs")

    best = max(CANDIDATES, key=lambda c: rows[c]["sharpe"])
    advance = rows[best]["sharpe"] > 0 and rows[best]["t"] >= MIN_T
    print(f"\n  Best candidate: {best} (Sharpe {rows[best]['sharpe']:.2f}, t {rows[best]['t']:.2f} vs baseline)")
    print(f"  DECISION (pre-registered rule): {'ADVANCE' if advance else 'DO NOT ADVANCE'}")

    with open(FORWARD_CONFIG, "w", encoding="utf-8") as f:
        json.dump({
            "candidate": best if advance else None,
            "selection": best if advance else None,
            "rules": {} if advance else None,
            "best_tested": best,
            "sharpe_2019_2023": round(rows[best]["sharpe"], 4),
            "t_vs_baseline": round(rows[best]["t"], 4),
        }, f, indent=2)
        f.write("\n")
    print(f"  Wrote {FORWARD_CONFIG}")


if __name__ == "__main__":
    main()
