"""
exit_test.py
------------
Pre-registered test of three rules for trades that never revert, on the
2019-2023 research sample. Its result picks what goes into the forward
holdout (forward_test.py).

Written and committed BEFORE any of these rules touched market data. Nothing
below may be changed after the run; any variation is a new trial and must be
reported as one.

Hypothesis (from the exit breakdown of the wide-universe baseline, e=2 x=0.5
lb=60, summed per-trade return on book capital):

                signal exits      period-end exits     stops
  2019-2023     +460% (76% win)   -331% (28% win)      -164% (21% win)
  2024-2026     +233% (78% win)   -164% (28% win)       -97% (21% win)

  The signal makes money. The losses come from trades whose spread has not
  reverted when the 63-day window closes, or that hit the stop. Cutting or
  rescuing those trades should raise the Sharpe.

This is NOT an out-of-sample test. The hypothesis came from looking at this
same sample (and at 2024-2026, which is why that sample cannot confirm it
either). A pass here only earns a place in the forward holdout. The rules
have no search: each multiple is set to 2, the HOLD_MULTIPLE the project
already uses for its triple-barrier time limit, and no other value is tried.

Rules (backtest/portfolio.py ExitRules; half-life = the pair's half-life on
its selection window):
  LATE   no new entry with fewer than 2 x half-life trading days left in the
         period. A pair with a half-life above ~31 days never trades.
  TIME   close a trade after 2 x half-life trading days held. It blocks
         re-entry until |z| is back under the entry level, as the stop does.
  CARRY  a position open at period end stays open if the same pair is
         selected for the next period, under that period's Kalman fit and
         z-score. The selection uses the period_end close, so it is known
         only after that close: a position in a pair that is not re-selected
         closes one day later, at the first close of the next period.
  ALL    the three together.

Frozen setup: main.py's wide default (universe, raw p<0.10 screen, half-life
5-60, 252d train / 63d test, 0.8%/day book budget, 2bp per leg, stop 3.5,
rolling z-score), e=2 x=0.5 lb=60, 2019-2023, no learning layer.

Candidates: LATE, TIME, CARRY, ALL, so 4 new trials. main.py's EXTRA_TRIALS
already includes them, whatever the result. Every figure also shows the
baseline, which is not a new trial.

Decision, on the candidate with the highest full-period Sharpe:
  ADVANCE       Sharpe > 0 AND t >= 2.5 on the daily PnL difference candidate
                minus baseline (two-sided 5%, Bonferroni over 4 candidates)
  DO NOT ADVANCE  anything else
The DSR over all trials is reported, with no verdict attached: at N = 82 and
SR0 ~ 1.2 it can hardly pass, as the tuned result's 0.24 showed.

The decision is mechanical: this script writes forward_config.json itself.
On ADVANCE it names the candidate; otherwise the candidate is null, and the
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
from signals.walkforward import generate_pair_schedule
from backtest.portfolio import ExitRules, prepare_pair_periods, run_portfolio_backtest
from backtest.report import summary_stats
from backtest.export import variant_tables, write_tables
from learn.evaluate import daily_sharpe, deflated_sharpe

MULT = 2.0   # the project's HOLD_MULTIPLE; deliberately not searched
MIN_T = 2.5  # two-sided 5%, Bonferroni over the 4 candidates
FORWARD_CONFIG = "forward_config.json"

CONFIGS = {
    "baseline" : ExitRules(),
    "LATE"     : ExitRules(late_entry_mult=MULT),
    "TIME"     : ExitRules(time_stop_mult=MULT),
    "CARRY"    : ExitRules(carry=True),
    "ALL"      : ExitRules(late_entry_mult=MULT, time_stop_mult=MULT, carry=True),
}
CANDIDATES = ["LATE", "TIME", "CARRY", "ALL"]


def prior_daily_sharpes() -> list[float]:
    """Per-day Sharpes of main.py's 19 configurations, from its last run."""
    tables = [pd.read_csv(os.path.join(M.DASHBOARD_DIR, f"insample_{name}.csv")) for name in ("variants", "grid")]
    return list(pd.concat(tables)["sharpe"] / np.sqrt(252))


def main():
    print(f"Fetching {len(M.UNIVERSE)} tickers {M.START_DATE} -> {M.END_DATE}...")
    prices = fetch_prices(M.UNIVERSE, M.START_DATE, M.END_DATE, max_missing=M.MAX_MISSING, cache_dir=M.CACHE_DIR)
    schedule = generate_pair_schedule(
        prices, prices.columns.tolist(), M.TRAIN_WINDOW, M.TEST_WINDOW,
        M.PVALUE_CUTOFF, M.MULTIPLE_TESTING, M.MIN_HALF_LIFE, M.MAX_HALF_LIFE,
        verbose=False, sectors=M.PAIR_SECTORS, n_jobs=M.N_JOBS, cache_dir=M.CACHE_DIR,
    )
    n_active = {}
    for pp in schedule:
        n_active[pp.period_start] = n_active.get(pp.period_start, 0) + 1
    print(f"  {len(schedule)} pair-periods, {len(n_active)} quarters")

    target_vol = {T: M.BOOK_DAILY_VOL / np.sqrt(k) for T, k in n_active.items()}
    prepared = prepare_pair_periods(prices, schedule, target_vol, M.TRAIN_WINDOW,
                                    n_jobs=M.N_JOBS, cache_dir=M.CACHE_DIR)

    results = {}
    for name, rules in CONFIGS.items():
        results[name] = run_portfolio_backtest(
            prices, schedule, lookback_window=M.LOOKBACK_WINDOW, entry_threshold=M.ENTRY_THRESHOLD,
            exit_threshold=M.EXIT_THRESHOLD, stop_threshold=M.STOP_THRESHOLD, signal=M.SIGNAL,
            prepared=prepared, exit_rules=rules,
        )

    trial_sharpes = prior_daily_sharpes() + [daily_sharpe(results[c].daily_pnl) for c in CANDIDATES]
    n_trials = 19 + M.EXTRA_TRIALS

    print()
    print(f"  {'Config':<10} {'Sharpe':>7} {'DSR':>6} {'Return':>9} {'Vol':>7} {'MaxDD':>9} {'Trades':>7} {'Win%':>6}   t vs base")
    rows, dsr = {}, {}
    base = results["baseline"].daily_pnl
    for name, res in results.items():
        st = summary_stats(res)
        d = deflated_sharpe(res.daily_pnl, trial_sharpes, n_trials)
        diff = res.daily_pnl - base.reindex(res.daily_pnl.index).fillna(0.0)
        t = daily_sharpe(diff) * np.sqrt(len(diff))
        rows[name], dsr[name] = {**st, "t": t}, d["dsr"]
        print(f"  {name:<10} {st['sharpe']:>7.2f} {d['dsr']:>6.2f} {st['total_return']:>9.2%} {st['ann_vol']:>7.2%} "
              f"{st['max_dd']:>9.2%} {st['trades']:>7d} {st['win_rate'] * 100:>5.1f}%   {t:>6.2f}")
    print(f"  DSR over N={n_trials} trials, SR0 = {d['sr0_ann']:.2f} (annualised)")

    print("\n  Sharpe by year:")
    for name, res in results.items():
        x = res.daily_pnl
        yrs = {int(y): round(float(daily_sharpe(x[x.index.year == y]) * np.sqrt(252)), 2) for y in sorted(set(x.index.year))}
        print(f"    {name:<10} {yrs}")

    tables = variant_tables("Exit rules 2019-2023", results, dsr, sectors=M.PAIR_SECTORS)
    print("\n  PnL by exit reason:")
    print(tables["exits"].drop(columns="sample").to_string(index=False))
    write_tables(tables, M.DASHBOARD_DIR, "exitrules")

    best = max(CANDIDATES, key=lambda c: rows[c]["sharpe"])
    advance = rows[best]["sharpe"] > 0 and rows[best]["t"] >= MIN_T
    print(f"\n  Best candidate: {best} (Sharpe {rows[best]['sharpe']:.2f}, t {rows[best]['t']:.2f} vs baseline)")
    print(f"  DECISION (pre-registered rule): {'ADVANCE' if advance else 'DO NOT ADVANCE'}")

    rules = CONFIGS[best]
    with open(FORWARD_CONFIG, "w", encoding="utf-8") as f:
        json.dump({
            "candidate": best if advance else None,
            "rules": {"late_entry_mult": rules.late_entry_mult, "time_stop_mult": rules.time_stop_mult,
                      "carry": rules.carry} if advance else None,
            "best_tested": best,
            "sharpe_2019_2023": round(rows[best]["sharpe"], 4),
            "t_vs_baseline": round(rows[best]["t"], 4),
        }, f, indent=2)
        f.write("\n")
    print(f"  Wrote {FORWARD_CONFIG}")


if __name__ == "__main__":
    main()
