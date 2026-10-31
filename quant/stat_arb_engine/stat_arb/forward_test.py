"""
forward_test.py
---------------
The forward holdout: the one sample nothing in this project has seen.

2024-2026 was spent on the x=0 test, and its exit breakdown then helped shape
the exit rules, so it can't judge them. Every trading day from HOLDOUT_START
on is new. This file was committed before that date.

Frozen:
  candidate   forward_frozen.json if it exists, otherwise forward_config.json.
              forward_config.json is written by exit_test.py or pair_test.py,
              never by hand. If its candidate is null, nothing is under test
              and the holdout stays unspent.
              forward_frozen.json is for a declared exception: a candidate that
              failed its bar and was frozen anyway. It says so in its own
              "exception" field, and the verdict is printed with it.
  universe    main.py's UNIVERSE as of this commit. It is fixed before the
              holdout starts, so unlike every earlier sample this one has no
              survivorship bias. A name that later delists keeps its last
              price, and the run reports it.
  setup       main.py's wide default, e=2 x=0.5 lb=60, no learning layer.
              Pair selection walks forward from before HOLDOUT_START, with
              enough lead for a persistence filter to have its history. Only
              periods starting on or after HOLDOUT_START count.
  horizon     HOLDOUT_DAYS trading days, about a year. One evaluation.

Blinding: until the horizon is reached this script reports only how many days
have passed. It shows no PnL, and has no switch to do so: looking early and
deciding what to do about it would turn the holdout into one more in-sample
result. Past the horizon it runs once and reports:

  CONFIRMED                   Sharpe > 0 and DSR >= 0.95 (N = 1 trial)
  SUPPORTED, NOT SIGNIFICANT  Sharpe > 0 and above the unchanged baseline
  NOT CONFIRMED               anything else

Over one year, DSR >= 0.95 takes a Sharpe of about 1.7, so the likely good
outcome is SUPPORTED, which would justify extending the horizon, not trading.
Any change to the candidate restarts the clock on days it hasn't seen.
"""

import json
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

HOLDOUT_START = "2026-11-02"   # first trading day after this file was committed
HOLDOUT_DAYS  = 252
LEAD_WINDOWS  = 3              # earlier rebalances, for a persistence filter
DATA_START    = "2024-12-01"   # more than 252 + 3 x 63 trading days before HOLDOUT_START
MAX_MISSING   = 0.5            # keep names that delist mid-holdout (ffilled)
FORWARD_CONFIG = "forward_config.json"
FORWARD_FROZEN = "forward_frozen.json"


def frame_for_holdout(prices: pd.DataFrame, lead: int) -> pd.DataFrame:
    """Trim history so `lead` days precede the holdout's first day, which then
    falls on a walk-forward rebalance if lead = train window + k x test window."""
    first = int(prices.index.searchsorted(pd.Timestamp(HOLDOUT_START)))
    if first < lead:
        raise ValueError(f"need {lead} days before {HOLDOUT_START}, have {first}")
    return prices.iloc[first - lead:]


def verdict(sharpe: float, base_sharpe: float, dsr: float) -> str:
    if sharpe > 0 and dsr >= 0.95:
        return "CONFIRMED"
    if sharpe > 0 and sharpe > base_sharpe:
        return "SUPPORTED, NOT SIGNIFICANT"
    return "NOT CONFIRMED"


def evaluate(prices: pd.DataFrame, cfg: dict):
    import main as M
    from backtest.portfolio import ExitRules, prepare_pair_periods, run_portfolio_backtest
    from backtest.report import summary_stats
    from learn.evaluate import deflated_sharpe
    from pair_test import build_schedule

    def run(selection, rules):
        schedule = [pp for pp in build_schedule(selection, prices)
                    if pp.period_start >= pd.Timestamp(HOLDOUT_START)]
        n_active = {}
        for pp in schedule:
            n_active[pp.period_start] = n_active.get(pp.period_start, 0) + 1
        target_vol = {T: M.BOOK_DAILY_VOL / np.sqrt(k) for T, k in n_active.items()}
        prepared = prepare_pair_periods(prices, schedule, target_vol, M.TRAIN_WINDOW,
                                        n_jobs=M.N_JOBS, cache_dir=M.CACHE_DIR)
        return run_portfolio_backtest(
            prices, schedule, lookback_window=M.LOOKBACK_WINDOW, entry_threshold=M.ENTRY_THRESHOLD,
            exit_threshold=M.EXIT_THRESHOLD, stop_threshold=M.STOP_THRESHOLD, signal=M.SIGNAL,
            prepared=prepared, exit_rules=rules,
        )

    res = {
        cfg["candidate"]: run(cfg.get("selection") or "baseline", ExitRules(**cfg["rules"])),
        "baseline": run("baseline", ExitRules()),
    }
    cand, base = summary_stats(res[cfg["candidate"]]), summary_stats(res["baseline"])
    dsr = deflated_sharpe(res[cfg["candidate"]].daily_pnl, [], 1)["dsr"]
    print(f"  {cfg['candidate']}: Sharpe {cand['sharpe']:.2f}, DSR {dsr:.2f}; baseline Sharpe {base['sharpe']:.2f}")
    print(f"  VERDICT (pre-registered rule): {verdict(cand['sharpe'], base['sharpe'], dsr)}")
    if cfg.get("exception"):
        print(f"  Declared exception: {cfg['exception']}")


def main():
    path = FORWARD_FROZEN if os.path.exists(FORWARD_FROZEN) else FORWARD_CONFIG
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    if cfg["candidate"] is None:
        print(f"No candidate advanced (best tested: {cfg['best_tested']}). Holdout unspent; nothing to do.")
        return

    today = pd.Timestamp.today().normalize()
    if today <= pd.Timestamp(HOLDOUT_START):
        print(f"Candidate {cfg['candidate']}. Holdout starts {HOLDOUT_START}: 0/{HOLDOUT_DAYS} days.")
        return

    import main as M
    from data.fetch import fetch_prices

    end = (today + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    prices = fetch_prices(M.UNIVERSE, DATA_START, end, max_missing=MAX_MISSING)
    stale = [c for c in prices.columns if prices[c].iloc[-5:].nunique() == 1]
    lead = M.TRAIN_WINDOW + LEAD_WINDOWS * M.TEST_WINDOW
    frame = frame_for_holdout(prices, lead)
    elapsed = int((frame.index >= pd.Timestamp(HOLDOUT_START)).sum())
    print(f"Candidate {cfg['candidate']}. Holdout: {elapsed}/{HOLDOUT_DAYS} days.")
    if stale:
        print(f"  No recent price (delisted or halted, held flat): {stale}")
    if elapsed < HOLDOUT_DAYS:
        return
    evaluate(frame.iloc[:lead + HOLDOUT_DAYS], cfg)


if __name__ == "__main__":
    main()
