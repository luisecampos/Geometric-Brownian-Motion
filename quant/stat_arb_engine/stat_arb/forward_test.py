"""
forward_test.py
---------------
The forward holdout: the one sample nothing in this project has seen.

2024-2026 was spent on the x=0 test, and its exit breakdown then helped shape
the exit rules, so it can't judge them. Every trading day from HOLDOUT_START
on is new. This file was committed before that date.

Frozen:
  candidate   forward_config.json, written by exit_test.py and never by hand.
              If its candidate is null, nothing is under test and the holdout
              stays unspent.
  universe    main.py's UNIVERSE as of this commit. It is fixed before the
              holdout starts, so unlike every earlier sample this one has no
              survivorship bias. A name that later delists keeps its last
              price, and the run reports it.
  setup       main.py's wide default, e=2 x=0.5 lb=60, no learning layer.
              Pair selection walks forward from the 252 trading days before
              HOLDOUT_START.
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
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

HOLDOUT_START = "2026-11-02"   # first trading day after this file was committed
HOLDOUT_DAYS  = 252
DATA_START    = "2025-06-01"   # more than 252 trading days before HOLDOUT_START
MAX_MISSING   = 0.5            # keep names that delist mid-holdout (ffilled)
FORWARD_CONFIG = "forward_config.json"


def frame_for_holdout(prices: pd.DataFrame, train_window: int) -> pd.DataFrame:
    """Trim history so the first walk-forward period starts on the holdout's first day."""
    first = int(prices.index.searchsorted(pd.Timestamp(HOLDOUT_START)))
    if first < train_window:
        raise ValueError(f"need {train_window} days before {HOLDOUT_START}, have {first}")
    return prices.iloc[first - train_window:]


def verdict(sharpe: float, base_sharpe: float, dsr: float) -> str:
    if sharpe > 0 and dsr >= 0.95:
        return "CONFIRMED"
    if sharpe > 0 and sharpe > base_sharpe:
        return "SUPPORTED, NOT SIGNIFICANT"
    return "NOT CONFIRMED"


def evaluate(prices: pd.DataFrame, cfg: dict):
    import main as M
    from signals.walkforward import generate_pair_schedule
    from backtest.portfolio import ExitRules, prepare_pair_periods, run_portfolio_backtest
    from backtest.report import summary_stats
    from learn.evaluate import deflated_sharpe

    schedule = generate_pair_schedule(
        prices, prices.columns.tolist(), M.TRAIN_WINDOW, M.TEST_WINDOW,
        M.PVALUE_CUTOFF, M.MULTIPLE_TESTING, M.MIN_HALF_LIFE, M.MAX_HALF_LIFE,
        verbose=False, sectors=M.PAIR_SECTORS, n_jobs=M.N_JOBS, cache_dir=M.CACHE_DIR,
    )
    first = min(pp.period_start for pp in schedule)
    assert first >= pd.Timestamp(HOLDOUT_START), f"a period starts before the holdout: {first.date()}"
    n_active = {}
    for pp in schedule:
        n_active[pp.period_start] = n_active.get(pp.period_start, 0) + 1
    target_vol = {T: M.BOOK_DAILY_VOL / np.sqrt(k) for T, k in n_active.items()}
    prepared = prepare_pair_periods(prices, schedule, target_vol, M.TRAIN_WINDOW,
                                    n_jobs=M.N_JOBS, cache_dir=M.CACHE_DIR)

    res = {}
    for name, rules in {cfg["candidate"]: ExitRules(**cfg["rules"]), "baseline": ExitRules()}.items():
        res[name] = run_portfolio_backtest(
            prices, schedule, lookback_window=M.LOOKBACK_WINDOW, entry_threshold=M.ENTRY_THRESHOLD,
            exit_threshold=M.EXIT_THRESHOLD, stop_threshold=M.STOP_THRESHOLD, signal=M.SIGNAL,
            prepared=prepared, exit_rules=rules,
        )
    cand, base = summary_stats(res[cfg["candidate"]]), summary_stats(res["baseline"])
    dsr = deflated_sharpe(res[cfg["candidate"]].daily_pnl, [], 1)["dsr"]
    print(f"  {cfg['candidate']}: Sharpe {cand['sharpe']:.2f}, DSR {dsr:.2f}; baseline Sharpe {base['sharpe']:.2f}")
    print(f"  VERDICT (pre-registered rule): {verdict(cand['sharpe'], base['sharpe'], dsr)}")


def main():
    with open(FORWARD_CONFIG, encoding="utf-8") as f:
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
    frame = frame_for_holdout(prices, M.TRAIN_WINDOW)
    elapsed = int((frame.index >= pd.Timestamp(HOLDOUT_START)).sum())
    print(f"Candidate {cfg['candidate']}. Holdout: {elapsed}/{HOLDOUT_DAYS} days.")
    if stale:
        print(f"  No recent price (delisted or halted, held flat): {stale}")
    if elapsed < HOLDOUT_DAYS:
        return
    evaluate(frame.iloc[:M.TRAIN_WINDOW + HOLDOUT_DAYS], cfg)


if __name__ == "__main__":
    main()
