"""
cost_test.py
------------
Pre-registered cost test of DISTANCE20, the book in the forward holdout.

Written and committed BEFORE any PnL ran at these costs. Nothing below may be
changed after the run; any variation must be reported as one.

Question: every result so far assumes 2bp per leg and no short-borrow fee. How
much of DISTANCE20's 2019-2023 Sharpe of 0.70 is left at more honest costs?

Cost levels, per leg on the dollar notional traded, plus an annual fee on short
notional for each day held:
  FREE        0bp   no fee     to measure gross PnL and break-even cost
  CURRENT     2bp   no fee     what every earlier test used
  REALISTIC   5bp   0.50%/yr   spread, slippage and fees on large caps, ordinary
                               general-collateral borrow
  HARSH      10bp   1.00%/yr   double REALISTIC, as a margin for error
These are assumptions, not measurements: no fill or borrow data is used.
REALISTIC and HARSH are single values, not searched.

Same frozen setup as pair_test.py: e=2 x=0.5 lb=60, no exit rules, no learning
layer, each book's own 0.8%/day budget. Period: all of 2019-2023. 2024-2026 is
not touched. Costs never feed back into the signals, so every level trades the
same trades; the script asserts this.

Trials: 0 added to main.py's EXTRA_TRIALS. No new trade or selection is tried,
and no level can be chosen for looking better, because higher costs can only
lower PnL.

Decision, on DISTANCE20's Sharpe:
  COST-ROBUST     HARSH > 0
  COST-SENSITIVE  REALISTIC > 0 and HARSH <= 0
  COST-DEPENDENT  REALISTIC <= 0

The decision does not change the holdout. forward_test.py stays as committed,
and its verdict stays at 2bp. It does say what that verdict would be worth:
below COST-ROBUST, a 2bp pass would not be evidence of a tradable edge.

Reported with no verdict attached: the same levels on SCREEN (the baseline,
about 13,900 trades), break-even cost per leg with and without REALISTIC borrow,
notional traded a year, average short notional, and Sharpe by year.

Limits: every fill is at the close, costs don't grow with position size, and
borrow is one flat rate with no recalls or hard-to-borrow names. The 2019-2023
Sharpe is in-sample and was the best of 3 candidates, so a net Sharpe here is
still an optimistic number.
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

import main as M
from data.fetch import fetch_prices
from backtest.portfolio import prepare_pair_periods, run_portfolio_backtest
from backtest.report import summary_stats
from backtest.export import variant_tables, write_tables
from learn.evaluate import daily_sharpe, deflated_sharpe
from pair_test import build_schedule

LEVELS = {                     # cost per leg, annual borrow fee
    "FREE": (0.0, 0.0),
    "CURRENT": (2e-4, 0.0),
    "REALISTIC": (5e-4, 0.005),
    "HARSH": (10e-4, 0.01),
}
BOOKS = {"SCREEN": "baseline", "DISTANCE20": "DISTANCE20"}


def prepared_book(prices: pd.DataFrame, selection: str):
    schedule = build_schedule(selection, prices)
    n_active = {}
    for pp in schedule:
        n_active[pp.period_start] = n_active.get(pp.period_start, 0) + 1
    target_vol = {T: M.BOOK_DAILY_VOL / np.sqrt(k) for T, k in n_active.items()}
    return schedule, prepare_pair_periods(prices, schedule, target_vol, M.TRAIN_WINDOW,
                                          n_jobs=M.N_JOBS, cache_dir=M.CACHE_DIR)


def run(prices, schedule, prepared, cost_bps: float, borrow_fee: float):
    return run_portfolio_backtest(
        prices, schedule, lookback_window=M.LOOKBACK_WINDOW, entry_threshold=M.ENTRY_THRESHOLD,
        exit_threshold=M.EXIT_THRESHOLD, stop_threshold=M.STOP_THRESHOLD, signal=M.SIGNAL,
        prepared=prepared, cost_bps=cost_bps, borrow_fee=borrow_fee,
    )


def trade_dates(res) -> list:
    return [(t.entry_date, t.exit_date) for r in res.per_pair.values() for t in r.trades]


def economics(free, current, realistic) -> dict:
    """Costs are linear in the rate, so three runs give all of these."""
    years = len(free.daily_pnl) / 252
    gross = free.daily_pnl.sum()
    traded = (gross - current.daily_pnl.sum()) / LEVELS["CURRENT"][0]
    cost, fee = LEVELS["REALISTIC"]
    borrow = gross - cost * traded - realistic.daily_pnl.sum()
    return {
        "gross_per_year": gross / years,
        "traded_per_year": traded / years,
        "avg_short": borrow / fee / years,
        "breakeven_bps": gross / traded * 1e4,
        "breakeven_with_borrow_bps": (gross - borrow) / traded * 1e4,
    }


def prior_daily_sharpes() -> list[float]:
    """Per-day Sharpes of the earlier in-sample configurations, from their last runs."""
    def read(name):
        return pd.read_csv(os.path.join(M.DASHBOARD_DIR, f"{name}.csv"))
    later = pd.concat([read("exitrules_variants"), read("pairs_variants"), read("news_variants")])
    later = later[later["variant"].str.contains(r"\+") | ~later["variant"].isin(["baseline", "SCREEN", "DISTANCE20"])]
    return list(pd.concat([read("insample_variants"), read("insample_grid"), later])["sharpe"] / np.sqrt(252))


def main():
    print(f"Fetching {len(M.UNIVERSE)} tickers {M.START_DATE} -> {M.END_DATE}...")
    prices = fetch_prices(M.UNIVERSE, M.START_DATE, M.END_DATE, max_missing=M.MAX_MISSING, cache_dir=M.CACHE_DIR)

    results = {}
    for book, selection in BOOKS.items():
        schedule, prepared = prepared_book(prices, selection)
        for level, (cost, fee) in LEVELS.items():
            results[f"{book} {level}"] = run(prices, schedule, prepared, cost, fee)
        same = {tuple(trade_dates(results[f"{book} {level}"])) for level in LEVELS}
        assert len(same) == 1, "costs must not change the trades"

    trial_sharpes = prior_daily_sharpes()
    n_trials = 19 + M.EXTRA_TRIALS

    print()
    print(f"  {'Config':<21} {'Sharpe':>7} {'DSR':>6} {'Return':>9} {'Vol':>7} {'MaxDD':>9} {'Trades':>7} {'Win%':>6}")
    rows, dsr = {}, {}
    for name, res in results.items():
        st = summary_stats(res)
        d = deflated_sharpe(res.daily_pnl, trial_sharpes, n_trials)
        rows[name], dsr[name] = st, d["dsr"]
        print(f"  {name:<21} {st['sharpe']:>7.2f} {d['dsr']:>6.2f} {st['total_return']:>9.2%} {st['ann_vol']:>7.2%} "
              f"{st['max_dd']:>9.2%} {st['trades']:>7d} {st['win_rate'] * 100:>5.1f}%")
    print(f"  DSR over N={n_trials} trials, SR0 = {d['sr0_ann']:.2f} (annualised)")

    print("\n  Sharpe by year:")
    for name, res in results.items():
        x = res.daily_pnl
        yrs = {int(y): round(float(daily_sharpe(x[x.index.year == y]) * np.sqrt(252)), 2) for y in sorted(set(x.index.year))}
        print(f"    {name:<21} {yrs}")

    print("\n  Economics:")
    for book in BOOKS:
        e = economics(*(results[f"{book} {level}"] for level in ["FREE", "CURRENT", "REALISTIC"]))
        print(f"    {book:<11} gross {e['gross_per_year']:>+7.2%}/yr, traded {e['traded_per_year']:>6.1f}x capital/yr, "
              f"avg short {e['avg_short']:.2f}x, break-even {e['breakeven_bps']:.1f}bp per leg "
              f"({e['breakeven_with_borrow_bps']:.1f}bp with REALISTIC borrow)")

    tables = variant_tables("Cost test 2019-2023", results, dsr, sectors=M.PAIR_SECTORS)
    write_tables(tables, M.DASHBOARD_DIR, "costs")

    realistic, harsh = rows["DISTANCE20 REALISTIC"]["sharpe"], rows["DISTANCE20 HARSH"]["sharpe"]
    decision = "COST-ROBUST" if harsh > 0 else "COST-SENSITIVE" if realistic > 0 else "COST-DEPENDENT"
    print(f"\n  DISTANCE20 Sharpe: CURRENT {rows['DISTANCE20 CURRENT']['sharpe']:.2f}, "
          f"REALISTIC {realistic:.2f}, HARSH {harsh:.2f}")
    print(f"  DECISION (pre-registered rule): {decision}")


if __name__ == "__main__":
    main()
