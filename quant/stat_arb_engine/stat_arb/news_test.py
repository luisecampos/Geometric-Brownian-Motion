"""
news_test.py
------------
Pre-registered test of a firm-news entry filter, built on SEC 8-K filings.

Written and committed BEFORE the filter touched any PnL. The filings were
downloaded and checked for coverage first, with no strategy data involved.
Nothing below may be changed after the run; any variation is a new trial and
must be reported as one.

Hypothesis, from outside this project: Engelberg, Gao and Jagannathan, "An
Anatomy of Pairs Trading", find pairs profits are lower when the price gap
comes from news about one stock, because such a gap tends to persist. Here,
exit_test.py showed the losing trades lose however they close, so skipping
entries around firm news should remove some of them. Unlike the exit rules,
this idea didn't come from this project's own data. It is still in-sample for
2019-2023, which has been used for every test so far.

Filters, applied to entries only (open positions are untouched):
  EARN   skip an entry when either leg filed an earnings 8-K (item 2.02)
         within the last 5 trading days
  NEWS   skip an entry when either leg filed any 8-K within the last 5 trading
         days
A filing counts from the first close that could have reacted to it: its own
day if EDGAR accepted it before the 16:00 ET close of a trading day, otherwise
the next trading day. An entry at close t is decided on close t-1, so a filing
blocks it only if it was effective by t-1. 5 days is one trading week, a single
value, not searched. Amended filings (8-K/A) are excluded. ETFs file no 8-Ks,
so an ETF leg never blocks.

Coverage, checked before this was committed: 39,448 8-Ks across 281 names,
2017-2026, a median of 4.0 earnings 8-Ks a year in 2019-2023. Known gaps, left
as they are: BLK and XOM have no item 2.02 filing in 2019-2023, and APA has
about 2 a year, so EARN misses those names' earnings.

Books: each filter runs on two, for 4 new trials, added to main.py's
EXTRA_TRIALS whatever the result (+1 if stage 2 runs).
  SCREEN      the baseline: raw p<0.10 + half-life screen. About 13,900 trades,
              so this is where the filter can be measured.
  DISTANCE20  the book in the forward holdout, about 500 trades.
Same frozen setup as pair_test.py: e=2 x=0.5 lb=60, no exit rules, no learning
layer, each book's own 0.8%/day budget. Period: all of 2019-2023.

Decision. t is the paired t on daily PnL, filtered book minus the same book
unfiltered. Unlike pair_test.py the two run at about the same vol, so this t
has power. A filter can't be judged on Sharpe > 0, because it only removes
trades from a book that loses money.
  Stage 1, 2019-2023. Take the filter with the higher t on SCREEN. It passes if
    t >= 2.24 (two-sided 5%, Bonferroni over 2 filters) AND it doesn't lower
    DISTANCE20's Sharpe.
  Stage 2, 2024-01 to 2026-09, run only if stage 1 passes, once, on SCREEN with
    oos_test.py's data window. It passes if t >= 1.65 (one-sided 5%). That
    sample has been seen, but never with any news data.
  ADVANCE         both stages pass
  DO NOT ADVANCE  anything else
On ADVANCE, DISTANCE20 + that filter may be frozen as a second forward
candidate. DISTANCE20 alone stays in the holdout either way.

Reported with no verdict attached: the DSR over all trials, the share of
entries each filter removes, and the hypothesis tested directly on the
unfiltered SCREEN book, trades that a filter would have skipped against the
rest (mean return and win rate).
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

import main as M
from data.events import blocked_days, earnings_only, entry_multipliers, fetch_8k_events
from data.fetch import fetch_prices
from backtest.portfolio import prepare_pair_periods, run_portfolio_backtest
from backtest.report import summary_stats
from backtest.export import variant_tables, write_tables
from learn.evaluate import daily_sharpe, deflated_sharpe
from pair_test import build_schedule

EVENTS_START = "2017-01-01"
EVENTS_AS_OF = "2026-10-31"    # the download these results rest on
WINDOW = 5                     # trading days; deliberately not searched
MIN_T1 = 2.24                  # two-sided 5%, Bonferroni over the 2 filters
MIN_T2 = 1.65                  # one-sided 5%
OOS_START, OOS_END = "2023-01-01", "2026-10-01"
FILTERS = ["EARN", "NEWS"]
BOOKS = {"SCREEN": "baseline", "DISTANCE20": "DISTANCE20"}


def prepared_book(prices: pd.DataFrame, selection: str):
    schedule = build_schedule(selection, prices)
    n_active = {}
    for pp in schedule:
        n_active[pp.period_start] = n_active.get(pp.period_start, 0) + 1
    target_vol = {T: M.BOOK_DAILY_VOL / np.sqrt(k) for T, k in n_active.items()}
    return schedule, prepare_pair_periods(prices, schedule, target_vol, M.TRAIN_WINDOW,
                                          n_jobs=M.N_JOBS, cache_dir=M.CACHE_DIR)


def run(prices, schedule, prepared, mults=None):
    return run_portfolio_backtest(
        prices, schedule, lookback_window=M.LOOKBACK_WINDOW, entry_threshold=M.ENTRY_THRESHOLD,
        exit_threshold=M.EXIT_THRESHOLD, stop_threshold=M.STOP_THRESHOLD, signal=M.SIGNAL,
        prepared=prepared, size_multipliers=mults,
    )


def paired_t(a: pd.Series, b: pd.Series) -> float:
    diff = a - b.reindex(a.index).fillna(0.0)
    return daily_sharpe(diff) * np.sqrt(len(diff))


def skipped_vs_kept(res, blocked: dict, calendar: pd.DatetimeIndex) -> dict:
    """Per-trade stats of an unfiltered book, split by whether a filter skips the entry."""
    none = np.zeros(len(calendar), dtype=bool)
    rows = []
    for (a, b, _), r in res.per_pair.items():
        for t in r.trades:
            decision = calendar.get_loc(t.entry_date) - 1
            rows.append((bool(blocked.get(a, none)[decision] or blocked.get(b, none)[decision]), t.pnl))
    df = pd.DataFrame(rows, columns=["skipped", "pnl"])
    out = {}
    for name, g in df.groupby("skipped"):
        out["skipped" if name else "kept"] = (len(g), g["pnl"].sum(), g["pnl"].mean(), (g["pnl"] > 0).mean())
    return out


def prior_daily_sharpes() -> list[float]:
    """Per-day Sharpes of the earlier in-sample configurations, from their last runs."""
    def read(name):
        return pd.read_csv(os.path.join(M.DASHBOARD_DIR, f"{name}.csv"))
    later = pd.concat([read("exitrules_variants"), read("pairs_variants")])
    later = later[later["variant"] != "baseline"]
    return list(pd.concat([read("insample_variants"), read("insample_grid"), later])["sharpe"] / np.sqrt(252))


def main():
    print(f"Fetching {len(M.UNIVERSE)} tickers {M.START_DATE} -> {M.END_DATE}...")
    prices = fetch_prices(M.UNIVERSE, M.START_DATE, M.END_DATE, max_missing=M.MAX_MISSING, cache_dir=M.CACHE_DIR)
    events = fetch_8k_events(M.UNIVERSE, EVENTS_START, EVENTS_AS_OF, M.CACHE_DIR)
    sets = {"EARN": earnings_only(events), "NEWS": events}
    blocked = {f: blocked_days(e, prices.index, WINDOW) for f, e in sets.items()}

    results, books = {}, {}
    for book, selection in BOOKS.items():
        schedule, prepared = prepared_book(prices, selection)
        books[book] = (schedule, prepared)
        results[book] = run(prices, schedule, prepared)
        for f in FILTERS:
            results[f"{book}+{f}"] = run(prices, schedule, prepared, entry_multipliers(prepared, blocked[f], prices.index))

    trial_sharpes = prior_daily_sharpes() + [daily_sharpe(r.daily_pnl) for n, r in results.items() if "+" in n]
    n_trials = 19 + M.EXTRA_TRIALS

    print()
    print(f"  {'Config':<17} {'Sharpe':>7} {'DSR':>6} {'Return':>9} {'Vol':>7} {'MaxDD':>9} {'Trades':>7} {'Win%':>6}   t vs unfiltered")
    rows, dsr = {}, {}
    for name, res in results.items():
        st = summary_stats(res)
        d = deflated_sharpe(res.daily_pnl, trial_sharpes, n_trials)
        t = paired_t(res.daily_pnl, results[name.split("+")[0]].daily_pnl) if "+" in name else 0.0
        rows[name], dsr[name] = {**st, "t": t}, d["dsr"]
        print(f"  {name:<17} {st['sharpe']:>7.2f} {d['dsr']:>6.2f} {st['total_return']:>9.2%} {st['ann_vol']:>7.2%} "
              f"{st['max_dd']:>9.2%} {st['trades']:>7d} {st['win_rate'] * 100:>5.1f}%   {t:>6.2f}")
    print(f"  DSR over N={n_trials} trials, SR0 = {d['sr0_ann']:.2f} (annualised)")

    print("\n  Sharpe by year:")
    for name, res in results.items():
        x = res.daily_pnl
        yrs = {int(y): round(float(daily_sharpe(x[x.index.year == y]) * np.sqrt(252)), 2) for y in sorted(set(x.index.year))}
        print(f"    {name:<17} {yrs}")

    print("\n  Unfiltered SCREEN trades, by whether the filter would skip them:")
    for f in FILTERS:
        for side, (n, total, mean, win) in skipped_vs_kept(results["SCREEN"], blocked[f], prices.index).items():
            print(f"    {f} {side:<8} {n:>6} trades, total {total:>+8.2%}, mean {mean:>+.4%}, win {win:.1%}")

    tables = variant_tables("News filter 2019-2023", results, dsr, sectors=M.PAIR_SECTORS)
    write_tables(tables, M.DASHBOARD_DIR, "news")

    best = max(FILTERS, key=lambda f: rows[f"SCREEN+{f}"]["t"])
    t1 = rows[f"SCREEN+{best}"]["t"]
    keeps_dist = rows[f"DISTANCE20+{best}"]["sharpe"] >= rows["DISTANCE20"]["sharpe"]
    stage1 = t1 >= MIN_T1 and keeps_dist
    print(f"\n  Stage 1: {best}, t {t1:.2f} on SCREEN (needs {MIN_T1}), DISTANCE20 Sharpe "
          f"{rows['DISTANCE20']['sharpe']:.2f} -> {rows[f'DISTANCE20+{best}']['sharpe']:.2f}: {'PASS' if stage1 else 'FAIL'}")

    advance = False
    if stage1:
        oos = fetch_prices(M.UNIVERSE, OOS_START, OOS_END, max_missing=M.MAX_MISSING, cache_dir=M.CACHE_DIR)
        schedule, prepared = prepared_book(oos, "baseline")
        plain = run(oos, schedule, prepared)
        mask = blocked_days(sets[best], oos.index, WINDOW)
        filtered = run(oos, schedule, prepared, entry_multipliers(prepared, mask, oos.index))
        t2 = paired_t(filtered.daily_pnl, plain.daily_pnl)
        advance = t2 >= MIN_T2
        print(f"  Stage 2, 2024-2026: Sharpe {summary_stats(plain)['sharpe']:.2f} -> {summary_stats(filtered)['sharpe']:.2f}, "
              f"t {t2:.2f} (needs {MIN_T2}): {'PASS' if advance else 'FAIL'}")
    print(f"  DECISION (pre-registered rule): {'ADVANCE ' + best if advance else 'DO NOT ADVANCE'}")


if __name__ == "__main__":
    main()
