"""
past_test.py
------------
Pre-registered test of DISTANCE20 on 2010-2017, years this project never used.

Written and committed BEFORE any PnL ran on these years. The prices were
downloaded and checked for coverage first, with no strategy data involved.
Nothing below may be changed after the run; any variation is a new trial and
must be reported as one.

Question: does the book in the forward holdout make money in eight years that
played no part in choosing it? Its 0.70 on 2019-2023 is in-sample and was the
best of 3 candidates, so this is the first look at it on unused data.

This is NOT a clean out-of-sample test. The universe is today's list, so a
company that failed or was bought before 2026 isn't in it, and only names
already trading in 2009 qualify. That bias is worse here than in any later
sample and it favours mean reversion, because the losers that never came back
are missing. A pass is support, not proof. The forward holdout, whose universe
is fixed in advance, remains the real test.

Data: main.py's universe from 2009-01-01 to 2018-01-01, the same 1% missing-data
rule. Coverage, checked before this was committed: 271 of 300 names, 3,383
same-sector candidate pairs, 32 quarters, trading from 2010-01-04 to 2017-12-29.
2009 only selects pairs. The dropped names are the ones listed after 2009 and 9
Yahoo no longer serves, left as they are: AVB, BK, CTRA, EA, EQR, IPG, MMC, SEE,
XLC. 2018 is not used, so nothing overlaps the 2019-2023 sample.

Known data flaw, left as it is: Yahoo's prices aren't adjusted for every
spin-off. DHR falls 38% on 2016-07-05 when Fortive was spun off, and no such
loss happened. Any trade open across a one-day move of 0.25 or more in log
price on either leg is listed, and the Sharpe without those trades is reported
with no verdict attached. Removing them is not part of the decision.

Setup, unchanged: pair_test.py's DISTANCE20, e=2 x=0.5 lb=60, no exit rules, no
learning layer, 0.8%/day budget. Cost levels are cost_test.py's, all four.

Decision, on DISTANCE20 at REALISTIC (5bp per leg, 0.50%/yr borrow). DSR is over
1 trial, because only one configuration is judged on these years:
  SUPPORTED      Sharpe > 0 and DSR >= 0.95, about a Sharpe of 0.6 over 8 years
  WEAK           Sharpe > 0
  NOT SUPPORTED  Sharpe <= 0
The decision does not change the holdout, which runs whatever this shows.

Reported with no verdict attached: SCREEN (the baseline) on the same data at the
same costs. It is not a candidate. It shares the survivorship bias, so it shows
how far the bias alone lifts a book that lost money on 2019-2023. Also Sharpe by
year, the economics from cost_test.py, and the paired t of DISTANCE20 minus
SCREEN at REALISTIC.

Trials: 1 added to main.py's EXTRA_TRIALS, whatever the result.
"""

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

import main as M
from data.fetch import fetch_prices
from backtest.report import summary_stats
from backtest.export import variant_tables, write_tables
from learn.evaluate import daily_sharpe, deflated_sharpe
from cost_test import BOOKS, LEVELS, economics, prepared_book, run, trade_dates

DATA_START, DATA_END = "2009-01-01", "2018-01-01"
MIN_DSR = 0.95
JUMP = 0.25                    # one-day move in log price


def jump_trades(res, prices: pd.DataFrame) -> tuple[list, pd.Series]:
    """Trades open across a JUMP on either leg, and the book without them."""
    moves = np.log(prices).diff().abs()
    book, hit = res.daily_pnl.copy(), []
    for (a, b, _), r in res.per_pair.items():
        for t in r.trades:
            held = moves.loc[t.entry_date:t.exit_date, [a, b]].iloc[1:]
            if len(held) and held.max().max() >= JUMP:
                hit.append((a, b, t.entry_date.date(), t.exit_date.date(), t.pnl))
                pnl = r.daily_pnl.loc[t.entry_date:t.exit_date]
                book = book.sub(pnl.reindex(book.index).fillna(0.0))
    return hit, book


def main():
    print(f"Fetching {len(M.UNIVERSE)} tickers {DATA_START} -> {DATA_END}...")
    prices = fetch_prices(M.UNIVERSE, DATA_START, DATA_END, max_missing=M.MAX_MISSING, cache_dir=M.CACHE_DIR)

    results = {}
    for book, selection in BOOKS.items():
        schedule, prepared = prepared_book(prices, selection)
        print(f"  {book}: {len(schedule)} pair-periods, {len({pp.period_start for pp in schedule})} quarters")
        for level, (cost, fee) in LEVELS.items():
            results[f"{book} {level}"] = run(prices, schedule, prepared, cost, fee)
        same = {tuple(trade_dates(results[f"{book} {level}"])) for level in LEVELS}
        assert len(same) == 1, "costs must not change the trades"

    print()
    print(f"  {'Config':<21} {'Sharpe':>7} {'DSR':>6} {'Return':>9} {'Vol':>7} {'MaxDD':>9} {'Trades':>7} {'Win%':>6}")
    rows, dsr = {}, {}
    for name, res in results.items():
        st = summary_stats(res)
        dsr[name] = deflated_sharpe(res.daily_pnl, [], 1)["dsr"]
        rows[name] = st
        print(f"  {name:<21} {st['sharpe']:>7.2f} {dsr[name]:>6.2f} {st['total_return']:>9.2%} {st['ann_vol']:>7.2%} "
              f"{st['max_dd']:>9.2%} {st['trades']:>7d} {st['win_rate'] * 100:>5.1f}%")
    print("  DSR over N=1 trial")

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

    best, base = results["DISTANCE20 REALISTIC"].daily_pnl, results["SCREEN REALISTIC"].daily_pnl
    diff = best - base.reindex(best.index).fillna(0.0)
    print(f"\n  DISTANCE20 minus SCREEN at REALISTIC: t {daily_sharpe(diff) * np.sqrt(len(diff)):.2f}")

    hit, without = jump_trades(results["DISTANCE20 REALISTIC"], prices)
    print(f"\n  DISTANCE20 REALISTIC trades open across a one-day move >= {JUMP}: {len(hit)}")
    for a, b, entry, exit_, pnl in hit:
        print(f"    {a}/{b} {entry} -> {exit_}: {pnl:+.2%}")
    print(f"  Sharpe without them: {daily_sharpe(without) * np.sqrt(252):.2f}")

    tables = variant_tables("Past test 2010-2017", results, dsr, sectors=M.PAIR_SECTORS)
    write_tables(tables, M.DASHBOARD_DIR, "past")

    sharpe, d = rows["DISTANCE20 REALISTIC"]["sharpe"], dsr["DISTANCE20 REALISTIC"]
    decision = "SUPPORTED" if sharpe > 0 and d >= MIN_DSR else "WEAK" if sharpe > 0 else "NOT SUPPORTED"
    print(f"\n  DISTANCE20 at REALISTIC: Sharpe {sharpe:.2f}, DSR {d:.2f}")
    print(f"  DECISION (pre-registered rule): {decision}")


if __name__ == "__main__":
    main()
