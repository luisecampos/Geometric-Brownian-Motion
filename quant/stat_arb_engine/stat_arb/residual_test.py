"""
residual_test.py
----------------
Pre-registered test of the sector-ETF residual model (Avellaneda & Lee 2010).
The full design is SECTOR_RESIDUAL.md.

Written and committed BEFORE any signal or PnL ran on real prices. ETF listing
dates and stock counts were checked first, with no strategy data involved.
Nothing below may be changed after the run; any variation is a new trial and
must be reported as one.

Question: does a stock's move against its sector ETF revert strongly enough to
clear a net Sharpe of 1.0 at realistic costs, in every sample, at a size that fits
2x gross overnight?

Model, from the paper: 60-day regression on the ETF, AR(1) fit of the cumulative
residual, kappa > 8.4, centred s-score, open at |s| > 1.25, close long at s > -0.50,
close short at s < 0.75, close either when the stock is rejected. No stop, time
limit or earnings filter. See signals/residual.py.

Changes from the paper: fills at the close after the signal, not the signal's own
close; large caps chosen with hindsight; at most 20 positions at 5% of capital and
gross capped at 2.0x; hedge frozen at entry; no volume-adjusted signal. See
backtest/residual_book.py. 5%, 20 and 2.0x are set by a $25,000 account and the 2x
limit. None of them was searched.

Samples, each with its own warm-up. A sector runs only if its ETF has 100 calendar
days of history before trading starts, so real estate and communication sit out
the past sample:
  PAST    2010-2017, warm-up from 2009, 229 stocks, 9 sectors
  RECENT  2019-2023, warm-up from 2018, 281 stocks, 11 sectors
  FRESH   2024-01 to 2026-10-30, warm-up from 2023, 11 sectors
Nothing from 2026-11-02 onward may be read: load() asserts it. FRESH runs only
after a stage 1 pass. forward_test.py and its holdout are not touched.

Cost levels are cost_test.py's, all four. Stock orders are charged in full, ETF
orders after same-day netting in a sector, and borrow on short stocks and net short
ETF holdings. The result with ETF orders charged before netting is also reported.

Decision, on Sharpe at REALISTIC (5bp per leg, 0.50%/yr borrow):
  Stage 1  PAST >= 1.0 and RECENT >= 1.0, or ADVANCE: NO and stage 2 never runs
  Stage 2  FRESH >= 1.0, or NOT CONFIRMED
  Both     CANDIDATE. Not tradable yet: it needs a point-in-time universe check
           and its own frozen forward test of 252 days
A failure ends this configuration. Its thresholds, lag and slots stay as they are.

Reported with no verdict attached: DSR at N = 91 on RECENT and N = 1 on the others;
paired t against DISTANCE20 at REALISTIC on RECENT, both scaled to equal vol; Sharpe
by year; break-even cost; turnover; average and peak gross; average positions; days
with blocked entries; PnL from long-stock and short-stock trades before ETF costs;
and trades open across a one-day move of 0.25 or more in log price, because Yahoo
doesn't adjust every spin-off (past_test.py, DHR 2016).

Trials: 1 added to main.py's EXTRA_TRIALS, whatever the result.

Limits: the universe is today's list, which flatters buying laggards, most of all
in PAST, so a pass isn't proof. Large stocks are part of their own ETF. Sector
membership is today's map. 20 positions is far fewer than the paper's. Whole-share
rounding isn't modelled.

Run `python residual_test.py stage1`, then `python residual_test.py stage2` only
if stage 1 passes.
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
from data.universe import SECTOR_ETFS, SECTORS
from backtest.report import summary_stats
from backtest.export import variant_tables, write_tables
from backtest.residual_book import run_residual_book
from learn.evaluate import daily_sharpe, deflated_sharpe
from signals.residual import residual_scores
from cost_test import LEVELS, prepared_book, prior_daily_sharpes, run

SAMPLES = {                    # data start, data end, first trading date
    "PAST": ("2009-01-01", "2018-01-01", "2010-01-01"),
    "RECENT": ("2018-01-01", "2024-01-01", "2019-01-01"),
    "FRESH": ("2023-01-01", "2026-10-31", "2024-01-01"),
}
BAR = 1.0                      # Sharpe at REALISTIC, in every sample
ETF_FIRST = {"XLRE": "2015-10-08", "XLC": "2018-06-19"}    # the other nine date from 1998
ETF_HISTORY = pd.Timedelta(days=100)
HOLDOUT_START = pd.Timestamp("2026-11-02")
JUMP = 0.25                    # one-day move in log price
STAGE1_PATH = os.path.join("outputs", "residual_stage1.json")


def load(sample: str):
    start, end, trade_start = SAMPLES[sample]
    prices = fetch_prices(M.UNIVERSE, start, end, max_missing=M.MAX_MISSING, cache_dir=M.CACHE_DIR)
    assert prices.index.max() < HOLDOUT_START, "this test must not read the holdout"

    active = {s: etf for s, etf in SECTOR_ETFS.items()
              if pd.Timestamp(ETF_FIRST.get(etf, "1998-12-22")) + ETF_HISTORY <= pd.Timestamp(trade_start)}
    names = [c for c in prices.columns if c not in SECTOR_ETFS.values() and SECTORS[c] in active]
    etfs = pd.concat([fetch_prices([etf], start, end, max_missing=1.0, cache_dir=M.CACHE_DIR)[[etf]]
                      for etf in active.values()], axis=1).reindex(prices.index)
    assert etfs.index.max() < HOLDOUT_START, "this test must not read the holdout"
    return prices, prices[names], etfs, {c: active[SECTORS[c]] for c in names}


def run_levels(stocks, etfs, etf_of, scores, betas, start) -> dict:
    results = {}
    for level, (cost, fee) in LEVELS.items():
        results[level] = run_residual_book(stocks, etfs, etf_of, scores, betas, start=start,
                                           cost_bps=cost, borrow_fee=fee)
    keys = {tuple(sorted((k[0], t.entry_date) for k, r in res.per_pair.items() for t in r.trades))
            for res in results.values()}
    assert len(keys) == 1, "costs must not change the trades"
    cost, fee = LEVELS["REALISTIC"]
    results["REALISTIC, ETF before netting"] = run_residual_book(
        stocks, etfs, etf_of, scores, betas, start=start, cost_bps=cost, borrow_fee=fee, net_etf=False)
    return results


def economics(free, realistic) -> dict:
    """Costs are linear in the rate, so two runs give all of these."""
    years = len(free.daily_pnl) / 252
    gross = free.daily_pnl.sum()
    traded = free.stock_traded.sum() + free.etf_traded.sum()
    cost = LEVELS["REALISTIC"][0]
    borrow = gross - cost * traded - realistic.daily_pnl.sum()
    return {
        "gross_per_year": gross / years,
        "stock_traded_per_year": free.stock_traded.sum() / years,
        "etf_traded_per_year": free.etf_traded.sum() / years,
        "breakeven_bps": gross / traded * 1e4,
        "breakeven_with_borrow_bps": (gross - borrow) / traded * 1e4,
    }


def jump_trades(res, stocks: pd.DataFrame) -> list:
    moves = np.log(stocks).diff().abs()
    hit = []
    for key, r in res.per_pair.items():
        for t in r.trades:
            held = moves.loc[t.entry_date:t.exit_date, key[0]].iloc[1:]
            if len(held) and held.max() >= JUMP:
                hit.append((key[0], t.entry_date.date(), t.exit_date.date(), t.pnl))
    return hit


def report(sample: str, results: dict, stocks: pd.DataFrame, dsr: dict, label: str) -> None:
    print(f"\n  {sample}: {stocks.shape[1]} stocks")
    print(f"  {'Config':<31} {'Sharpe':>7} {'DSR':>6} {'Return':>9} {'Vol':>7} {'MaxDD':>9} {'Trades':>7} {'Win%':>6}")
    for name, res in results.items():
        st = summary_stats(res)
        print(f"  {name:<31} {st['sharpe']:>7.2f} {dsr[name]:>6.2f} {st['total_return']:>9.2%} {st['ann_vol']:>7.2%} "
              f"{st['max_dd']:>9.2%} {st['trades']:>7d} {st['win_rate'] * 100:>5.1f}%")
    print(f"  {label}")

    print("\n  Sharpe by year:")
    for name in LEVELS:
        x = results[name].daily_pnl
        yrs = {int(y): round(float(daily_sharpe(x[x.index.year == y]) * np.sqrt(252)), 2) for y in sorted(set(x.index.year))}
        print(f"    {name:<10} {yrs}")

    real = results["REALISTIC"]
    e = economics(results["FREE"], real)
    print(f"\n  Economics: gross {e['gross_per_year']:+.2%}/yr, traded {e['stock_traded_per_year']:.1f}x stock + "
          f"{e['etf_traded_per_year']:.1f}x ETF capital/yr, break-even {e['breakeven_bps']:.1f}bp per leg "
          f"({e['breakeven_with_borrow_bps']:.1f}bp with REALISTIC borrow)")
    print(f"  Gross: mean {real.gross.mean():.2f}x, peak {real.gross.max():.2f}x. Positions: mean {real.positions.mean():.1f}. "
          f"Days with blocked entries: {(real.blocked > 0).mean():.0%}")
    trades = [t for r in real.per_pair.values() for t in r.trades]
    longs = sum(t.pnl for t in trades if t.direction > 0)
    shorts = sum(t.pnl for t in trades if t.direction < 0)
    print(f"  Trade PnL before ETF costs: long stock {longs:+.2%}, short stock {shorts:+.2%}")
    hit = jump_trades(real, stocks)
    print(f"  Trades open across a one-day move >= {JUMP}: {len(hit)}, total PnL {sum(h[3] for h in hit):+.2%}")
    for name, entry, exit_, pnl in hit:
        print(f"    {name} {entry} -> {exit_}: {pnl:+.2%}")


def run_sample(sample: str, n_trials: int, trial_sharpes: list) -> tuple[dict, pd.DataFrame]:
    print(f"\nFetching {sample} {SAMPLES[sample][0]} -> {SAMPLES[sample][1]}...")
    prices, stocks, etfs, etf_of = load(sample)
    print(f"  Scoring {stocks.shape[1]} stocks against {etfs.shape[1]} ETFs...")
    scores, betas = residual_scores(stocks, etfs, etf_of)
    results = run_levels(stocks, etfs, etf_of, scores, betas, SAMPLES[sample][2])
    dsr = {name: deflated_sharpe(res.daily_pnl, trial_sharpes, n_trials)["dsr"] for name, res in results.items()}
    report(sample, results, stocks, dsr, f"DSR over N={n_trials} trial{'s' if n_trials > 1 else ''}")
    tables = variant_tables(f"Residual {sample}", {f"RESIDUAL {k}": v for k, v in results.items()},
                            {f"RESIDUAL {k}": v for k, v in dsr.items()}, sectors=M.PAIR_SECTORS)
    write_tables(tables, M.DASHBOARD_DIR, f"residual_{sample.lower()}")
    return results, prices


def against_distance20(residual, prices: pd.DataFrame) -> float:
    schedule, prepared = prepared_book(prices, "DISTANCE20")
    pairs = run(prices, schedule, prepared, *LEVELS["REALISTIC"]).daily_pnl
    a = residual.daily_pnl / residual.daily_pnl.std()
    b = pairs.reindex(a.index).fillna(0.0) / pairs.std()
    diff = a - b
    return daily_sharpe(diff) * np.sqrt(len(diff))


def stage1() -> None:
    past, _ = run_sample("PAST", 1, [])
    recent, prices = run_sample("RECENT", 19 + M.EXTRA_TRIALS, prior_daily_sharpes())
    print(f"\n  RESIDUAL minus DISTANCE20 at REALISTIC, equal vol, 2019-2023: t {against_distance20(recent['REALISTIC'], prices):.2f}")

    sharpes = {"PAST": summary_stats(past["REALISTIC"])["sharpe"], "RECENT": summary_stats(recent["REALISTIC"])["sharpe"]}
    passed = all(s >= BAR for s in sharpes.values())
    os.makedirs(os.path.dirname(STAGE1_PATH), exist_ok=True)
    with open(STAGE1_PATH, "w") as f:
        json.dump({"sharpe": sharpes, "passed": passed}, f, indent=2)
    print(f"\n  At REALISTIC: PAST {sharpes['PAST']:.2f}, RECENT {sharpes['RECENT']:.2f}, bar {BAR:.1f} in both")
    print(f"  DECISION (pre-registered rule): {'STAGE 1 PASS, run stage 2 once' if passed else 'ADVANCE: NO'}")


def stage2() -> None:
    if not os.path.exists(STAGE1_PATH) or not json.load(open(STAGE1_PATH))["passed"]:
        sys.exit("Stage 2 runs only after a stage 1 pass.")
    fresh, _ = run_sample("FRESH", 1, [])
    sharpe = summary_stats(fresh["REALISTIC"])["sharpe"]
    print(f"\n  At REALISTIC: FRESH {sharpe:.2f}, bar {BAR:.1f}")
    print(f"  DECISION (pre-registered rule): {'CANDIDATE' if sharpe >= BAR else 'NOT CONFIRMED'}")


if __name__ == "__main__":
    stages = {"stage1": stage1, "stage2": stage2}
    if len(sys.argv) != 2 or sys.argv[1] not in stages:
        sys.exit("Usage: python residual_test.py stage1 | stage2")
    stages[sys.argv[1]]()
