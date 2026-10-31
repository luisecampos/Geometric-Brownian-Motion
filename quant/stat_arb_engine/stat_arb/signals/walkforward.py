"""
signals/walkforward.py
-----------------------
Roll a cointegration re-fit forward through time instead of testing once on
the full history.

Why this matters:
  The original pipeline picked its pair(s) with one Engle-Granger test over
  the *entire* backtest window, then traded that same pair for the entire
  window. That's in-sample selection — you're using information ("this pair
  stayed cointegrated the whole time") that you couldn't have known on day 1.

  Walk-forward fixes this: at each rebalance date, only look at the trailing
  `train_window` days to decide which pairs are tradeable, then trade those
  pairs for the next `test_window` days without peeking further ahead. Roll
  forward and repeat. A pair can drop in and out of the active set over time,
  same as it would on a real desk re-running pair selection periodically.

Multiple testing:
  15 tickers = 105 pairs tested every quarter. At p < 0.05 with NO real
  cointegration anywhere you'd still expect ~5 "cointegrated" pairs per
  window — pure false positives that then get traded. Two corrections:
    - "bh"         : Benjamini-Hochberg, controls the false DISCOVERY rate at
                     `pvalue_cutoff` (of the pairs you pick, at most ~5% are
                     flukes on average). Default — less conservative, keeps power.
    - "bonferroni" : p < pvalue_cutoff / n_tests, controls the chance of even
                     ONE false positive per window. Very strict at n=105.
    - None         : the old raw cutoff.

Half-life filter:
  Even a genuinely cointegrated pair is useless if it reverts on a 200-day
  horizon inside a 63-day trading window. Pairs whose training-window spread
  half-life falls outside [min_half_life, max_half_life] are dropped.

Sector restriction:
  On a wide universe, testing every pair is both slow (C(300, 2) ~ 45,000
  tests per window) and mostly pointless: a utility and a chip maker can
  pass Engle-Granger by chance, but there is no economic reason for the
  spread to revert. Passing `sectors` (ticker -> sector) restricts the
  candidate family to same-sector pairs, and the multiple-testing correction
  is applied to that family.

Tests run on LOG prices, matching the log-space Kalman hedge. With
`n_jobs` > 1 they run in a process pool, and with `cache_dir` each window's
results are cached on disk, keyed by a hash of the window's prices.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass

from itertools import combinations

from signals.cache import DiskCache, fingerprint, parallel_map
from signals.cointegration import _coint_chunk, spread_half_life


@dataclass
class PairPeriod:
    ticker_a     : str
    ticker_b     : str
    pvalue       : float
    period_start : pd.Timestamp   # first tradeable date (out-of-sample)
    period_end   : pd.Timestamp   # last tradeable date, exclusive of next period
    train_start  : pd.Timestamp = None   # first date of the selection window
    half_life    : float = float("nan")  # training-window spread half-life (days)
    adf_stat     : float = float("nan")  # Engle-Granger ADF statistic on the selection window


def select_pairs(
    pvals            : list[tuple[str, str, float]],
    alpha            : float,
    multiple_testing : str | None = "bh",
) -> list[tuple[str, str, float]]:
    """
    Apply a multiple-testing rule to the FULL family of (a, b, p) results.

    Benjamini-Hochberg: sort p ascending, find the largest k with
    p_(k) <= (k / m) * alpha, and accept the first k.
    """
    m = len(pvals)
    if m == 0:
        return []
    ranked = sorted(pvals, key=lambda x: x[2])

    if multiple_testing is None:
        return [r for r in ranked if r[2] < alpha]
    if multiple_testing == "bonferroni":
        return [r for r in ranked if r[2] < alpha / m]
    if multiple_testing == "bh":
        p = np.array([r[2] for r in ranked])
        passed = np.nonzero(p <= alpha * np.arange(1, m + 1) / m)[0]
        k = passed[-1] + 1 if len(passed) else 0
        return ranked[:k]
    raise ValueError(f"unknown multiple_testing: {multiple_testing!r}")


def _candidate_pairs(tickers: list[str], sectors: dict | None) -> list[tuple]:
    if sectors is None:
        return list(combinations(tickers, 2))
    return [(a, b) for a, b in combinations(tickers, 2)
            if sectors.get(a) is not None and sectors.get(a) == sectors.get(b)]


def generate_pair_schedule(
    prices           : pd.DataFrame,
    universe         : list[str],
    train_window     : int   = 252,
    test_window      : int   = 63,
    pvalue_cutoff    : float = 0.05,
    multiple_testing : str | None = "bh",
    min_half_life    : float = 5.0,
    max_half_life    : float = 60.0,
    verbose          : bool  = False,
    sectors          : dict | None = None,
    n_jobs           : int = 1,
    cache_dir        : str | None = None,
) -> list[PairPeriod]:
    """
    Roll a `train_window`-day cointegration test forward in `test_window`-day
    steps across `prices`.

    At each step, tests every pair in `universe` for cointegration using only
    the trailing `train_window` days (data strictly before `period_start`),
    corrects for testing all of them at once, drops pairs whose half-life is
    outside [min_half_life, max_half_life], and makes the survivors tradeable
    for the following `test_window` days. No pair is ever selected using data
    from its own trading period.

    Returns a flat list of PairPeriod, one entry per (pair, active window).
    Multiple pairs can be active in the same period; the same pair can
    reappear in a later period if it re-qualifies.

    sectors: optional {ticker: sector}; only same-sector pairs are tested.
    """
    prices = prices[universe].dropna()
    log_px = np.log(prices)
    dates  = prices.index
    n      = len(dates)

    tickers = log_px.columns.tolist()
    pairs = _candidate_pairs(tickers, sectors)
    pos = {t: k for k, t in enumerate(tickers)}

    # Every window's train slice, then all Engle-Granger tests in one pass:
    # cached windows are read back, the rest go to one process pool.
    windows = []
    start = train_window
    while start < n:
        end = min(start + test_window, n)
        windows.append((start, end))
        start = end

    cache = DiskCache(cache_dir, "coint")
    tests: dict = {}
    todo, blocks = [], []
    for start, end in windows:
        values = log_px.values[start - train_window : start]
        key = fingerprint(values, tickers, pairs)
        if cache.get(key) is not None:
            tests[start] = cache.get(key)
            continue
        idx_pairs = [(pos[a], pos[b]) for a, b in pairs]
        nb = 0
        for k in range(0, len(idx_pairs), 150):
            blocks.append((values, idx_pairs[k:k + 150]))
            nb += 1
        todo.append((start, key, nb))
    if blocks:
        if verbose:
            print(f"      Running {len(todo) * len(pairs)} Engle-Granger tests "
                  f"({len(pairs)} pairs x {len(todo)} uncached windows, {n_jobs} process(es))...")
        out = parallel_map(_coint_chunk, blocks, n_jobs)
        k = 0
        for start, key, nb in todo:
            res = {}
            for block in out[k:k + nb]:
                for i, j, stat, pval in block:
                    res[(tickers[i], tickers[j])] = (stat, pval)
            k += nb
            tests[start] = res
            cache.put(key, res)
        cache.save()

    schedule: list[PairPeriod] = []

    for start, end in windows:
        train_slice  = log_px.iloc[start - train_window : start]
        period_start = dates[start]
        period_end   = dates[end - 1]

        res       = tests[start]
        all_pvals = sorted(((a, b, p) for (a, b), (_, p) in res.items()), key=lambda x: x[2])
        passed    = select_pairs(all_pvals, pvalue_cutoff, multiple_testing)

        kept = 0
        for ticker_a, ticker_b, pval in passed:
            hl = spread_half_life(train_slice[ticker_a], train_slice[ticker_b])
            if not (min_half_life <= hl <= max_half_life):
                continue
            kept += 1
            schedule.append(PairPeriod(
                ticker_a     = ticker_a,
                ticker_b     = ticker_b,
                pvalue       = pval,
                period_start = period_start,
                period_end   = period_end,
                train_start  = dates[start - train_window],
                half_life    = hl,
                adf_stat     = res[(ticker_a, ticker_b)][0],
            ))

        if verbose:
            raw = sum(p < pvalue_cutoff for *_, p in all_pvals)
            print(f"      {period_start.date()}: {raw:>3d} raw p<{pvalue_cutoff}"
                  f"  -> {len(passed):>2d} after {multiple_testing or 'no'} correction"
                  f"  -> {kept:>2d} after half-life filter")

    return schedule


def persistent_pairs(schedule: list[PairPeriod], windows: int = 2) -> list[PairPeriod]:
    """
    Keep a pair-period only if the same pair is also in the schedule in each
    of the previous `windows - 1` rebalances. Rebalances are the schedule's
    own period starts, so nothing survives in the first `windows - 1` of
    them. Causal: it only looks backwards.
    """
    starts = sorted({pp.period_start for pp in schedule})
    index = {T: k for k, T in enumerate(starts)}
    present = {(pp.ticker_a, pp.ticker_b, index[pp.period_start]) for pp in schedule}
    return [pp for pp in schedule
            if all((pp.ticker_a, pp.ticker_b, index[pp.period_start] - k) in present
                   for k in range(1, windows))]


def distance_schedule(
    prices        : pd.DataFrame,
    universe      : list[str],
    train_window  : int   = 252,
    test_window   : int   = 63,
    top_n         : int   = 20,
    min_half_life : float = 5.0,
    max_half_life : float = 60.0,
    sectors       : dict | None = None,
) -> list[PairPeriod]:
    """
    The distance method (Gatev, Goetzmann and Rouwenhorst, 2006) on the same
    rolling windows as generate_pair_schedule: rank candidate pairs by the
    sum of squared differences of their log prices, each rebased to 0 at the
    start of the train window, and take the `top_n` closest whose half-life
    is in range. No cointegration test, so `pvalue` is NaN.
    """
    log_px = np.log(prices[universe].dropna())
    dates  = log_px.index
    n      = len(dates)

    tickers = log_px.columns.tolist()
    pairs = _candidate_pairs(tickers, sectors)
    pos = {t: k for k, t in enumerate(tickers)}
    ia = np.array([pos[a] for a, _ in pairs])
    ib = np.array([pos[b] for _, b in pairs])

    schedule: list[PairPeriod] = []
    start = train_window
    while start < n:
        end = min(start + test_window, n)
        train = log_px.values[start - train_window : start]
        x = train - train[0]
        ssd = ((x[:, ia] - x[:, ib]) ** 2).sum(axis=0)

        kept = 0
        for k in np.argsort(ssd, kind="stable"):
            hl = spread_half_life(train[:, ia[k]], train[:, ib[k]])
            if not (min_half_life <= hl <= max_half_life):
                continue
            schedule.append(PairPeriod(
                ticker_a     = pairs[k][0],
                ticker_b     = pairs[k][1],
                pvalue       = float("nan"),
                period_start = dates[start],
                period_end   = dates[end - 1],
                train_start  = dates[start - train_window],
                half_life    = hl,
            ))
            kept += 1
            if kept == top_n:
                break
        start = end

    return schedule
