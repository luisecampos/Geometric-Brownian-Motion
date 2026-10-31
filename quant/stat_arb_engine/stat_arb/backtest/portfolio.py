"""
backtest/portfolio.py
----------------------
Run every pair from a walk-forward schedule simultaneously and combine them
into one portfolio.

A single pair's backtest tells you if that one relationship was profitable.
It doesn't tell you what happens when you run a book of them together —
whether the pairs actually diversify each other or all blow up on the same
day because they share a common factor (e.g. two energy-sector pairs both
getting hit by the same oil shock). That's what `pair_pnl_corr` is for.

The expensive part of a pair-period is the Kalman MLE fit, which doesn't
depend on entry/exit thresholds or the z-score lookback. `prepare_pair_periods`
does that work once; `run_portfolio_backtest` can then be re-run cheaply for
other thresholds (learn/tuning.py) or with per-pair size multipliers
(learn/metalabel.py, learn/allocation.py) on the same prepared data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from functools import cached_property

from signals.cache import DiskCache, fingerprint, parallel_map
from signals.kalman import KalmanHedgeRatio, KalmanResult
from signals.zscore import compute_zscore
from signals.sizing import vol_target_size
from signals.walkforward import PairPeriod
from backtest.engine import run_backtest, BacktestResults, COST_BPS


@dataclass
class PortfolioResults:
    equity_curve   : pd.Series
    daily_pnl      : pd.Series
    per_pair       : dict          # (ticker_a, ticker_b, period_start) -> BacktestResults
    pair_pnl       : pd.DataFrame  # daily PnL per pair-period, columns = pair labels
    kalman_params  : dict = field(default_factory=dict)  # label -> fitted Kalman params

    @cached_property
    def pair_pnl_corr(self) -> pd.DataFrame:
        """Correlation matrix of pair_pnl columns. Computed on first use:
        with thousands of pair-periods it is large, and only the tearsheet
        needs it."""
        return self.pair_pnl.corr()

    def avg_pair_corr(self) -> float:
        """
        Mean off-diagonal correlation of pair_pnl columns, without building
        the matrix: with standardised columns Z, sum(corr) = |Z 1|^2 / (T-1).
        Columns with zero variance (no trades) are left out, as corr() would
        give them NaN.
        """
        x = self.pair_pnl.values
        sd = x.std(axis=0, ddof=1)
        keep = sd > 0
        k = int(keep.sum())
        if k < 2:
            return float("nan")
        z = (x[:, keep] - x[:, keep].mean(axis=0)) / sd[keep]
        total = float(np.sum(z.sum(axis=1) ** 2)) / (len(x) - 1)
        return (total - k) / (k * k - k)


@dataclass(frozen=True)
class ExitRules:
    """
    Optional rules for trades that haven't reverted (see exit_test.py). Both
    multiples are of the pair's half-life on its selection window.

    late_entry_mult: no new entry with fewer than mult x half-life trading
        days left in the period.
    time_stop_mult : close a trade after mult x half-life trading days held.
    carry          : a position open at period end stays open if the same
        pair is selected for the next period. The next selection uses the
        period_end close, so it is only known after that close; a position
        in a pair that is NOT re-selected therefore closes one day later, at
        the first close of the next period. The next period's Kalman fit
        and z-score take over for the exit, the shares stay as entered.
    """
    late_entry_mult: float | None = None
    time_stop_mult : float | None = None
    carry          : bool = False


@dataclass
class PreparedPair:
    """One pair-period's causal inputs: selection window + active period."""
    pp         : PairPeriod
    label      : str
    series_a   : pd.Series      # prices from train_start through period_end
    series_b   : pd.Series
    train_size : int            # rows before period_start
    kf         : KalmanResult
    size       : pd.Series      # vol-target gross notional / capital
    kalman_params: dict
    _z_cache   : dict = field(default_factory=dict, repr=False)

    def zscores(self, lookback: int = 60, signal: str = "rolling") -> pd.Series:
        key = (signal, lookback)
        if key not in self._z_cache:
            if signal == "rolling":
                self._z_cache[key] = compute_zscore(self.kf.spread, window=lookback)
            elif signal == "innovation":
                self._z_cache[key] = self.kf.innovation_z
            else:
                raise ValueError(f"unknown signal: {signal!r}")
        return self._z_cache[key]

    @property
    def first(self) -> int:
        """Window position the engine starts on: the day before period_start."""
        return max(0, self.train_size - 1)


def pair_label(pp: PairPeriod) -> str:
    return f"{pp.ticker_a}/{pp.ticker_b} [{pp.period_start.date()}]"


def _fit_kalman(args):
    """Worker: one pair-period's Kalman fit -> (KalmanResult, fitted params)."""
    series_a, series_b, train_size = args
    kalman = KalmanHedgeRatio()
    kf = kalman.fit(series_a, series_b, train_size=train_size)
    return kf, kalman.fitted_params


def prepare_pair_periods(
    prices          : pd.DataFrame,
    pair_schedule   : list[PairPeriod],
    target_daily_vol: float | dict = 0.01,
    train_window    : int = 252,
    n_jobs          : int = 1,
    cache_dir       : str | None = None,
) -> list[PreparedPair]:
    """
    Fit each pair-period's Kalman filter (centring, OLS seed, MLE noise
    parameters) on that pair's own selection window — the `train_window`
    days before period_start, the same data walk-forward used to pick it —
    and filter causally through the active period. Also computes the
    vol-target size series.

    target_daily_vol: one per-pair target, or {period_start: target} to set
        it per rebalance (e.g. a book budget split across that period's
        pairs, which is known at period_start).
    n_jobs / cache_dir: fit in a process pool, and cache fits on disk keyed
        by a hash of the pair's window prices and train size.
    """
    windows = []
    for pp in pair_schedule:
        if pp.train_start is not None:
            window = prices.loc[pp.train_start:pp.period_end]
        else:
            start_pos = max(0, prices.index.get_loc(pp.period_start) - train_window)
            window = prices.loc[prices.index[start_pos]:pp.period_end]
        series_a = window[pp.ticker_a]
        series_b = window[pp.ticker_b]
        train_size = int((window.index < pp.period_start).sum())
        windows.append((series_a, series_b, train_size))

    cache = DiskCache(cache_dir, "kalman")
    keys = [fingerprint(a.values, b.values, a.index[0], a.index[-1], ts) for a, b, ts in windows]
    todo = [k for k, key in enumerate(keys) if cache.get(key) is None]
    fits = parallel_map(_fit_kalman, [windows[k] for k in todo], n_jobs, chunksize=16)
    for k, fit in zip(todo, fits):
        cache.put(keys[k], fit)
    cache.save()

    prepared = []
    for pp, (series_a, series_b, train_size), key in zip(pair_schedule, windows, keys):
        kf, params = cache.get(key)
        vol = target_daily_vol[pp.period_start] if isinstance(target_daily_vol, dict) else target_daily_vol
        size = vol_target_size(series_a, series_b, kf.hedge_ratio, target_daily_vol=vol)

        prepared.append(PreparedPair(
            pp=pp, label=pair_label(pp), series_a=series_a, series_b=series_b,
            train_size=train_size, kf=kf, size=size,
            kalman_params=params,
        ))
    return prepared


def run_portfolio_backtest(
    prices          : pd.DataFrame,
    pair_schedule   : list[PairPeriod],
    lookback_window : int = 60,
    entry_threshold : float = 2.0,
    exit_threshold  : float = 0.5,
    stop_threshold  : float | None = 3.5,
    target_daily_vol: float = 0.01,
    signal          : str = "rolling",
    train_window    : int = 252,
    prepared        : list[PreparedPair] | None = None,
    size_multipliers: dict | None = None,
    exit_rules      : ExitRules | None = None,
    cost_bps        : float = COST_BPS,
    borrow_fee      : float = 0.0,
) -> PortfolioResults:
    """
    Run the full pipeline (Kalman hedge ratio -> z-score -> vol-target sizing
    -> backtest) for every PairPeriod in `pair_schedule`, restricted to each
    period's own active date range, then sum them into one portfolio.

    Nothing here re-introduces look-ahead: each Kalman filter is fit on its
    selection window only (see prepare_pair_periods), and the z-score / vol
    warm-ups are complete by the first tradeable day.

    signal: "rolling"    -> rolling z-score of the spread (lookback_window)
            "innovation" -> Kalman innovation z = e / sqrt(S)

    prepared: output of prepare_pair_periods (same schedule, same vol
        target), to skip re-fitting the Kalman filters.
    size_multipliers: optional {pair label: float | pd.Series}. A Series is
        indexed by DECISION date (the close the engine decides on) and
        multiplies size there; 0 means "don't enter". This is the hook the
        learning layer uses to skip or resize trades.
    exit_rules: optional ExitRules. None (or all defaults) leaves every
        result unchanged.
    cost_bps, borrow_fee: see backtest.engine.run_backtest.

    PnL is a return on book capital: each pair's daily_pnl is its return on
    the whole book, so the portfolio return is their sum.
    """
    if prepared is None:
        prepared = prepare_pair_periods(prices, pair_schedule, target_daily_vol, train_window)
    size_multipliers = size_multipliers or {}
    rules = exit_rules or ExitRules()
    if rules.carry:
        starts = [prep.pp.period_start for prep in prepared]
        assert starts == sorted(starts), "carry needs pair-periods in time order"
    selected = {(p.pp.ticker_a, p.pp.ticker_b, p.pp.period_start) for p in prepared}
    carried: dict = {}         # (ticker_a, ticker_b) -> OpenPosition

    per_pair  : dict = {}
    pair_pnl_cols: dict[str, pd.Series] = {}
    kalman_params: dict = {}

    for prep in prepared:
        pp = prep.pp
        zscores = prep.zscores(lookback_window, signal)
        size = prep.size
        mult = size_multipliers.get(prep.label)
        if mult is not None:
            if isinstance(mult, pd.Series):
                mult = mult.reindex(size.index).fillna(1.0)
            size = size * mult

        # Start the backtest one day BEFORE period_start, so day-1 decisions
        # can use the prior close's z / hedge / size (the engine always acts
        # on [t-1]). The engine can't enter on its own first row, so no
        # position is ever opened before period_start.
        active = slice(prep.first, None)
        series_a, series_b = prep.series_a.iloc[active], prep.series_b.iloc[active]
        hedge, z, gross = prep.kf.hedge_ratio.iloc[active], zscores.iloc[active], size.iloc[active]
        n = len(series_a)

        no_entry_from = None
        if rules.late_entry_mult is not None:
            no_entry_from = n - int(np.ceil(rules.late_entry_mult * pp.half_life))
        time_stop = None
        if rules.time_stop_mult is not None:
            time_stop = int(np.ceil(rules.time_stop_mult * pp.half_life))

        carry_out = False
        if rules.carry:
            pos = prices.index.get_loc(pp.period_end) + 1
            nxt = prices.index[pos] if pos < len(prices.index) else None
            carry_out = nxt is not None and (pp.ticker_a, pp.ticker_b, nxt) in selected
            if nxt is not None and not carry_out:
                # One more row, for the close of any open position. Its z /
                # hedge / size are never read: the engine acts on [t-1] and
                # no entry is allowed here or on period_end, as usual.
                no_entry_from = min(n - 1, n if no_entry_from is None else no_entry_from)
                series_a = pd.concat([series_a, prices[pp.ticker_a].iloc[[pos]]])
                series_b = pd.concat([series_b, prices[pp.ticker_b].iloc[[pos]]])
                hedge, z, gross = (x.reindex(series_a.index) for x in (hedge, z, gross))

        results = run_backtest(
            series_a, series_b, hedge, z,
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold,
            stop_threshold=stop_threshold,
            size=gross,
            cost_bps=cost_bps,
            borrow_fee=borrow_fee,
            time_stop=time_stop,
            no_entry_from=no_entry_from,
            carry_in=carried.pop((pp.ticker_a, pp.ticker_b), None),
            carry_out=carry_out,
        )
        if results.open_position is not None:
            carried[(pp.ticker_a, pp.ticker_b)] = results.open_position

        per_pair[(pp.ticker_a, pp.ticker_b, pp.period_start)] = results
        pair_pnl_cols[prep.label] = results.daily_pnl
        kalman_params[prep.label] = prep.kalman_params

    if not pair_pnl_cols:
        raise ValueError("No pair-periods to backtest — pair_schedule was empty.")

    calendar = prices.loc[min(prep.pp.period_start for prep in prepared):].index
    return combine_pair_results(per_pair, pair_pnl_cols, calendar, kalman_params)


def combine_pair_results(
    per_pair      : dict,
    pair_pnl_cols : dict[str, pd.Series],
    calendar      : pd.DatetimeIndex,
    kalman_params : dict | None = None,
) -> PortfolioResults:
    """Sum per-pair-period daily PnL into one book on a common calendar."""
    # Reindex onto the full calendar so days with nothing active count as
    # flat (0 return) days — otherwise Sharpe/vol only see "in-market" days.
    # (Filled column by column: pd.concat of thousands of differently-indexed
    # Series is slow, and this runs once per variant and grid point.)
    values = np.zeros((len(calendar), len(pair_pnl_cols)))
    for k, s in enumerate(pair_pnl_cols.values()):
        pos = calendar.get_indexer(s.index)
        ok = pos >= 0
        values[pos[ok], k] = np.nan_to_num(s.values[ok], nan=0.0)
    pair_pnl = pd.DataFrame(values, index=calendar, columns=list(pair_pnl_cols))
    daily_pnl = pair_pnl.sum(axis=1).rename("daily_pnl")
    equity_curve = daily_pnl.cumsum().rename("equity")

    return PortfolioResults(
        equity_curve  = equity_curve,
        daily_pnl     = daily_pnl,
        per_pair      = per_pair,
        pair_pnl      = pair_pnl,
        kalman_params = kalman_params or {},
    )
