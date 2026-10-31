"""
backtest/engine.py
------------------
Tick through history day by day, open/close positions based on z-score
thresholds, and track PnL as a RETURN ON CAPITAL.

Position logic:
  - We are always either FLAT, LONG_SPREAD, or SHORT_SPREAD
  - LONG_SPREAD  = long A, short B*hedge_ratio  (bet spread widens back up)
  - SHORT_SPREAD = short A, long B*hedge_ratio  (bet spread narrows back down)

Sizing:
  `size[t]` is gross notional as a fraction of book capital (see
  signals/sizing.py), split across the legs by the hedge ratio. At entry the
  dollar legs are converted to SHARE counts, which are then held fixed until
  exit — no silent daily rebalancing, so the only trading costs are the
  entry and exit trades actually modelled here. Daily PnL is
  sum(shares * price change), i.e. a return on $1 of capital.

Costs: COST_BPS per leg on the dollar notional traded, at entry AND exit,
  both included in Trade.pnl.

Timing (no look-ahead):
  Every decision — entry, exit, stop — uses information up to the close of
  t-1 (z[t-1], hedge_ratio[t-1], size[t-1]) and executes at the close of t.
  The old engine entered on z[t-1] but exited on z[t] at the same close, and
  marked PnL with beta[t], which had already been updated with day t's
  prices. Both are fixed: entry/exit share one rule, and the hedge is frozen
  at entry.

Exits:
  - "signal"     : |z| < exit_threshold (spread reverted), or z unavailable
  - "stop"       : |z| > stop_threshold (spread blew out — pair likely broke)
  - "period_end" : force-closed on the last day of the active window, costs
                   charged. Previously these positions just vanished from the
                   trade log and never paid their exit cost.
  After a stop, no re-entry until |z| falls back inside the entry band, so a
  broken pair doesn't get re-entered the next morning at |z|=3.6.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass


FLAT         =  0
LONG_SPREAD  =  1   # long A, short B
SHORT_SPREAD = -1   # short A, long B

COST_BPS = 2e-4     # 2 basis points per leg per trade


@dataclass
class Trade:
    entry_date  : pd.Timestamp
    exit_date   : pd.Timestamp
    direction   : int    # LONG_SPREAD or SHORT_SPREAD
    entry_z     : float
    exit_z      : float
    pnl         : float  # return on capital, net of entry + exit costs
    exit_reason : str   = "signal"   # "signal" | "stop" | "period_end"
    hedge_ratio : float = float("nan")   # beta frozen at entry
    gross       : float = float("nan")   # gross notional / capital at entry
    holding_days: int   = 0              # trading days held


@dataclass
class OpenPosition:
    """A position handed from one pair-period to the next (see carry_in)."""
    direction  : int
    sh_a       : float
    sh_b       : float
    entry_date : pd.Timestamp
    entry_z    : float
    hedge_ratio: float
    gross      : float
    held_days  : int             # trading days held so far
    pnl        : float           # PnL so far, entry cost included


@dataclass
class BacktestResults:
    equity_curve : pd.Series     # cumulative return on capital
    trades       : list[Trade]
    daily_pnl    : pd.Series     # daily return on capital
    open_position: OpenPosition | None = None   # set only with carry_out=True


def run_backtest(
    series_a        : pd.Series,
    series_b        : pd.Series,
    hedge_ratios    : pd.Series,
    zscores         : pd.Series,
    entry_threshold : float = 2.0,
    exit_threshold  : float = 0.5,
    stop_threshold  : float | None = 3.5,
    size            : pd.Series = None,
    cost_bps        : float = COST_BPS,
    time_stop       : int | None = None,
    no_entry_from   : int | None = None,
    carry_in        : OpenPosition | None = None,
    carry_out       : bool = False,
) -> BacktestResults:
    """
    Simulate the pairs trading strategy day by day.

    Args:
        series_a/b      : price series for each leg
        hedge_ratios    : log-space (dollar) hedge ratio each day, as of that
                          day's close
        zscores         : z-score of the spread, as of each day's close
        entry_threshold : |z| above which we enter a trade
        exit_threshold  : |z| below which we exit a trade
        stop_threshold  : |z| above which an open trade is stopped out (and
                          new entries are refused). None disables the stop.
        size            : gross notional / capital per day (e.g. from
                          signals.sizing.vol_target_size). Defaults to 1.0.
        cost_bps        : one-way cost per leg, as a fraction of notional
        time_stop       : close a trade held this many trading days (carried
                          days included), exit_reason "time". Blocks re-entry
                          the way the stop does. None disables it.
        no_entry_from   : refuse new entries from this row on.
        carry_in        : a position to start with, held from before row 0.
                          Row 0 must be the last row of the run it came from.
        carry_out       : don't force-close on the last day; hand any open
                          position back in BacktestResults.open_position.

    Returns:
        BacktestResults with equity curve, trade log, and daily returns.
        Any position open on the last day is force-closed there, unless
        carry_out is set. A carried trade is logged once, by the run that
        closes it, with its whole PnL and holding period.
    """
    dates = series_a.index
    n     = len(dates)
    a     = series_a.values.astype(float)
    b     = series_b.values.astype(float)
    beta  = hedge_ratios.values.astype(float)
    z     = zscores.values.astype(float)
    sz    = size.values.astype(float) if size is not None else np.ones(n)
    stop  = np.inf if stop_threshold is None else stop_threshold
    no_entry_from = n if no_entry_from is None else no_entry_from

    position    = FLAT
    sh_a = sh_b = 0.0          # shares held, per $1 of capital (signed)
    entry_idx   = None
    entry_z     = np.nan
    entry_beta  = np.nan
    entry_gross = np.nan
    entry_date  = None
    prior_days  = 0            # holding days and PnL from before row 0
    prior_pnl   = 0.0
    stopped_out = False
    if carry_in is not None:
        position, sh_a, sh_b = carry_in.direction, carry_in.sh_a, carry_in.sh_b
        entry_idx, entry_date = 0, carry_in.entry_date
        entry_z, entry_beta, entry_gross = carry_in.entry_z, carry_in.hedge_ratio, carry_in.gross
        prior_days, prior_pnl = carry_in.held_days, carry_in.pnl
    daily_pnl   = np.zeros(n)
    trades      = []

    for t in range(1, n):
        # ── Mark open position to market over (t-1, t] ───────────────────
        if position != FLAT:
            daily_pnl[t] += sh_a * (a[t] - a[t - 1]) + sh_b * (b[t] - b[t - 1])

        signal_z = z[t - 1]   # everything below is decided on yesterday's close
        last_day = t == n - 1

        # ── Exit logic (executes at close t) ─────────────────────────────
        if position != FLAT:
            reason = None
            if np.isnan(signal_z) or abs(signal_z) < exit_threshold:
                reason = "signal"
            elif abs(signal_z) > stop:
                reason = "stop"
            elif time_stop is not None and prior_days + t - entry_idx >= time_stop:
                reason = "time"
            elif last_day and not carry_out:
                reason = "period_end"

            if reason is not None:
                daily_pnl[t] -= cost_bps * (abs(sh_a) * a[t] + abs(sh_b) * b[t])
                trades.append(Trade(
                    entry_date   = entry_date,
                    exit_date    = dates[t],
                    direction    = position,
                    entry_z      = entry_z,
                    exit_z       = signal_z,
                    pnl          = prior_pnl + float(daily_pnl[entry_idx:t + 1].sum()),
                    exit_reason  = reason,
                    hedge_ratio  = entry_beta,
                    gross        = entry_gross,
                    holding_days = prior_days + t - entry_idx,
                ))
                stopped_out = reason in ("stop", "time")
                position, sh_a, sh_b, entry_idx = FLAT, 0.0, 0.0, None
                prior_days, prior_pnl = 0, 0.0
            continue   # no same-day flip; re-entry is considered tomorrow

        # ── Entry logic (executes at close t) ────────────────────────────
        if last_day or t >= no_entry_from or np.isnan(signal_z):
            continue
        if stopped_out:
            if abs(signal_z) > entry_threshold:
                continue
            stopped_out = False

        gross, hedge = sz[t - 1], beta[t - 1]
        if not (entry_threshold < abs(signal_z) <= stop):
            continue
        if not (np.isfinite(gross) and gross > 0 and np.isfinite(hedge)):
            continue

        # z > 0: spread rich -> short spread (short A, long B); z < 0: the reverse
        position = SHORT_SPREAD if signal_z > 0 else LONG_SPREAD
        notional_a = position * gross / (1.0 + abs(hedge))
        notional_b = -position * gross * hedge / (1.0 + abs(hedge))
        sh_a, sh_b = notional_a / a[t], notional_b / b[t]

        daily_pnl[t] -= cost_bps * (abs(notional_a) + abs(notional_b))
        entry_idx, entry_z     = t, signal_z
        entry_date = dates[t]
        entry_beta, entry_gross = hedge, gross

    daily_pnl_series = pd.Series(daily_pnl, index=dates, name="daily_pnl")
    equity_curve     = daily_pnl_series.cumsum().rename("equity")

    open_position = None
    if position != FLAT:       # only reachable with carry_out=True
        open_position = OpenPosition(
            direction=position, sh_a=sh_a, sh_b=sh_b, entry_date=entry_date,
            entry_z=entry_z, hedge_ratio=entry_beta, gross=entry_gross,
            held_days=prior_days + n - 1 - entry_idx,
            pnl=prior_pnl + float(daily_pnl[entry_idx:].sum()),
        )

    return BacktestResults(
        equity_curve  = equity_curve,
        trades        = trades,
        daily_pnl     = daily_pnl_series,
        open_position = open_position,
    )
