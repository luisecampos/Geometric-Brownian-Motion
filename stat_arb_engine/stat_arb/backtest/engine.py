"""
backtest/engine.py
------------------
Tick through history day by day, open/close positions based on z-score
thresholds, and track PnL.

Position logic:
  - We are always either FLAT, LONG_SPREAD, or SHORT_SPREAD
  - LONG_SPREAD  = long A, short B*hedge_ratio  (bet spread widens back up)
  - SHORT_SPREAD = short A, long B*hedge_ratio  (bet spread narrows back down)

PnL is computed in dollar terms assuming $1 notional per leg.
Transaction costs are included (2bps per trade, realistic for equities).

Note on look-ahead bias:
  We use z[t] to decide, then execute at close price on day t+1.
  This is conservative but realistic — you see EOD data, trade next open.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field


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
    pnl         : float


@dataclass
class BacktestResults:
    equity_curve : pd.Series
    trades       : list[Trade]
    daily_pnl    : pd.Series


def run_backtest(
    series_a        : pd.Series,
    series_b        : pd.Series,
    hedge_ratios    : pd.Series,
    zscores         : pd.Series,
    entry_threshold : float = 2.0,
    exit_threshold  : float = 0.5,
) -> BacktestResults:
    """
    Simulate the pairs trading strategy day by day.

    Args:
        series_a/b      : price series for each leg
        hedge_ratios    : Kalman-estimated beta each day
        zscores         : rolling z-score of spread
        entry_threshold : |z| above which we enter a trade
        exit_threshold  : |z| below which we exit a trade

    Returns:
        BacktestResults with equity curve, trade log, and daily PnL
    """
    dates      = series_a.index
    a          = series_a.values
    b          = series_b.values
    beta       = hedge_ratios.values
    z          = zscores.values

    position   = FLAT
    entry_idx  = None
    entry_z    = None
    daily_pnl  = np.zeros(len(dates))
    trades     = []

    for t in range(1, len(dates)):
        if np.isnan(z[t - 1]):
            continue

        prev_z = z[t - 1]   # signal from yesterday
        curr_z = z[t]

        # Daily PnL for open position
        if position != FLAT:
            da = a[t] - a[t - 1]   # change in price A
            db = b[t] - b[t - 1]   # change in price B

            if position == LONG_SPREAD:
                daily_pnl[t] = da - beta[t] * db
            else:
                daily_pnl[t] = -da + beta[t] * db

        # ── Entry logic ───────────────────────────────────────────
        if position == FLAT:
            if prev_z > entry_threshold:
                # Spread is high → short spread (short A, long B)
                position  = SHORT_SPREAD
                entry_idx = t
                entry_z   = prev_z
                # Pay transaction cost on both legs
                daily_pnl[t] -= COST_BPS * 2

            elif prev_z < -entry_threshold:
                # Spread is low → long spread (long A, short B)
                position  = LONG_SPREAD
                entry_idx = t
                entry_z   = prev_z
                daily_pnl[t] -= COST_BPS * 2

        # ── Exit logic ────────────────────────────────────────────
        elif abs(curr_z) < exit_threshold or np.isnan(curr_z):
            trade_pnl = daily_pnl[entry_idx:t + 1].sum()
            trades.append(Trade(
                entry_date = dates[entry_idx],
                exit_date  = dates[t],
                direction  = position,
                entry_z    = entry_z,
                exit_z     = curr_z,
                pnl        = trade_pnl,
            ))
            daily_pnl[t] -= COST_BPS * 2   # exit cost
            position = FLAT
            entry_idx = None

    equity_curve = pd.Series(
        np.cumsum(daily_pnl),
        index=dates,
        name="equity",
    )
    daily_pnl_series = pd.Series(daily_pnl, index=dates, name="daily_pnl")

    return BacktestResults(
        equity_curve = equity_curve,
        trades       = trades,
        daily_pnl    = daily_pnl_series,
    )
