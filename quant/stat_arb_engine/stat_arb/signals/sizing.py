"""
signals/sizing.py
------------------
Volatility-targeted position sizing.

The original engine traded a flat $1 notional per leg regardless of how
volatile the spread was — a quiet, low-vol pair and a wild, high-vol pair got
identical size. That means most of the portfolio's risk ends up concentrated
in whichever pair happens to be loudest, which isn't a real risk decision,
it's an accident.

Vol targeting instead scales notional so every pair contributes roughly the
same amount of expected daily risk: size is inversely proportional to the
pair's own trailing realized volatility. Quiet pairs get sized up, noisy
pairs get sized down, capped so a suddenly-dead-quiet spread doesn't imply
absurd leverage.
"""

import numpy as np
import pandas as pd


def vol_target_size(
    spread        : pd.Series,
    target_daily_vol : float = 0.01,
    lookback      : int   = 20,
    max_leverage  : float = 3.0,
) -> pd.Series:
    """
    Compute a per-day size multiplier that scales the spread's $1-notional
    daily P&L to a target daily volatility.

    size[t] = target_daily_vol / realized_vol[t-lookback:t]   (capped)

    Args:
        spread           : raw spread series (e.g. from the Kalman filter)
        target_daily_vol : desired daily standard deviation of PnL, in the
                            same units as spread changes (default 1%)
        lookback          : trailing window (days) used to estimate realized vol
        max_leverage      : hard cap on the size multiplier, so a spread that's
                            gone unusually quiet doesn't imply unbounded size

    Returns:
        size multiplier series, same index as spread. NaN for the first
        `lookback` rows (insufficient history) — callers should treat NaN
        size as "don't trade yet", same as the zscore warm-up period.
    """
    spread_ret = spread.diff()
    realized_vol = spread_ret.rolling(window=lookback).std()

    size = target_daily_vol / realized_vol
    size = size.clip(upper=max_leverage)
    size.name = "size"
    return size
