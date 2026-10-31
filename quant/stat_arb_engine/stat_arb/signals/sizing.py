"""
signals/sizing.py
------------------
Volatility-targeted position sizing on a real capital base.

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

Units (this is what changed):
  size is GROSS NOTIONAL AS A FRACTION OF CAPITAL. size=0.8 means $0.80 of
  combined long+short exposure per $1 of book capital, split across the legs
  by the hedge ratio:

      A leg = size * 1    / (1 + |beta|)
      B leg = size * beta / (1 + |beta|)    (opposite side)

  beta is the log-space Kalman hedge ratio, i.e. a dollar ratio, so this
  position's return is exactly the spread's return. When beta=1 the legs are
  dollar-neutral; otherwise they're hedge-neutral (dollar-neutrality would
  stop the PnL tracking the spread the signal is built on).

  The old version divided a $-target by the std of *raw spread changes* (a
  share-price quantity) — with the broken Kalman spread that sat at the 3.0
  cap on 100% of days, so "vol targeting" was really "always max size".
"""

import numpy as np
import pandas as pd


def hedged_returns(
    series_a     : pd.Series,
    series_b     : pd.Series,
    hedge_ratios : pd.Series,
) -> pd.Series:
    """
    Daily return of $1 gross of the hedged spread (long A / short beta*B),
    using the hedge known at the PREVIOUS close — the one you'd actually be
    holding over day t.
    """
    beta_prev = hedge_ratios.shift(1)
    r_a = series_a.pct_change()
    r_b = series_b.pct_change()
    ret = (r_a - beta_prev * r_b) / (1.0 + beta_prev.abs())
    ret.name = "spread_return"
    return ret


def vol_target_size(
    series_a         : pd.Series,
    series_b         : pd.Series,
    hedge_ratios     : pd.Series,
    target_daily_vol : float = 0.01,
    lookback         : int   = 20,
    max_leverage     : float = 3.0,
) -> pd.Series:
    """
    Gross notional (fraction of capital) that scales the pair's daily return
    to `target_daily_vol`.

    size[t] = target_daily_vol / std(hedged_returns[t-lookback+1 .. t])   (capped)

    size[t] uses returns through the close of day t, so a trade decided at the
    close of t-1 should use size[t-1] — the engine does exactly that.

    Args:
        series_a/b       : price series for each leg
        hedge_ratios     : log-space (dollar) hedge ratio, e.g. from the Kalman filter
        target_daily_vol : desired daily std of the pair's PnL, as a fraction
                           of book capital (default 1%)
        lookback         : trailing window (days) used to estimate realized vol
        max_leverage     : cap on gross notional / capital for one pair

    Returns:
        size series, same index as the inputs. NaN until `lookback` returns
        are available — callers should treat NaN as "don't trade yet".
    """
    realized_vol = hedged_returns(series_a, series_b, hedge_ratios).rolling(window=lookback).std()

    size = target_daily_vol / realized_vol.replace(0.0, np.nan)
    size = size.clip(upper=max_leverage)
    size.name = "size"
    return size
