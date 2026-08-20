"""
signals/zscore.py
-----------------
Convert the spread into a z-score — the core trading signal.

The z-score tells you how many standard deviations the current spread is
from its recent mean. This is what drives entries and exits:

  z > +2.0  → spread is too wide → short A, long B  (expect convergence)
  z < -2.0  → spread is too narrow → long A, short B
  |z| < 0.5 → spread has mean-reverted → exit the trade
"""

import pandas as pd


def compute_zscore(spreads: pd.Series, window: int = 60) -> pd.Series:
    """
    Compute rolling z-score of the spread.

    z[t] = (spread[t] - mean(spread[t-window:t])) / std(spread[t-window:t])

    Uses a rolling window rather than expanding so the signal adapts to
    changing volatility regimes (important — vol is not constant).

    Args:
        spreads : raw spread series from Kalman filter
        window  : lookback in trading days (default 60 ~ 3 months)

    Returns:
        z-score series, NaN for the first `window` rows (insufficient history)
    """
    roll_mean = spreads.rolling(window=window).mean()
    roll_std  = spreads.rolling(window=window).std()

    zscore = (spreads - roll_mean) / roll_std
    zscore.name = "zscore"
    return zscore
