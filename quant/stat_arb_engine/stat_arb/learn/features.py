"""
learn/features.py
-----------------
Features for the meta-labeler, computed for one candidate signal AS OF its
decision date — the close the engine decides on (it executes at the next
close). Every series is sliced with `.loc[:decision_date]` before anything
is computed, so a feature cannot see data from after the decision even if a
caller passes the full history. tests/test_learn.py checks this by
perturbing every price after the decision date and recomputing.

Feature groups:
  Signal       abs_z, direction
  Pair quality coint_pvalue, adf_stat, half_life   (selection window only,
               i.e. strictly before period_start)
  Pair state   hurst (of the Kalman spread), beta_std (hedge-ratio
               stability), spread_vol_pct (current spread vol vs its past
               year), leg_corr (rolling correlation of the legs' returns)
  Market       spy_vol20 (annualised 20d realised vol), spy_trend (SPY vs
               its 50d moving average)
  Pair meta    days_qualified (trading days since the pair first passed the
               walk-forward screen), same_sector
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

from signals.cointegration import spread_half_life


FEATURES = [
    "abs_z", "direction",
    "coint_pvalue", "adf_stat", "half_life",
    "hurst", "beta_std", "spread_vol_pct", "leg_corr",
    "spy_vol20", "spy_trend",
    "days_qualified", "same_sector",
]

HURST_WINDOW    = 120   # days of spread used for the Hurst estimate
BETA_STD_WINDOW = 20
VOL_WINDOW      = 20
VOL_PCT_WINDOW  = 252
CORR_WINDOW     = 60


def hurst_exponent(x: np.ndarray, max_lag: int = 20) -> float:
    """
    Hurst exponent from the scaling of lagged differences:
    std(x[t+k] - x[t]) ~ k^H. H < 0.5 = mean-reverting, 0.5 = random walk,
    H > 0.5 = trending.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    lags = np.arange(2, min(max_lag, len(x) // 4) + 1)
    if len(lags) < 3:
        return float("nan")
    tau = np.array([np.std(x[k:] - x[:-k]) for k in lags])
    ok = tau > 0
    if ok.sum() < 3:
        return float("nan")
    return float(np.polyfit(np.log(lags[ok]), np.log(tau[ok]), 1)[0])


def selection_window_features(train_a: pd.Series, train_b: pd.Series) -> dict:
    """
    Pair-quality features from the walk-forward selection window (log prices
    strictly before period_start): Engle-Granger ADF stat and p-value, and
    OU half-life of the OLS spread.
    """
    la, lb = np.log(train_a.astype(float)), np.log(train_b.astype(float))
    stat, pval, _ = coint(la.values, lb.values)
    return {
        "coint_pvalue": float(pval),
        "adf_stat"    : float(stat),
        "half_life"   : float(spread_half_life(la, lb)),
    }


def market_features(spy: pd.Series | None, decision_date: pd.Timestamp) -> dict:
    """SPY regime as of decision_date: 20d realised vol (annualised) and trend."""
    if spy is None or len(spy) == 0:
        return {"spy_vol20": float("nan"), "spy_trend": float("nan")}
    s = spy.loc[:decision_date].dropna()
    r = np.log(s).diff().dropna()
    vol = float(r.iloc[-VOL_WINDOW:].std() * np.sqrt(252)) if len(r) >= VOL_WINDOW else float("nan")
    trend = float(s.iloc[-1] / s.iloc[-50:].mean() - 1.0) if len(s) >= 50 else float("nan")
    return {"spy_vol20": vol, "spy_trend": trend}


def signal_features(
    series_a      : pd.Series,
    series_b      : pd.Series,
    hedge_ratio   : pd.Series,
    spread        : pd.Series,
    zscores       : pd.Series,
    decision_date : pd.Timestamp,
) -> dict:
    """Signal and pair-state features from data up to and including decision_date."""
    a  = series_a.loc[:decision_date]
    b  = series_b.loc[:decision_date]
    hr = hedge_ratio.loc[:decision_date]
    sp = spread.loc[:decision_date]
    z  = float(zscores.loc[:decision_date].iloc[-1])

    ds = sp.diff()
    roll_vol = ds.rolling(VOL_WINDOW).std().iloc[-VOL_PCT_WINDOW:].dropna()
    vol_pct = float((roll_vol <= roll_vol.iloc[-1]).mean()) if len(roll_vol) >= 20 else float("nan")

    ra, rb = np.log(a).diff().iloc[-CORR_WINDOW:], np.log(b).diff().iloc[-CORR_WINDOW:]
    return {
        "abs_z"         : abs(z),
        # +1 = long spread (z < 0), -1 = short spread, matching engine.py
        "direction"     : 1.0 if z < 0 else -1.0,
        "hurst"         : hurst_exponent(sp.iloc[-HURST_WINDOW:].values),
        "beta_std"      : float(hr.iloc[-BETA_STD_WINDOW:].std()),
        "spread_vol_pct": vol_pct,
        "leg_corr"      : float(ra.corr(rb)),
    }


def compute_event_features(
    prep,
    zscores        : pd.Series,
    decision_date  : pd.Timestamp,
    static         : dict,
    spy            : pd.Series | None,
    first_qualified: pd.Timestamp,
    calendar       : pd.DatetimeIndex,
    sectors        : dict | None = None,
) -> dict:
    """
    All features for one candidate signal of a PreparedPair, as of
    decision_date. `static` is selection_window_features() for this
    pair-period; `first_qualified` is the earliest period_start at which this
    pair passed the screen (known at period_start, so it is causal).
    """
    feats = dict(static)
    feats.update(signal_features(
        prep.series_a, prep.series_b, prep.kf.hedge_ratio, prep.kf.spread,
        zscores, decision_date,
    ))
    feats.update(market_features(spy, decision_date))
    days = calendar.get_loc(decision_date) - calendar.get_loc(first_qualified)
    feats["days_qualified"] = float(max(0, days))
    sectors = sectors or {}
    sa, sb = sectors.get(prep.pp.ticker_a), sectors.get(prep.pp.ticker_b)
    feats["same_sector"] = float(sa is not None and sa == sb)
    return {k: feats[k] for k in FEATURES}
