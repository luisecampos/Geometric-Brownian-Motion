"""
learn/evaluate.py
-----------------
Honest scoring for the learning layer.

Deflated Sharpe ratio (Bailey & López de Prado, 2014):
  If you try N configurations and report the best, its Sharpe is biased up
  even if none has any skill. The DSR is the probability that the true
  Sharpe is above zero AFTER accounting for:
    - the expected maximum Sharpe of N skill-less trials,
        SR0 = sqrt(V[SR]) * ((1 - g) * Z(1 - 1/N) + g * Z(1 - 1/(N e)))
      with V[SR] the variance of Sharpe across the trials, g the
      Euler-Mascheroni constant, and Z the standard normal quantile;
    - the track-record length T, and the skew and kurtosis of the returns
      (fat tails make a given Sharpe less certain).
  DSR = Phi( (SR - SR0) * sqrt(T - 1) / sqrt(1 - skew*SR + (kurt - 1)/4 * SR^2) )
  All Sharpes are per-day here. DSR > 0.95 is the usual bar.

Classifier quality is measured only on out-of-sample predictions: each
period's candidates are scored by the model trained before that period.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import brier_score_loss, roc_auc_score

EULER_GAMMA = 0.5772156649015329


def daily_sharpe(x: pd.Series) -> float:
    sd = x.std()
    return float(x.mean() / sd) if np.isfinite(sd) and sd > 0 else 0.0


def deflated_sharpe(daily_pnl: pd.Series, trial_sharpes: list[float], n_trials: int | None = None) -> dict:
    """
    daily_pnl     : the strategy being judged (all days, flat ones included)
    trial_sharpes : per-day Sharpe of every configuration tried (including
                    this one), used for V[SR]
    n_trials      : total configurations tried, if more than len(trial_sharpes)
                    (e.g. offline experiments whose returns weren't kept)
    """
    x = daily_pnl.values.astype(float)
    T = len(x)
    sr = daily_sharpe(daily_pnl)
    N = max(n_trials or len(trial_sharpes), 1)
    var_sr = float(np.var(trial_sharpes, ddof=1)) if len(trial_sharpes) > 1 else 0.0
    if N > 1:
        sr0 = np.sqrt(var_sr) * (
            (1 - EULER_GAMMA) * stats.norm.ppf(1 - 1.0 / N)
            + EULER_GAMMA * stats.norm.ppf(1 - 1.0 / (N * np.e))
        )
    else:
        sr0 = 0.0
    skew = float(stats.skew(x)) if T > 2 else 0.0
    kurt = float(stats.kurtosis(x, fisher=False)) if T > 3 else 3.0
    denom = 1 - skew * sr + (kurt - 1) / 4 * sr ** 2
    dsr = float(stats.norm.cdf((sr - sr0) * np.sqrt(T - 1) / np.sqrt(max(denom, 1e-12))))
    return {
        "sharpe_ann": sr * np.sqrt(252),
        "sr0_ann"   : sr0 * np.sqrt(252),
        "dsr"       : dsr,
        "n_trials"  : N,
    }


def classification_stats(y, p, groups=None) -> dict:
    """
    Pooled AUC / Brier, plus (if `groups` = each candidate's period is given)
    the n-weighted mean AUC WITHIN periods. Pooled AUC mixes two things:
    ranking candidates against each other, and getting each period's overall
    win rate right. Within-period AUC isolates the ranking. A model can rank
    well inside each quarter and still be useless for a fixed p threshold if
    the win-rate level shifts from quarter to quarter.
    """
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    ok = np.isfinite(p)
    g = np.asarray(groups)[ok] if groups is not None else None
    y, p = y[ok], p[ok]
    if len(y) == 0 or len(np.unique(y)) < 2:
        return {"n": int(len(y)), "auc": np.nan, "auc_within": np.nan,
                "brier": np.nan, "brier_base": np.nan, "base_rate": np.nan}
    auc_within = np.nan
    if g is not None:
        aucs, ns = [], []
        for k in np.unique(g):
            m = g == k
            if m.sum() >= 5 and len(np.unique(y[m])) == 2:
                aucs.append(roc_auc_score(y[m], p[m]))
                ns.append(m.sum())
        if aucs:
            auc_within = float(np.average(aucs, weights=ns))
    return {
        "n"         : int(len(y)),
        "auc"       : float(roc_auc_score(y, p)),
        "auc_within": auc_within,
        "brier"     : float(brier_score_loss(y, p)),
        # Brier of always predicting the test base rate: the bar to beat
        "brier_base": float(np.mean((y - y.mean()) ** 2)),
        "base_rate" : float(y.mean()),
    }


def oos_permutation_importance(records: list, features: list[str], n_repeats: int = 30, seed: int = 0) -> pd.Series:
    """
    Drop in pooled out-of-sample AUC when one feature is shuffled within
    each period's test set, scored by that period's own model. Positive =
    the model relied on the feature AND it helped out of sample. Near zero or
    negative = noise.
    """
    if not records:
        return pd.Series(dtype=float)
    rng = np.random.default_rng(seed)
    y_all = np.concatenate([r[3].values for r in records])
    if len(np.unique(y_all)) < 2:
        return pd.Series(np.nan, index=features)
    base = roc_auc_score(y_all, np.concatenate([r[1].predict_proba(r[2]) for r in records]))
    out = {}
    for f in features:
        drops = []
        for _ in range(n_repeats):
            preds = []
            for _, model, X, _y in records:
                Xp = X.copy()
                Xp[f] = rng.permutation(Xp[f].values)
                preds.append(model.predict_proba(Xp))
            drops.append(base - roc_auc_score(y_all, np.concatenate(preds)))
        out[f] = float(np.mean(drops))
    return pd.Series(out).sort_values(ascending=False)


def calibration_table(y, p, bins: int = 5) -> pd.DataFrame:
    """Out-of-sample predicted vs realised win rate, by probability quantile."""
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p, dtype=float)}).dropna()
    if len(df) < bins:
        return pd.DataFrame(columns=["p_mean", "win_rate", "n"])
    df["bin"] = pd.qcut(df["p"], bins, duplicates="drop")
    g = df.groupby("bin", observed=True)
    return pd.DataFrame({"p_mean": g["p"].mean(), "win_rate": g["y"].mean(), "n": g["y"].size()})
