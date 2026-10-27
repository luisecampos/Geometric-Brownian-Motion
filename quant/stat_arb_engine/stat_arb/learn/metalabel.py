"""
learn/metalabel.py
------------------
Meta-labeling (López de Prado, AFML ch. 3): the z-score rule stays the
primary model and decides the SIDE of every trade. A secondary classifier
predicts P(trade profitable | features) and decides whether to take it and
how big.

Walk-forward retraining, with no look-ahead:
  At each rebalance date T (a walk-forward period_start), the model is
  refit on an EXPANDING window of past labelled candidates and then used for
  every candidate in the period starting at T. The first of those decisions
  is made at the close of T-1, so a candidate may only be used for training
  if its label was known by then:

    purge   : drop candidates whose outcome window [entry, t1] reaches T
              (t1 >= T). Their label isn't known yet, and using it would
              leak the test period's prices into the model.
    embargo : also drop candidates with t1 in the last `embargo_days`
              before T. Features and labels are serially correlated (the
              same spread excursion, the same market regime), so a label
              that resolved yesterday is nearly a copy of today's test
              point.

  Until enough labelled history exists (min_train candidates and min_class
  of each outcome), the meta-labeler abstains and every signal is taken at
  baseline size, so the early periods are identical to the baseline.

Sizing from the probability (capped Kelly):
  f* = p - (1 - p) / b,  where b = avg win / avg loss in the training set.
  "gate"  : take the trade at baseline size if p >= p_min and f* > 0, else skip
  "kelly" : the same gate, then size = baseline * clip(f* / kelly_unit, 0, max_mult)
            kelly_unit = 0.25 means a quarter-Kelly-sized edge gets the
            baseline vol-target size, and nothing gets more than max_mult x.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from learn.features import FEATURES


def training_mask(
    candidates    : pd.DataFrame,
    rebalance_date: pd.Timestamp,
    calendar      : pd.DatetimeIndex,
    embargo_days  : int = 5,
) -> pd.Series:
    """
    Candidates usable for training a model that will make its first decision
    at the close before `rebalance_date`: label resolved (t1) strictly before
    the embargo cutoff, which is `embargo_days` trading days before T.
    """
    pos = calendar.get_loc(rebalance_date)
    cutoff = calendar[max(0, pos - embargo_days)]
    return (candidates["t1"] < cutoff) & (candidates["decision_date"] < rebalance_date)


class MetaLabeler:
    def __init__(self, kind: str = "logit", random_state: int = 0):
        if kind not in ("logit", "hgb"):
            raise ValueError(f"unknown kind: {kind!r}")
        self.kind = kind
        self.random_state = random_state
        self.model = None
        self.win_loss = float("nan")
        self.base_rate = float("nan")

    def _make(self):
        if self.kind == "logit":
            # Strong L2: ~50-300 noisy labels, 13 features.
            return make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                LogisticRegression(C=0.1, max_iter=2000),
            )
        return HistGradientBoostingClassifier(
            max_depth=2, learning_rate=0.05, max_iter=100,
            min_samples_leaf=20, l2_regularization=1.0,
            random_state=self.random_state,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series, ret: pd.Series) -> "MetaLabeler":
        self.model = self._make().fit(X[FEATURES].values, y.values)
        wins, losses = ret[ret > 0], -ret[ret <= 0]
        if len(wins) and len(losses) and losses.mean() > 0:
            self.win_loss = float(wins.mean() / losses.mean())
        else:
            self.win_loss = 1.0
        self.base_rate = float(y.mean())
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X[FEATURES].values)[:, 1]


def kelly_fraction(p, win_loss):
    return p - (1.0 - p) / win_loss


def size_multiplier(
    p          : np.ndarray,
    win_loss   : np.ndarray,
    mode       : str = "kelly",
    p_min      : float = 0.5,
    kelly_unit : float = 0.25,
    max_mult   : float = 2.0,
) -> np.ndarray:
    """Map P(profitable) to a size multiplier; NaN p (abstain) -> 1.0."""
    p = np.asarray(p, dtype=float)
    f = kelly_fraction(p, np.asarray(win_loss, dtype=float))
    take = (p >= p_min) & (f > 0)
    if mode == "gate":
        mult = take.astype(float)
    elif mode == "kelly":
        mult = np.where(take, np.clip(f / kelly_unit, 0.0, max_mult), 0.0)
    else:
        raise ValueError(f"unknown mode: {mode!r}")
    return np.where(np.isfinite(p), mult, 1.0)


def walk_forward_metalabel(
    candidates   : pd.DataFrame,
    calendar     : pd.DatetimeIndex,
    kind         : str = "logit",
    embargo_days : int = 5,
    min_train    : int = 40,
    min_class    : int = 8,
) -> tuple[pd.DataFrame, list]:
    """
    Retrain at every rebalance date and predict that period's candidates.

    Returns:
      preds   : DataFrame aligned to `candidates` with p (NaN = abstained)
                and win_loss
      records : [(T, model, X_test, y_test)] for out-of-sample evaluation
    """
    preds = pd.DataFrame({"p": np.nan, "win_loss": np.nan}, index=candidates.index)
    records = []
    for T in sorted(candidates["period_start"].unique()):
        T = pd.Timestamp(T)
        test = candidates["period_start"] == T
        train = training_mask(candidates, T, calendar, embargo_days)
        y = candidates.loc[train, "y"]
        if len(y) < min_train or y.sum() < min_class or (len(y) - y.sum()) < min_class:
            continue
        model = MetaLabeler(kind).fit(
            candidates.loc[train], y, candidates.loc[train, "ret"]
        )
        X_test = candidates.loc[test, FEATURES]
        preds.loc[test, "p"] = model.predict_proba(X_test)
        preds.loc[test, "win_loss"] = model.win_loss
        records.append((T, model, X_test, candidates.loc[test, "y"]))
    return preds, records


def multiplier_series(candidates: pd.DataFrame, mult: np.ndarray, prepared: list) -> dict:
    """
    Turn per-candidate multipliers into the {pair label: Series} the
    portfolio backtest takes. The value set on a candidate's decision date
    is carried forward until the next candidate, so a skipped signal stays
    skipped for the rest of its excursion (the engine would otherwise enter
    on the next day of the same excursion).
    """
    out = {}
    df = candidates.assign(_m=mult)
    by_label = {p.label: p for p in prepared}
    for label, g in df.groupby("pair_label"):
        prep = by_label[label]
        s = pd.Series(np.nan, index=prep.series_a.index)
        s.loc[g["decision_date"].values] = g["_m"].values
        out[label] = s.ffill().fillna(1.0)
    return out
