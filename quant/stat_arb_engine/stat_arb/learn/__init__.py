"""
learn/ — the self-learning layer on top of the walk-forward pairs engine.

  features.py   features for each candidate signal, as of its decision date
  trade_log.py  candidate signals, triple-barrier labels, persistence
  metalabel.py  walk-forward meta-labeler (purged + embargoed retraining)
  tuning.py     nested walk-forward choice of entry / exit / lookback
  allocation.py Thompson-sampling capital allocation across pairs
  evaluate.py   deflated Sharpe, out-of-sample AUC, permutation importance
"""
