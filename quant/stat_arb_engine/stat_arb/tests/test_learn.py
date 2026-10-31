"""
tests/test_learn.py
-------------------
No-look-ahead checks for the self-learning layer. Each one perturbs every
price after some date and asserts that nothing decided before that date
changes: features, labels, the training set, and (end to end) every
learned variant's probabilities, size multipliers and PnL.

Run with `python -m pytest tests` or just `python tests/test_learn.py`.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.portfolio import prepare_pair_periods, run_portfolio_backtest  # noqa: E402
from learn.features import FEATURES, compute_event_features, selection_window_features  # noqa: E402
from learn.metalabel import training_mask  # noqa: E402
from learn.pipeline import LearnConfig, run_learning_layer  # noqa: E402
from learn.trade_log import find_candidate_signals, triple_barrier  # noqa: E402
from signals.walkforward import PairPeriod, generate_pair_schedule  # noqa: E402


def _synthetic(n=900, n_pairs=3, seed=11):
    """n_pairs cointegrated pairs (OU spread, half-life ~10d) on a common market factor."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    market = np.cumsum(rng.normal(0.0003, 0.01, n))
    cols = {}
    for k in range(n_pairs):
        base = np.cumsum(rng.normal(0, 0.012, n)) + 0.6 * market
        s = np.zeros(n)
        for t in range(1, n):
            s[t] = 0.93 * s[t - 1] + rng.normal(0, 0.012)
        cols[f"A{k}"] = 40 * np.exp(base + s)
        cols[f"B{k}"] = 60 * np.exp(base)
    prices = pd.DataFrame(cols, index=idx)
    spy = pd.Series(300 * np.exp(market), index=idx, name="SPY")
    return prices, spy


def _perturb_after(df_or_s, cut, seed=99):
    """Replace every value strictly after `cut` with a different random path."""
    rng = np.random.default_rng(seed)
    out = df_or_s.copy()
    after = out.index > cut
    shape = out.loc[after].shape
    out.loc[after] = out.loc[after].values * np.exp(np.cumsum(rng.normal(0, 0.03, shape), axis=0))
    return out


def _same(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    return x.shape == y.shape and np.allclose(x, y, equal_nan=True)


def _one_pair_period(prices):
    idx = prices.index
    return PairPeriod("A0", "B0", 0.01, period_start=idx[300], period_end=idx[362],
                      train_start=idx[48], half_life=10.0)


def test_features_use_no_future_data():
    prices, spy = _synthetic()
    pp = _one_pair_period(prices)
    prep = prepare_pair_periods(prices, [pp], 0.0025)[0]
    z = prep.zscores(60)
    static = selection_window_features(prep.series_a.iloc[:prep.train_size],
                                       prep.series_b.iloc[:prep.train_size])
    checked = 0
    for d in range(prep.first, len(z) - 2, 7):
        dd = z.index[d]
        f0 = compute_event_features(prep, z, dd, static, spy, pp.period_start, prices.index)

        p2, s2 = _perturb_after(prices, dd), _perturb_after(spy, dd)
        prep2 = prepare_pair_periods(p2, [pp], 0.0025)[0]
        z2 = prep2.zscores(60)
        static2 = selection_window_features(prep2.series_a.iloc[:prep2.train_size],
                                            prep2.series_b.iloc[:prep2.train_size])
        f1 = compute_event_features(prep2, z2, dd, static2, s2, pp.period_start, prices.index)
        assert _same([f0[k] for k in FEATURES], [f1[k] for k in FEATURES]), dd
        checked += 1
    assert checked >= 5


def test_labels_use_no_data_after_t1():
    prices, _ = _synthetic()
    pp = _one_pair_period(prices)
    prep = prepare_pair_periods(prices, [pp], 0.0025)[0]
    z = prep.zscores(60)
    cands = find_candidate_signals(z, prep.first, 1.5, 3.5)
    assert cands, "synthetic pair should produce candidate signals"
    for d in cands:
        lab = triple_barrier(prep.series_a, prep.series_b, prep.kf.hedge_ratio, z, d, 0.5, 3.5, 20)
        assert lab["t1"] > z.index[d]
        prep2 = prepare_pair_periods(_perturb_after(prices, lab["t1"]), [pp], 0.0025)[0]
        z2 = prep2.zscores(60)
        lab2 = triple_barrier(prep2.series_a, prep2.series_b, prep2.kf.hedge_ratio, z2, d, 0.5, 3.5, 20)
        assert lab == lab2


def test_training_set_purges_and_embargoes():
    cal = pd.bdate_range("2020-01-01", periods=60)
    T = cal[40]
    cands = pd.DataFrame({
        "decision_date": [cal[5], cal[30], cal[33], cal[34], cal[36], cal[38]],
        "t1"           : [cal[10], cal[34], cal[35], cal[39], cal[45], cal[41]],
    })
    m0 = training_mask(cands, T, cal, embargo_days=0)
    assert m0.tolist() == [True, True, True, True, False, False]   # purge: t1 >= T
    m5 = training_mask(cands, T, cal, embargo_days=5)
    assert m5.tolist() == [True, True, False, False, False, False]  # embargo: t1 >= cal[35]
    assert (cands.loc[m5, "t1"] < T).all()


def test_end_to_end_no_lookahead():
    """
    Perturb every price after `cut`. For every learned variant, nothing
    decided before `cut` may change: candidate features, OOS probabilities,
    size multipliers, tuned parameters, and daily PnL before `cut`.
    """
    prices, spy = _synthetic()

    def run(px, sp):
        sched = generate_pair_schedule(px, list(px.columns), 252, 63, 0.10, None, 3, 60)
        prep = prepare_pair_periods(px, sched, 0.0025)
        base = run_portfolio_backtest(px, sched, entry_threshold=1.5, prepared=prep)
        cfg = LearnConfig(entry_threshold=1.5, min_train=20, embargo_days=5)
        return run_learning_layer(px, sched, prep, base, spy=sp, cfg=cfg, verbose=False)

    cut = prices.index[700]
    r0 = run(prices, spy)
    r1 = run(_perturb_after(prices, cut), _perturb_after(spy, cut))

    c0 = r0.candidates[r0.candidates["decision_date"] < cut].reset_index(drop=True)
    c1 = r1.candidates[r1.candidates["decision_date"] < cut].reset_index(drop=True)
    assert len(c0) == len(c1) and len(c0) > 0
    p_cols = [c for c in c0.columns if c.startswith(("p_", "mult_"))]
    assert c0["p_logit"].notna().sum() > 0, "no model was trained before the cut; test is vacuous"
    for col in FEATURES + p_cols:
        assert _same(c0[col], c1[col]), col
    # Labels may legitimately differ only where the outcome resolved after the cut
    resolved = c0["t1"] < cut
    assert _same(c0.loc[resolved, "ret"], c1.loc[resolved, "ret"])

    assert {T: k for T, k in r0.tuned_choice.items() if T <= cut} == \
           {T: k for T, k in r1.tuned_choice.items() if T <= cut}
    for name in r0.variants:
        d0 = r0.variants[name].daily_pnl.loc[:cut].iloc[:-1]
        d1 = r1.variants[name].daily_pnl.loc[:cut].iloc[:-1]
        assert _same(d0, d1), name


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS  {name}")
