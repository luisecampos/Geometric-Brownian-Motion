"""
tests/test_selection.py
-----------------------
Checks for the stricter pair-selection methods in signals/walkforward.py.

Run with `python -m pytest tests` or just `python tests/test_selection.py`.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signals.walkforward import PairPeriod, distance_schedule, persistent_pairs  # noqa: E402


def _pp(a, b, start):
    return PairPeriod(a, b, 0.05, pd.Timestamp(start), pd.Timestamp(start))


def test_persistent_needs_every_previous_window():
    schedule = [
        _pp("A", "B", "2020-01-01"), _pp("C", "D", "2020-01-01"),
        _pp("A", "B", "2020-04-01"), _pp("E", "F", "2020-04-01"),
        _pp("A", "B", "2020-07-01"), _pp("C", "D", "2020-07-01"), _pp("E", "F", "2020-07-01"),
    ]
    two = [(p.ticker_a, p.period_start.month) for p in persistent_pairs(schedule, 2)]
    assert two == [("A", 4), ("A", 7), ("E", 7)]       # C/D has a gap in April
    three = [(p.ticker_a, p.period_start.month) for p in persistent_pairs(schedule, 3)]
    assert three == [("A", 7)]


def test_persistent_first_windows_are_empty():
    schedule = [_pp("A", "B", "2020-01-01"), _pp("A", "B", "2020-04-01")]
    assert all(p.period_start.month == 4 for p in persistent_pairs(schedule, 2))
    assert persistent_pairs(schedule, 3) == []


def _prices(n=150, seed=2):
    """A/B share a mean-reverting gap; C wanders away; D is in another sector."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    common = np.cumsum(rng.normal(0, 0.01, n))
    gap = np.zeros(n)
    for t in range(1, n):
        gap[t] = 0.8 * gap[t - 1] + rng.normal(0, 0.004)
    a, b = common + gap, common - gap
    c = common + np.cumsum(rng.normal(0.002, 0.01, n))
    d = common + np.cumsum(rng.normal(0, 0.001, n))
    return pd.DataFrame(np.exp(np.column_stack([a, b, c, d])) * 100, index=idx, columns=list("ABCD"))


SECTORS = {"A": "x", "B": "x", "C": "x", "D": "y"}


def test_distance_picks_the_closest_same_sector_pair():
    p = _prices()
    s = distance_schedule(p, list(p.columns), 100, 25, top_n=1, min_half_life=0.1,
                          max_half_life=1e9, sectors=SECTORS)
    assert len(s) == 2 and all((x.ticker_a, x.ticker_b) == ("A", "B") for x in s)
    assert s[0].period_start == p.index[100] and s[0].period_end == p.index[124]
    assert s[0].train_start == p.index[0]


def test_distance_respects_half_life_filter():
    p = _prices()
    # no half-life is in this range, so nothing can be selected
    assert distance_schedule(p, list(p.columns), 100, 25, top_n=1, min_half_life=1e8,
                             max_half_life=1e9, sectors=SECTORS) == []


def test_distance_uses_no_future_data():
    p = _prices()
    kw = dict(top_n=2, min_half_life=0.1, max_half_life=1e9, sectors=SECTORS)
    base = distance_schedule(p, list(p.columns), 100, 25, **kw)
    q = p.copy()
    q.iloc[100:, 0] *= 3.0                               # shock A from the first trading day on
    pert = distance_schedule(q, list(q.columns), 100, 25, **kw)
    first = [x for x in base if x.period_start == p.index[100]]
    same = [x for x in pert if x.period_start == p.index[100]]
    assert [(x.ticker_a, x.ticker_b, x.half_life) for x in first] == [(x.ticker_a, x.ticker_b, x.half_life) for x in same]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS  {name}")
