"""
tests/test_engine.py
--------------------
Accounting and timing checks for backtest/engine.py — the properties a
learning layer trained on Trade.pnl silently depends on.

Run with `python -m pytest tests` or just `python tests/test_engine.py`.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.engine import run_backtest  # noqa: E402


def _inputs(n=300, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-01", periods=n)
    b = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), idx)
    a = pd.Series(50 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), idx)
    beta = pd.Series(1.0 + 0.1 * rng.normal(size=n), idx)
    z = pd.Series(np.sin(np.arange(n) / 6.0) * 3.0, idx)   # swings through +-3
    return a, b, beta, z


def test_trade_pnl_sums_to_total():
    """Every PnL day belongs to exactly one trade, entry+exit costs included."""
    a, b, beta, z = _inputs()
    res = run_backtest(a, b, beta, z, stop_threshold=None)
    assert len(res.trades) > 3
    assert np.isclose(sum(t.pnl for t in res.trades), res.daily_pnl.sum())


def test_open_position_force_closed():
    a, b, beta, z = _inputs()
    z = pd.Series(2.5, index=z.index)   # enters on day 1 and never reverts
    res = run_backtest(a, b, beta, z, stop_threshold=None)
    assert len(res.trades) == 1
    t = res.trades[0]
    assert t.exit_reason == "period_end" and t.exit_date == a.index[-1]
    assert np.isclose(t.pnl, res.daily_pnl.sum())


def test_no_lookahead():
    """Changing day-t data must not change PnL on any day before t."""
    a, b, beta, z = _inputs()
    base = run_backtest(a, b, beta, z).daily_pnl

    cut = 150
    a2, b2, beta2, z2 = a.copy(), b.copy(), beta.copy(), z.copy()
    a2.iloc[cut:] *= 1.3
    beta2.iloc[cut:] = 5.0
    z2.iloc[cut:] = -z2.iloc[cut:]
    pert = run_backtest(a2, b2, beta2, z2).daily_pnl

    assert np.allclose(base.iloc[:cut], pert.iloc[:cut])


def test_stop_loss_fires_and_blocks_reentry():
    a, b, beta, _ = _inputs(n=40)
    zv = np.full(40, np.nan)
    zv[5:] = [2.2, 2.8, 3.2, 3.8, 3.9, 3.0, 2.5, 2.1, 1.0] + [0.2] * 26
    z = pd.Series(zv, index=a.index)
    res = run_backtest(a, b, beta, z)
    assert res.trades[0].exit_reason == "stop"
    # z stayed above the entry band after the stop: no re-entry until it fell back in
    assert len(res.trades) == 1


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS  {name}")
