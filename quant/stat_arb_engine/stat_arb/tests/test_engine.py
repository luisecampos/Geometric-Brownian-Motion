"""
tests/test_engine.py
--------------------
Accounting and timing checks for backtest/engine.py — the properties a
learning layer trained on Trade.pnl silently depends on.

Run with `python -m pytest tests` or just `python tests/test_engine.py`.
"""

import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.engine import run_backtest  # noqa: E402
from backtest.portfolio import ExitRules, run_portfolio_backtest  # noqa: E402
from signals.walkforward import PairPeriod  # noqa: E402


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


def test_time_stop_fires_and_blocks_reentry():
    a, b, beta, _ = _inputs(n=40)
    z = pd.Series(2.5, index=a.index)            # never reverts
    res = run_backtest(a, b, beta, z, time_stop=7)
    assert res.trades[0].exit_reason == "time" and res.trades[0].holding_days == 7
    assert len(res.trades) == 1                  # z stayed above entry: no re-entry


def test_no_entry_from():
    a, b, beta, _ = _inputs(n=40)
    z = pd.Series(2.5, index=a.index)
    assert not run_backtest(a, b, beta, z, no_entry_from=1).trades
    assert run_backtest(a, b, beta, z, no_entry_from=10).trades[0].entry_date == a.index[1]


def test_carry_matches_one_long_run():
    """Splitting a run with carry_out/carry_in changes nothing: PnL, the
    trade log and holding days match a single run over the same days."""
    a, b, beta, _ = _inputs(n=120)
    z = pd.Series(np.sin(np.arange(120) / 9.0) * 3.0, index=a.index)
    size = pd.Series(1.0, index=a.index)
    kw = dict(stop_threshold=None)
    whole = run_backtest(a, b, beta, z, size=size, **kw)

    carried = 0
    for cut in (40, 47, 53, 60, 71):             # some cuts fall inside a trade
        first = run_backtest(a[:cut + 1], b[:cut + 1], beta[:cut + 1], z[:cut + 1],
                             size=size[:cut + 1], carry_out=True, **kw)
        second = run_backtest(a[cut:], b[cut:], beta[cut:], z[cut:], size=size[cut:],
                              carry_in=first.open_position, **kw)
        carried += first.open_position is not None
        joined = pd.concat([first.daily_pnl, second.daily_pnl.iloc[1:]])
        assert np.allclose(joined.values, whole.daily_pnl.values)
        trades = first.trades + second.trades
        assert len(trades) == len(whole.trades)
        for x, y in zip(trades, whole.trades):
            assert x.entry_date == y.entry_date and x.exit_date == y.exit_date
            assert x.holding_days == y.holding_days and np.isclose(x.pnl, y.pnl)
    assert carried > 0


def test_carried_days_count_toward_time_stop():
    a, b, beta, _ = _inputs(n=40)
    z = pd.Series(2.5, index=a.index)
    first = run_backtest(a[:11], b[:11], beta[:11], z[:11], carry_out=True, time_stop=15)
    assert first.open_position is not None and first.open_position.held_days == 9
    second = run_backtest(a[10:], b[10:], beta[10:], z[10:], time_stop=15,
                          carry_in=first.open_position)
    assert second.trades[0].holding_days == 15 and second.trades[0].exit_reason == "time"


def _book(n_periods, reselected):
    """Two tickers, 10-day periods, z stuck above entry so a trade is always
    open. `reselected`: which period indices have the pair selected."""
    rng = np.random.default_rng(1)
    days = 10 * n_periods + 15
    idx = pd.bdate_range("2021-01-01", periods=days)
    prices = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, (days, 2)), axis=0)),
                          index=idx, columns=["A", "B"])
    train = 5
    prepared, schedule = [], []
    for k in reselected:
        s, e = train + 10 * k, train + 10 * k + 9
        pp = PairPeriod("A", "B", 0.01, idx[s], idx[e], idx[0], half_life=4.0)
        w = prices.iloc[:e + 1]
        z = pd.Series(2.5, index=w.index)
        prep = SimpleNamespace(
            pp=pp, label=f"A/B [{k}]", series_a=w["A"], series_b=w["B"], first=s - 1,
            kf=SimpleNamespace(hedge_ratio=pd.Series(1.0, index=w.index)),
            size=pd.Series(1.0, index=w.index), kalman_params={},
            zscores=lambda lookback, signal, z=z: z,
        )
        prepared.append(prep)
        schedule.append(pp)
    return prices, schedule, prepared


def _run(prices, schedule, prepared, rules):
    return run_portfolio_backtest(prices, schedule, stop_threshold=None, prepared=prepared, exit_rules=rules)


def test_default_rules_change_nothing():
    prices, schedule, prepared = _book(3, [0, 1, 2])
    a, b = _run(prices, schedule, prepared, None), _run(prices, schedule, prepared, ExitRules())
    assert a.daily_pnl.equals(b.daily_pnl)


def test_carry_spans_reselected_periods():
    prices, schedule, prepared = _book(3, [0, 1, 2])
    res = _run(prices, schedule, prepared, ExitRules(carry=True))
    trades = [t for r in res.per_pair.values() for t in r.trades]
    assert len(trades) == 1                      # one trade across all three periods
    t = trades[0]
    assert t.exit_reason == "period_end" and t.entry_date == prices.index[5]
    assert t.exit_date == schedule[-1].period_end + pd.tseries.offsets.BDay(1)
    assert np.isclose(t.pnl, res.daily_pnl.sum())


def test_carry_closes_one_day_late_if_not_reselected():
    prices, schedule, prepared = _book(3, [0, 2])  # selected, gap, selected again
    res = _run(prices, schedule, prepared, ExitRules(carry=True))
    first = res.per_pair[("A", "B", schedule[0].period_start)].trades
    assert len(first) == 1
    assert first[0].exit_date == prices.index[prices.index.get_loc(schedule[0].period_end) + 1]
    assert np.isclose(sum(t.pnl for r in res.per_pair.values() for t in r.trades), res.daily_pnl.sum())


def test_late_entry_and_time_stop_use_half_life():
    prices, schedule, prepared = _book(1, [0])
    # half-life 4 -> no entry with fewer than 8 days left, time stop at 8 days
    res = _run(prices, schedule, prepared, ExitRules(time_stop_mult=2.0))
    assert res.per_pair[("A", "B", schedule[0].period_start)].trades[0].holding_days == 8
    z_late = pd.Series(np.nan, index=prepared[0].series_a.index)
    z_late.iloc[-10:] = 2.5                      # first decision has 8 days left
    prepared[0].zscores = lambda lookback, signal, z=z_late: z
    res = _run(prices, schedule, prepared, ExitRules(late_entry_mult=2.0))
    t = res.per_pair[("A", "B", schedule[0].period_start)].trades
    assert len(t) == 1 and t[0].holding_days == 8   # exactly 8 left: allowed
    z_late.iloc[-10] = np.nan
    res = _run(prices, schedule, prepared, ExitRules(late_entry_mult=2.0))
    assert not res.per_pair[("A", "B", schedule[0].period_start)].trades   # 7 left: refused


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS  {name}")
