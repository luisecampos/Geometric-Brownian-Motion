"""
tests/test_residual.py
----------------------
Checks for signals/residual.py and backtest/residual_book.py, on synthetic
prices only. No real price enters these tests.

Run with `python -m pytest tests` or just `python tests/test_residual.py`.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.report import sharpe_ratio, summary_stats  # noqa: E402
from backtest.residual_book import run_residual_book  # noqa: E402
from signals.residual import MAX_B, WINDOW, fit_window, residual_scores  # noqa: E402


def _market(n=500, k=12, seed=1, reversion=0.8, spread_vol=0.015):
    """One ETF and k stocks: beta * ETF plus an AR(1) log residual (1.0 = a random walk)."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-01", periods=n)
    f = rng.normal(0.0003, 0.01, n)
    etf = pd.DataFrame({"ETF": 100 * np.exp(np.cumsum(f))}, index=idx)
    prices = {}
    for j in range(k):
        beta = 0.7 + 0.6 * j / max(k - 1, 1)
        x = np.zeros(n)
        for t in range(1, n):
            x[t] = reversion * x[t - 1] + rng.normal(0, spread_vol)
        prices[f"S{j:02d}"] = 50 * np.exp(beta * np.cumsum(f) + x)
    stocks = pd.DataFrame(prices, index=idx)
    return stocks, etf, {c: "ETF" for c in stocks.columns}


def _flat(n=12, names=("A", "B"), etfs=("ETF",)):
    """Constant prices, so only costs and borrow can move PnL."""
    idx = pd.bdate_range("2021-01-01", periods=n)
    stocks = pd.DataFrame(100.0, index=idx, columns=list(names))
    etf = pd.DataFrame(50.0, index=idx, columns=list(etfs))
    scores = pd.DataFrame(0.0, index=idx, columns=list(names))
    betas = pd.DataFrame(1.0, index=idx, columns=list(names))
    return stocks, etf, scores, betas


def test_fit_matches_the_paper_by_hand():
    rng = np.random.default_rng(0)
    r, f = rng.normal(0, 0.01, (WINDOW, 1)), rng.normal(0, 0.01, WINDOW)
    beta, a, b, var_z = fit_window(r, f)

    slope, intercept = np.polyfit(f, r[:, 0], 1)
    assert np.isclose(beta[0], slope)
    x = np.cumsum(r[:, 0] - intercept - slope * f)
    assert abs(x[-1]) < 1e-12                       # the paper's X_60 = 0
    b_hand, a_hand = np.polyfit(x[:-1], x[1:], 1)
    assert np.isclose(b[0], b_hand) and np.isclose(a[0], a_hand)
    assert np.isclose(var_z[0], np.var(x[1:] - a_hand - b_hand * x[:-1]))
    assert np.isclose(MAX_B, 0.9672, atol=1e-4)


def test_score_uses_the_formula_and_centres_across_stocks():
    stocks, etf, etf_of = _market(n=120)
    scores, betas = residual_scores(stocks, etf, etf_of)
    t = 100
    r = stocks.pct_change().iloc[t - WINDOW + 1:t + 1].to_numpy()
    f = etf["ETF"].pct_change().iloc[t - WINDOW + 1:t + 1].to_numpy()
    beta, a, b, var_z = fit_window(r, f)
    ok = (b > 0) & (b < MAX_B)
    m = a / (1 - b)
    expected = -(m - m[ok].mean()) / np.sqrt(var_z / (1 - b ** 2))
    got = scores.iloc[t].to_numpy()
    assert np.allclose(got[ok], expected[ok])
    assert np.isnan(got[~ok]).all()
    assert np.allclose(betas.iloc[t].to_numpy()[ok], beta[ok])


def test_no_score_before_a_full_window():
    stocks, etf, etf_of = _market(n=120)
    scores, _ = residual_scores(stocks, etf, etf_of)
    assert scores.iloc[:WINDOW].isna().all().all()
    assert scores.iloc[WINDOW:].notna().any().any()


def test_slow_residual_is_rejected():
    stocks, etf, etf_of = _market(n=200, reversion=1.0, spread_vol=0.002)
    fast, _ = residual_scores(*_market(n=200)[:2], etf_of)
    slow, _ = residual_scores(stocks, etf, etf_of)
    assert slow.notna().mean().mean() < fast.notna().mean().mean()


def test_scores_cannot_see_later_prices():
    stocks, etf, etf_of = _market(n=200)
    base, _ = residual_scores(stocks, etf, etf_of)
    t = 150
    later, later_etf = stocks.copy(), etf.copy()
    later.iloc[t:] *= 1.5
    later_etf.iloc[t:] *= 0.7
    changed, _ = residual_scores(later, later_etf, etf_of)
    assert base.iloc[:t].equals(changed.iloc[:t])


def test_a_price_at_t_changes_no_decision_before_t_plus_1():
    """A shock at the close of t reaches the score at t, so it can only act at t+1."""
    stocks, etf, etf_of = _market(n=300)
    scores, betas = residual_scores(stocks, etf, etf_of)
    base = run_residual_book(stocks, etf, etf_of, scores, betas, start=stocks.index[WINDOW + 1])
    t = 200
    shocked = stocks.copy()
    shocked.iloc[t:, 0] *= 1.4
    s2, b2 = residual_scores(shocked, etf, etf_of)
    run2 = run_residual_book(shocked, etf, etf_of, s2, b2, start=stocks.index[WINDOW + 1])

    def decisions(res):
        out = []
        for key, r in res.per_pair.items():
            out += [(key[0], x.entry_date) for x in r.trades]
        return sorted(d for d in out if d[1] <= stocks.index[t])

    assert decisions(base) == decisions(run2)


def test_one_day_lag_on_entry_and_exit():
    stocks, etf, scores, betas = _flat(n=8, names=("A",))
    scores.iloc[2:4] = -2.0             # scores on days 2 and 3, so in on day 3
    res = run_residual_book(stocks, etf, {"A": "ETF"}, scores, betas, cost_bps=0.0)
    tr = next(iter(res.per_pair.values())).trades[0]
    assert tr.entry_date == stocks.index[3]
    assert tr.exit_date == stocks.index[5]   # s back to 0 on day 4, so out on day 5
    assert tr.direction == 1 and tr.exit_reason == "signal"


def test_exit_bands_are_asymmetric():
    stocks, etf, scores, betas = _flat(n=10, names=("L", "H"))
    scores.loc[:, "L"] = -2.0
    scores.loc[:, "H"] = 2.0
    scores.iloc[4:, 0] = -0.6           # still inside the long's band
    scores.iloc[4:, 1] = 0.6            # already past the short's band
    res = run_residual_book(stocks, etf, {"L": "ETF", "H": "ETF"}, scores, betas, cost_bps=0.0)
    trades = {k[0]: r.trades[0] for k, r in res.per_pair.items()}
    assert trades["H"].exit_reason == "signal" and trades["H"].exit_date == stocks.index[5]
    assert trades["L"].exit_reason == "period_end"


def test_rejected_stock_closes():
    stocks, etf, scores, betas = _flat(n=10, names=("A",))
    scores.iloc[:] = -2.0
    scores.iloc[5:] = np.nan
    res = run_residual_book(stocks, etf, {"A": "ETF"}, scores, betas, cost_bps=0.0)
    tr = next(iter(res.per_pair.values())).trades[0]
    assert tr.exit_reason == "model" and tr.exit_date == stocks.index[6]


def test_slots_take_the_largest_scores_first():
    names = [f"S{j:02d}" for j in range(25)]
    stocks, etf, scores, betas = _flat(n=6, names=names)
    scores.iloc[:] = [-(1.3 + 0.1 * j) for j in range(25)]
    betas.iloc[:] = 0.0
    res = run_residual_book(stocks, etf, {c: "ETF" for c in names}, scores, betas, cost_bps=0.0)
    assert {k[0] for k in res.per_pair} == set(names[5:])
    assert res.positions.max() == 20 and res.blocked.iloc[1] == 5


def test_gross_cap_blocks_entries():
    names = [f"S{j:02d}" for j in range(20)]
    stocks, etf, scores, betas = _flat(n=6, names=names)
    scores.iloc[:] = -2.0
    betas.iloc[:] = 3.0                 # all one way: each adds 5% + 15% of gross
    res = run_residual_book(stocks, etf, {c: "ETF" for c in names}, scores, betas, cost_bps=0.0)
    assert res.gross.max() <= 2.0 + 1e-9
    assert res.positions.iloc[1] == 10 and res.blocked.iloc[1] == 10


def test_offsetting_hedges_net_for_gross_and_for_cost():
    stocks, etf, scores, betas = _flat(n=6, names=("A", "B"))
    scores["A"], scores["B"] = -2.0, 2.0
    netted = run_residual_book(stocks, etf, {"A": "ETF", "B": "ETF"}, scores, betas, cost_bps=1e-4)
    plain = run_residual_book(stocks, etf, {"A": "ETF", "B": "ETF"}, scores, betas, cost_bps=1e-4, net_etf=False)
    assert np.isclose(netted.gross.iloc[1], 0.10)      # the ETF legs cancel
    assert netted.etf_traded.sum() == 0.0
    assert np.isclose(netted.daily_pnl.sum(), -1e-4 * 0.20)      # two stocks, in and out
    assert np.isclose(plain.etf_traded.sum(), 0.20)
    assert np.isclose(plain.daily_pnl.sum(), -1e-4 * 0.40)


def test_borrow_charges_short_stock_and_net_short_etf():
    stocks, etf, scores, betas = _flat(n=7, names=("A", "B"))
    scores["A"] = 2.0                   # short A, long the ETF
    res = run_residual_book(stocks, etf, {"A": "ETF", "B": "ETF"}, scores, betas, cost_bps=0.0, borrow_fee=0.01)
    held_nights = 5                     # in at the close of day 1, out at the close of day 6
    assert np.isclose(res.daily_pnl.sum(), -0.01 / 252 * 0.05 * held_nights)

    scores["A"] = -2.0                  # long A, short the ETF
    res = run_residual_book(stocks, etf, {"A": "ETF", "B": "ETF"}, scores, betas, cost_bps=0.0, borrow_fee=0.01)
    assert np.isclose(res.daily_pnl.sum(), -0.01 / 252 * 0.05 * held_nights)


def test_costs_are_linear_and_leave_trades_alone():
    stocks, etf, etf_of = _market(n=400)
    scores, betas = residual_scores(stocks, etf, etf_of)
    kw = dict(start=stocks.index[WINDOW + 1])
    free = run_residual_book(stocks, etf, etf_of, scores, betas, cost_bps=0.0, **kw)
    paid = run_residual_book(stocks, etf, etf_of, scores, betas, cost_bps=5e-4, borrow_fee=0.005, **kw)

    def dates(res):
        return sorted((k[0], t.entry_date, t.exit_date) for k, r in res.per_pair.items() for t in r.trades)

    assert dates(free) == dates(paid) and len(dates(free)) > 20
    traded = free.stock_traded.sum() + free.etf_traded.sum()
    assert paid.daily_pnl.sum() < free.daily_pnl.sum() - 5e-4 * traded + 1e-12


def test_trade_pnl_adds_up_at_zero_cost():
    stocks, etf, etf_of = _market(n=400)
    scores, betas = residual_scores(stocks, etf, etf_of)
    res = run_residual_book(stocks, etf, etf_of, scores, betas, start=stocks.index[WINDOW + 1], cost_bps=0.0)
    total = sum(t.pnl for r in res.per_pair.values() for t in r.trades)
    assert np.isclose(total, res.daily_pnl.sum())
    assert summary_stats(res)["trades"] > 20


def test_start_leaves_warm_up_out():
    stocks, etf, etf_of = _market(n=300)
    scores, betas = residual_scores(stocks, etf, etf_of)
    res = run_residual_book(stocks, etf, etf_of, scores, betas, start=stocks.index[200])
    assert res.daily_pnl.index[0] == stocks.index[200]
    assert min(t.entry_date for r in res.per_pair.values() for t in r.trades) >= stocks.index[200]


def test_reporting_runs_on_synthetic_results():
    """The script's reporting path, so the one real run can't stop halfway."""
    import residual_test as R
    from backtest.export import variant_tables

    stocks, etf, etf_of = _market(n=400)
    scores, betas = residual_scores(stocks, etf, etf_of)
    results = R.run_levels(stocks, etf, etf_of, scores, betas, stocks.index[WINDOW + 1])
    assert list(results)[:4] == list(R.LEVELS)
    dsr = {name: 0.5 for name in results}
    R.report("SYNTHETIC", results, stocks, dsr, "synthetic")
    tables = variant_tables("synthetic", results, dsr, sectors={c: "x" for c in stocks.columns})
    assert len(tables["variants"]) == 5 and not tables["sectors"].empty

    e = R.economics(results["FREE"], results["REALISTIC"])
    assert e["breakeven_with_borrow_bps"] < e["breakeven_bps"]


def test_planted_reversion_is_found_and_noise_is_not():
    k = dict(n=1500, k=40, spread_vol=0.01)
    def sharpe(**kw):
        stocks, etf, etf_of = _market(**k, **kw)
        scores, betas = residual_scores(stocks, etf, etf_of)
        res = run_residual_book(stocks, etf, etf_of, scores, betas, start=stocks.index[WINDOW + 1],
                                cost_bps=5e-4, borrow_fee=0.005)
        return sharpe_ratio(res.daily_pnl)

    assert sharpe(seed=2, reversion=0.8) > 1.0
    noise = [sharpe(seed=s, reversion=1.0) for s in (3, 4, 5, 6)]
    assert max(noise) < 1.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
