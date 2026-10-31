"""
learn/allocation.py
-------------------
Thompson-sampling capital allocation across the pairs active in a period.

Each pair (ticker_a, ticker_b) is an arm. Its reward for a pair-period is
the total PnL that pair-period produced at baseline size. A reward is
observed once the period is over, so at rebalance T only pair-periods that
ended before T count.

Model: reward ~ N(theta_pair, sigma^2), prior theta ~ N(0, tau^2). The
prior is centred at zero (skeptical), and sigma^2 = tau^2 = the pooled
variance of all rewards seen so far, so the prior is worth one observation.
A pair that has never traded keeps the prior, with wide uncertainty, and so
still gets explored.

Allocation at T: draw theta for every active pair many times and estimate
P(pair is the best active pair). Weight = n_active * P(best), so equal
probabilities give everyone baseline size. That weight is shrunk halfway
toward 1 and capped, so one lucky quarter can't put the whole book on one
pair.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def thompson_multipliers(
    baseline     : "PortfolioResults",
    prepared     : list,
    n_draws      : int = 4000,
    shrink       : float = 0.5,
    max_mult     : float = 3.0,
    min_history  : int = 8,
    seed         : int = 0,
) -> tuple[dict, pd.DataFrame]:
    """Returns ({pair label: multiplier}, per-pair-period allocation log)."""
    rng = np.random.default_rng(seed)
    rewards = []
    for prep in prepared:
        pp = prep.pp
        r = float(baseline.pair_pnl[prep.label].loc[pp.period_start:pp.period_end].sum())
        rewards.append(((pp.ticker_a, pp.ticker_b), pp.period_start, pp.period_end, r))
    rewards = pd.DataFrame(rewards, columns=["pair", "period_start", "period_end", "reward"])

    mults, log = {}, []
    for T in sorted(rewards["period_start"].unique()):
        active = [p for p in prepared if p.pp.period_start == T]
        done = rewards[rewards["period_end"] < T]
        if len(done) < min_history or len(active) < 2:
            for p in active:
                mults[p.label] = 1.0
                log.append((p.label, T, np.nan, 1.0))
            continue

        sigma2 = max(float(done["reward"].var(ddof=1)), 1e-12)
        tau2 = sigma2
        g = done.groupby("pair")["reward"]
        stats = dict(zip(g.sum().index, zip(g.sum().values, g.count().values)))
        post_mean, post_sd = [], []
        for p in active:
            r_sum, r_n = stats.get((p.pp.ticker_a, p.pp.ticker_b), (0.0, 0))
            var = 1.0 / (1.0 / tau2 + r_n / sigma2)
            post_mean.append(var * r_sum / sigma2)
            post_sd.append(np.sqrt(var))
        draws = rng.normal(post_mean, post_sd, size=(n_draws, len(active)))
        p_best = np.bincount(draws.argmax(axis=1), minlength=len(active)) / n_draws
        w = (1 - shrink) * len(active) * p_best + shrink
        w = np.clip(w, 0.0, max_mult)
        for p, pb, wi in zip(active, p_best, w):
            mults[p.label] = float(wi)
            log.append((p.label, T, float(pb), float(wi)))

    return mults, pd.DataFrame(log, columns=["pair_label", "period_start", "p_best", "mult"])
