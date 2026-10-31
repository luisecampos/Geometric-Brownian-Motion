"""
signals/kalman.py
-----------------
Dynamic hedge ratio estimation using a Kalman filter.

Why not just use OLS?
  OLS gives you a static hedge ratio fit over the whole history. But the true
  relationship between two stocks drifts over time — sector weights change,
  capital structures shift, macro regimes rotate. A Kalman filter treats the
  hedge ratio as a hidden state that evolves, updating it every day as new
  prices arrive.

The model (in LOG prices, x centred on its training-window mean):
  Observation:  log_a[t] = beta[t] * x[t] + alpha[t] + eps[t]
  State:        [beta, alpha][t] = [beta, alpha][t-1] + w[t]

  Where:
    x[t]      = log_b[t] - mean(log_b over the training window)
    beta[t]   = hedge ratio (hidden state) — in log space this is a DOLLAR
                ratio: $1 of A is hedged with $beta of B
    alpha[t]  = intercept (hidden state) — the spread's slowly-moving mean
    eps[t]    ~ N(0, R)   observation noise (the spread itself lives here)
    w[t]      ~ N(0, Q)   state noise, Q = q * diag(1/var(x), 1)

Why the old version was broken:
  It worked in raw prices with Q=1e-4, R=1e-3. For ~$100 stocks the gain on
  beta was ~1, so beta moved every day to explain that day's price — the
  "spread" a - beta*b became the one-day change, not a mean-reverting level
  (autocorrelation ~0 instead of ~0.96 on a synthetic OU pair).

  Two things fix it:
    1. Log prices + centring make R and Q scale-free (a $20 stock and a $400
       stock look the same) and decorrelate beta from alpha.
    2. The state must adapt MUCH slower than the spread mean-reverts,
       otherwise the filter learns the spread away. R is estimated by maximum
       likelihood on the training window; the state/observation noise ratio
       q/R is also estimated by MLE but capped so the filter's memory is at
       least `min_state_memory` days (a local-level filter's steady-state
       gain is ~sqrt(q/R), i.e. it forgets on a ~1/sqrt(q/R)-day horizon).
       Left uncapped, MLE happily drives q/R up, because a fast-moving
       intercept "explains" an autocorrelated spread better than white noise
       does — which is exactly the failure mode above. In practice the cap
       binds on essentially every pair, so q is "MLE-scaled R / memory^2".

  Choosing min_state_memory (default 500 days): on synthetic OU pairs
  (half-life ~13 days, 252-day training window, 20 seeds) the median
  correlation between estimated and true spread is 0.87 / 0.92 / 0.96 for a
  252 / 500 / 1000-day memory. Shorter memory = the intercept eats more of the
  spread; longer memory = the hedge is effectively static OLS. 500 is the
  compromise — the hedge still drifts, just on a multi-year horizon.

Outputs:
  spread[t] = log_a[t] - beta_prior[t]*x[t] - alpha_prior[t]
    — computed with the state from the PREVIOUS close (one-step-ahead
    prediction), so today's price can't leak into today's hedge. This is
    also the filter innovation e[t]; its variance S[t] gives a second,
    self-normalised signal z = e / sqrt(S).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar


@dataclass
class KalmanResult:
    hedge_ratio    : pd.Series   # posterior beta at each close (use [t-1] to trade at t)
    intercept      : pd.Series   # posterior alpha at each close
    spread         : pd.Series   # one-step-ahead residual (== innovation e[t])
    innovation_var : pd.Series   # S[t], predicted variance of e[t]

    @property
    def innovation(self) -> pd.Series:
        return self.spread.rename("innovation")

    @property
    def innovation_z(self) -> pd.Series:
        """z = e / sqrt(S): the filter's own standardised surprise."""
        return (self.spread / np.sqrt(self.innovation_var)).rename("zscore")


def _filter(y, x, m0, P0, q_beta, q_alpha, R):
    """
    Run the 2-state filter with plain-float arithmetic (2x2 matrices by hand
    are ~50x faster than numpy for scalar-sized ops in a Python loop — this
    runs inside the MLE optimiser, so it matters).

    Returns (beta_post, alpha_post, e, S) arrays. e/S at index 0 are computed
    against the initial state m0.
    """
    n = len(y)
    beta_post  = np.empty(n)
    alpha_post = np.empty(n)
    e_arr      = np.empty(n)
    S_arr      = np.empty(n)

    b, a = float(m0[0]), float(m0[1])
    p11, p12, p22 = float(P0[0, 0]), float(P0[0, 1]), float(P0[1, 1])

    for t in range(n):
        xt = x[t]
        # ── Predict (random-walk state: mean unchanged, covariance grows) ──
        p11 += q_beta
        p22 += q_alpha

        # ── Innovation against the prior state ────────────────────────────
        e   = y[t] - (b * xt + a)
        ph0 = p11 * xt + p12          # P h,  h = [x, 1]
        ph1 = p12 * xt + p22
        S   = xt * ph0 + ph1 + R

        # ── Update ────────────────────────────────────────────────────────
        k0, k1 = ph0 / S, ph1 / S
        b += k0 * e
        a += k1 * e
        p11 -= k0 * ph0
        p12 -= k0 * ph1
        p22 -= k1 * ph1

        beta_post[t], alpha_post[t] = b, a
        e_arr[t], S_arr[t] = e, S

    return beta_post, alpha_post, e_arr, S_arr


class KalmanHedgeRatio:
    def __init__(
        self,
        delta            : float | None = None,
        obs_noise        : float | None = None,
        min_state_memory : float = 500.0,
    ):
        """
        delta            : state noise variance q (log-price units). None (the
                           default) estimates it by MLE on the training window.
        obs_noise        : observation noise variance R (log-price units).
                           None estimates it by MLE on the training window.
        min_state_memory : lower bound, in days, on how slowly the hedge may
                           adapt (caps q/R at 1/min_state_memory^2). Should be
                           well above the spread half-lives you want to trade.

        Pass both delta and obs_noise to skip estimation, or neither to
        estimate both.
        """
        if (delta is None) != (obs_noise is None):
            raise ValueError("pass both delta and obs_noise, or neither (MLE)")
        self.delta            = delta
        self.obs_noise        = obs_noise
        self.min_state_memory = min_state_memory
        self.fitted_params    : dict = {}

    def fit(
        self,
        series_a   : pd.Series,
        series_b   : pd.Series,
        train_size : int | None = None,
    ) -> KalmanResult:
        """
        Estimate noise parameters on the first `train_size` observations (the
        training window — defaults to all of them), then run the filter over
        the full series.

        Everything that's estimated (centring constant, initial state, Q, R)
        uses only the training window, so later observations are filtered
        strictly causally.
        """
        n = len(series_a)
        train_size = n if train_size is None else int(train_size)
        if train_size < 20:
            raise ValueError(f"need >= 20 training observations, got {train_size}")

        y_all = np.log(series_a.values.astype(float))
        lb    = np.log(series_b.values.astype(float))

        x_mean = lb[:train_size].mean()
        x_all  = lb - x_mean
        y_tr, x_tr = y_all[:train_size], x_all[:train_size]

        # Initial state: OLS on the training window. Its covariance (per unit
        # of residual variance) seeds P0, so P0, Q and R all scale with R —
        # which lets R be concentrated out of the likelihood analytically.
        X        = np.column_stack([x_tr, np.ones(train_size)])
        m0, *_   = np.linalg.lstsq(X, y_tr, rcond=None)
        XtX_inv  = np.linalg.inv(X.T @ X)
        var_x    = max(float(x_tr.var()), 1e-8)

        # Q = q * diag(1/var(x), 1): beta-noise and alpha-noise each move the
        # fitted value by ~q per day, so one number sets the adaptation speed.
        def unit_filter(ratio, y, x):
            return _filter(y, x, m0, XtX_inv, ratio / var_x, ratio, 1.0)

        max_ratio = 1.0 / self.min_state_memory ** 2

        if self.delta is not None:
            R     = float(self.obs_noise)
            ratio = float(self.delta) / R
            q     = float(self.delta)
            method = "fixed"
        else:
            # Concentrated likelihood: with every variance proportional to R,
            # e[t] doesn't depend on R and S[t] = R * S~[t], so
            #   R_hat = mean(e^2 / S~),   nll = 0.5*sum(log S~) + n/2*log(R_hat)
            def nll(log_ratio):
                _, _, e, S = unit_filter(math.exp(log_ratio), y_tr, x_tr)
                r_hat = float(np.mean(e * e / S))
                return 0.5 * float(np.sum(np.log(S))) + 0.5 * train_size * math.log(r_hat)

            res = minimize_scalar(
                nll,
                bounds=(math.log(min(1e-10, max_ratio * 1e-3)), math.log(max_ratio)),
                method="bounded",
                options={"xatol": 1e-3},
            )
            ratio = math.exp(res.x)
            _, _, e, S = unit_filter(ratio, y_tr, x_tr)
            R = float(np.mean(e * e / S))
            q = ratio * R
            method = "mle"

        beta, alpha, e, S = _filter(
            y_all, x_all, m0, XtX_inv * R, q / var_x, q, R
        )

        self.fitted_params = {
            "method"      : method,
            "R"           : R,
            "q"           : q,
            "q_over_R"    : ratio,
            "ratio_capped": bool(ratio >= max_ratio * 0.999),
            "x_mean"      : x_mean,
            "beta0"       : float(m0[0]),
        }

        idx = series_a.index
        return KalmanResult(
            hedge_ratio    = pd.Series(beta,  index=idx, name="hedge_ratio"),
            intercept      = pd.Series(alpha, index=idx, name="intercept"),
            spread         = pd.Series(e,     index=idx, name="spread"),
            innovation_var = pd.Series(S,     index=idx, name="innovation_var"),
        )
