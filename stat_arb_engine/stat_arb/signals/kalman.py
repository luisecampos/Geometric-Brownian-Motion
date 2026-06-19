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

The model:
  Observation:  price_a[t] = beta[t] * price_b[t] + eps[t]
  State:        beta[t]    = beta[t-1] + delta[t]

  Where:
    beta[t]   = hedge ratio (hidden state we estimate)
    eps[t]    ~ N(0, R)   observation noise
    delta[t]  ~ N(0, Q)   state transition noise (how fast beta drifts)

Tuning:
  - Higher Q  → beta tracks faster, more noise
  - Lower  Q  → beta is smoother, slower to adapt
"""

import numpy as np
import pandas as pd


class KalmanHedgeRatio:
    def __init__(self, delta: float = 1e-4, obs_noise: float = 1e-3):
        """
        delta     : state transition variance (how much hedge ratio can drift per day)
        obs_noise : observation noise variance (R in the Kalman equations)
        """
        self.delta     = delta
        self.obs_noise = obs_noise

    def fit(
        self, series_a: pd.Series, series_b: pd.Series
    ) -> tuple[pd.Series, pd.Series]:
        """
        Run the Kalman filter over the full price history.

        Returns:
            hedge_ratios : pd.Series — dynamic beta estimate each day
            spreads      : pd.Series — spread = price_a - beta * price_b
        """
        n = len(series_a)
        a = series_a.values
        b = series_b.values

        # State and covariance
        beta    = np.zeros(n)
        P       = np.zeros(n)   # estimate error covariance
        Q       = self.delta    # process noise
        R       = self.obs_noise

        # Initialise from first observation using OLS ratio
        beta[0] = a[0] / b[0] if b[0] != 0 else 1.0
        P[0]    = 1.0

        for t in range(1, n):
            # ── Predict ───────────────────────────────────────────
            beta_pred = beta[t - 1]
            P_pred    = P[t - 1] + Q

            # ── Update ────────────────────────────────────────────
            # Innovation (how wrong our prediction was)
            y_hat       = beta_pred * b[t]
            innovation  = a[t] - y_hat

            # Kalman gain — how much to trust the new observation
            S           = b[t] ** 2 * P_pred + R
            K           = P_pred * b[t] / S

            # Posterior estimates
            beta[t]     = beta_pred + K * innovation
            P[t]        = (1 - K * b[t]) * P_pred

        hedge_ratios = pd.Series(beta, index=series_a.index, name="hedge_ratio")
        spreads      = pd.Series(a - beta * b, index=series_a.index, name="spread")

        return hedge_ratios, spreads
