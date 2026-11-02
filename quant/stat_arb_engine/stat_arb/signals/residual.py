"""
signals/residual.py
-------------------
Score each stock's residual against its sector ETF (Avellaneda & Lee 2010).

Each day, on the last 60 daily returns of a stock and its ETF:

  1. Regress R = beta0 + beta * R_etf + e
  2. Cumulate the residuals: X_k = e_1 + ... + e_k
  3. Fit X_(n+1) = a + b * X_n + z, an AR(1) version of an OU process
  4. Reject the stock unless 0 < b < exp(-1/30), i.e. kappa = -ln(b) * 252 > 8.4
  5. m = a / (1 - b), centred across that day's accepted stocks
  6. sigma_eq = sqrt(Var(z) / (1 - b^2)) and s = -m_centred / sigma_eq

The regression forces X_60 = 0, so s is just -m / sigma_eq. s < 0 means the stock
is cheap against its sector, s > 0 rich.

A score at date t uses returns up to and including t's close. The book must act
on it no earlier than the close of t+1 (see backtest/residual_book.py).
"""

import numpy as np
import pandas as pd

WINDOW    = 60             # trading days, for both fits
MIN_KAPPA = 252 / 30       # mean reversion within ~1.5 months
MAX_B     = float(np.exp(-MIN_KAPPA / 252))


def fit_window(r: np.ndarray, f: np.ndarray) -> tuple:
    """One window: r is days x stocks, f is the ETF's returns. Returns beta, a, b, Var(z)."""
    f_c = f - f.mean()
    beta = f_c @ (r - r.mean(axis=0)) / (f_c @ f_c)
    beta0 = r.mean(axis=0) - beta * f.mean()
    x = np.cumsum(r - beta0 - np.outer(f, beta), axis=0)

    lag, lead = x[:-1], x[1:]
    lag_c = lag - lag.mean(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        b = (lag_c * (lead - lead.mean(axis=0))).sum(axis=0) / (lag_c ** 2).sum(axis=0)
    a = lead.mean(axis=0) - b * lag.mean(axis=0)
    return beta, a, b, (lead - a - b * lag).var(axis=0)


def residual_scores(
    stocks: pd.DataFrame,
    etfs  : pd.DataFrame,
    etf_of: dict[str, str],
    window: int = WINDOW,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Daily s-scores and betas, one column per stock.

    Args:
        stocks : closing prices, one column per stock
        etfs   : ETF closing prices on the same dates; NaN before an ETF listed
        etf_of : stock -> its sector ETF

    Returns:
        (scores, betas). A score is NaN when the stock is rejected that day or
        doesn't have `window` returns yet.
    """
    r = stocks.pct_change(fill_method=None).to_numpy()
    f_all = etfs.pct_change(fill_method=None)
    n, k = r.shape
    m, sigma, beta = np.full((n, k), np.nan), np.full((n, k), np.nan), np.full((n, k), np.nan)

    groups: dict[str, list[int]] = {}
    for i, name in enumerate(stocks.columns):
        groups.setdefault(etf_of[name], []).append(i)

    for etf, cols in groups.items():
        f = f_all[etf].to_numpy()
        for t in range(window, n):
            fw = f[t - window + 1:t + 1]
            if np.isnan(fw).any():
                continue
            bt, a, b, var_z = fit_window(r[t - window + 1:t + 1][:, cols], fw)
            ok = (b > 0) & (b < MAX_B) & (var_z > 0)
            for j, col in enumerate(cols):
                if ok[j]:
                    beta[t, col] = bt[j]
                    m[t, col] = a[j] / (1 - b[j])
                    sigma[t, col] = np.sqrt(var_z[j] / (1 - b[j] ** 2))

    has_any = ~np.isnan(m).all(axis=1)
    centre = np.full(n, np.nan)
    centre[has_any] = np.nanmean(m[has_any], axis=1)
    scores = -(m - centre[:, None]) / sigma
    return (pd.DataFrame(scores, index=stocks.index, columns=stocks.columns),
            pd.DataFrame(beta, index=stocks.index, columns=stocks.columns))
