"""
Implied volatility surface builder.

Part 1: Black-Scholes pricer + implied vol solver (the actual math).
Part 2: Synthetic surface (same smile/skew demo as the interactive version).
Part 3: Real surface pulled from a live options chain (needs `yfinance`).

Install deps:
    pip install numpy scipy plotly yfinance --break-system-packages
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import plotly.graph_objects as go


# ---------- Part 1: Black-Scholes + implied vol ----------

def bs_call_price(S, K, T, r, sigma):
    """Black-Scholes price of a European call."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def implied_vol(market_price, S, K, T, r, low=1e-4, high=5.0):
    """
    Solve for the sigma that makes bs_call_price match the market price.
    No closed-form inverse exists, so we root-find (bisection via brentq).
    """
    def diff(sigma):
        return bs_call_price(S, K, T, r, sigma) - market_price

    # If the market price is outside what any vol could produce, bail out.
    if diff(low) > 0 or diff(high) < 0:
        return np.nan
    return brentq(diff, low, high)


# ---------- Part 2: synthetic surface (illustrative, no market data needed) ----------

def synthetic_surface(atm_vol=0.28, skew=-0.15, smile=0.35, term_decay=0.6):
    strikes = np.arange(70, 131, 4)
    expiries = np.arange(0.08, 2.01, 0.16)
    spot = 100

    def vol_at(K, T):
        log_m = np.log(K / spot)
        base = atm_vol + skew * log_m + smile * log_m**2
        return max(base * np.exp(-term_decay * T) + atm_vol * 0.7 * (1 - np.exp(-term_decay * T)), 0.03)

    Z = np.array([[vol_at(K, T) * 100 for K in strikes] for T in expiries])
    return strikes, expiries, Z


def plot_surface(strikes, expiries, Z, title="Implied volatility surface"):
    fig = go.Figure(data=[go.Surface(x=strikes, y=expiries, z=Z, colorscale="Portland")])
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="strike",
            yaxis_title="time to expiry (yrs)",
            zaxis_title="implied vol (%)",
        ),
        height=600,
    )
    fig.show()


# ---------- Part 3: real surface from a live options chain ----------

def real_surface(ticker="AAPL", r=0.045):
    """
    Pulls a real options chain and backs out implied vol per strike/expiry
    using the solver above. Requires `yfinance`.
    """
    import yfinance as yf
    from datetime import datetime

    tk = yf.Ticker(ticker)
    spot = tk.history(period="1d")["Close"].iloc[-1]
    today = datetime.now()

    strikes_all, expiries_all, ivs_all = [], [], []

    for exp_str in tk.options[:8]:  # cap to first 8 expiries for speed
        chain = tk.option_chain(exp_str).calls
        T = (datetime.strptime(exp_str, "%Y-%m-%d") - today).days / 365
        if T <= 0:
            continue
        for _, row in chain.iterrows():
            mid = (row["bid"] + row["ask"]) / 2
            if mid <= 0 or row["strike"] <= 0:
                continue
            iv = implied_vol(mid, spot, row["strike"], T, r)
            if not np.isnan(iv):
                strikes_all.append(row["strike"])
                expiries_all.append(T)
                ivs_all.append(iv * 100)

    return np.array(strikes_all), np.array(expiries_all), np.array(ivs_all), spot


def plot_real_surface_scatter(strikes, expiries, ivs, ticker):
    """Real chains are irregular (not a clean grid), so scatter3d fits better than Surface."""
    fig = go.Figure(data=[go.Scatter3d(
        x=strikes, y=expiries, z=ivs,
        mode="markers",
        marker=dict(size=3, color=ivs, colorscale="Portland"),
    )])
    fig.update_layout(
        title=f"{ticker} implied vol surface (live chain)",
        scene=dict(
            xaxis_title="strike",
            yaxis_title="time to expiry (yrs)",
            zaxis_title="implied vol (%)",
        ),
        height=600,
    )
    fig.show()


def run_synthetic_interactive():
    """Prompt for the four shape parameters and plot the synthetic surface."""
    print("Leave blank to use the default value shown in brackets.\n")
    atm_vol = input("ATM vol, e.g. 0.28 [0.28]: ").strip()
    skew = input("Skew, e.g. -0.15 [-0.15]: ").strip()
    smile = input("Smile curvature, e.g. 0.35 [0.35]: ").strip()
    term_decay = input("Term decay, e.g. 0.6 [0.6]: ").strip()

    kwargs = {}
    if atm_vol:
        kwargs["atm_vol"] = float(atm_vol)
    if skew:
        kwargs["skew"] = float(skew)
    if smile:
        kwargs["smile"] = float(smile)
    if term_decay:
        kwargs["term_decay"] = float(term_decay)

    strikes, expiries, Z = synthetic_surface(**kwargs)
    plot_surface(strikes, expiries, Z, title="Synthetic vol surface")


def run_real_interactive():
    """Prompt for a ticker and plot its live implied vol surface."""
    ticker = input("Ticker, e.g. AAPL: ").strip().upper()
    if not ticker:
        print("No ticker entered, skipping.")
        return
    print(f"Pulling options chain for {ticker}... (needs yfinance + internet)")
    try:
        strikes, expiries, ivs, spot = real_surface(ticker)
        if len(strikes) == 0:
            print(f"No usable option quotes found for {ticker} — it may not have listed options.")
            return
        plot_real_surface_scatter(strikes, expiries, ivs, ticker)
    except Exception as e:
        print(f"Couldn't build a surface for {ticker}: {e}")


if __name__ == "__main__":
    mode = input("Type 'synthetic' or 'real': ").strip().lower()
    if mode == "real":
        run_real_interactive()
    else:
        run_synthetic_interactive()
