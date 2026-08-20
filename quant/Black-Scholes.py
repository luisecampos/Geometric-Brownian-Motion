import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import norm
import yfinance as yf
from datetime import datetime

plt.style.use('dark_background')

# ── Color palette ──────────────────────────────────────────────
C_CALL   = '#00FF9C'   # green
C_PUT    = '#FF4C4C'   # red
C_NEUTRAL = '#FFD700'  # gold
C_BAND   = '#4488FF'   # blue
C_WHITE  = '#FFFFFF'

# ══════════════════════════════════════════════════════════════
# CORE BLACK-SCHOLES ENGINE
# ══════════════════════════════════════════════════════════════

def black_scholes(S, K, T, r, sigma, option_type='call'):
    """
    S     : Current stock price
    K     : Strike price
    T     : Time to expiration in years
    r     : Risk-free rate (e.g. 0.05 for 5%)
    sigma : Annualized volatility (e.g. 0.25 for 25%)
    """
    if T <= 0:
        # At expiration — option is worth only intrinsic value
        if option_type == 'call':
            return max(S - K, 0)
        else:
            return max(K - S, 0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return price


def greeks(S, K, T, r, sigma, option_type='call'):
    """Returns all 5 Greeks for a given option."""
    if T <= 0:
        return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Delta — sensitivity to stock price
    if option_type == 'call':
        delta = norm.cdf(d1)
    else:
        delta = norm.cdf(d1) - 1

    # Gamma — same for calls and puts
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))

    # Theta — daily time decay (divide by 365)
    if option_type == 'call':
        theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
                 - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    else:
        theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
                 + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365

    # Vega — sensitivity to 1% change in volatility
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100

    # Rho — sensitivity to 1% change in interest rate
    if option_type == 'call':
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    else:
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

    return {'delta': delta, 'gamma': gamma,
            'theta': theta, 'vega': vega, 'rho': rho}


def implied_volatility(market_price, S, K, T, r, option_type='call'):
    """
    Reverse-engineers volatility from a known market price.
    Uses bisection method — tries volatilities between 0.01% and 500%
    until it finds one that produces the observed market price.
    """
    low, high = 0.001, 5.0
    for _ in range(200):
        mid = (low + high) / 2
        bs_price = black_scholes(S, K, T, r, mid, option_type)
        if abs(bs_price - market_price) < 0.0001:
            return mid
        if bs_price < market_price:
            low = mid
        else:
            high = mid
    return mid


# ══════════════════════════════════════════════════════════════
# USER INPUTS
# ══════════════════════════════════════════════════════════════

ticker_symbol = input("Enter ticker symbol (e.g. AAPL): ").upper().strip()
ticker = yf.Ticker(ticker_symbol)
hist   = ticker.history(period="1y")

if hist.empty:
    print(f"Could not retrieve data for {ticker_symbol}.")
else:
    closes   = hist['Close']
    log_rets = np.log(closes / closes.shift(1)).dropna()
    hist_vol = log_rets.std() * np.sqrt(252)
    S        = closes.iloc[-1]

    print(f"\n  Last close:        ${S:.2f}")
    print(f"  Historical vol:    {hist_vol*100:.1f}%\n")

    K     = float(input("Strike price (e.g. 200): "))
    T_days = int(input("Days until expiration (e.g. 30): "))
    T     = T_days / 365
    r     = float(input("Risk-free rate (e.g. 0.05 for 5%): "))
    sigma_input = input(f"Volatility — press Enter to use historical ({hist_vol*100:.1f}%), or type a value (e.g. 0.30): ")
    sigma = float(sigma_input) if sigma_input.strip() else hist_vol

    # ── Calculate prices and Greeks ────────────────────────────
    call_price = black_scholes(S, K, T, r, sigma, 'call')
    put_price  = black_scholes(S, K, T, r, sigma, 'put')
    call_greeks = greeks(S, K, T, r, sigma, 'call')
    put_greeks  = greeks(S, K, T, r, sigma, 'put')

    # Moneyness label
    if S > K:
        moneyness = "IN THE MONEY"
    elif S < K:
        moneyness = "OUT OF THE MONEY"
    else:
        moneyness = "AT THE MONEY"

    print(f"\n  {'═'*44}")
    print(f"  BLACK-SCHOLES RESULTS — {ticker_symbol}")
    print(f"  {'═'*44}")
    print(f"  Stock price:       ${S:.2f}  ({moneyness})")
    print(f"  Strike price:      ${K:.2f}")
    print(f"  Expiration:        {T_days} days")
    print(f"  Volatility used:   {sigma*100:.1f}%")
    print(f"  {'─'*44}")
    print(f"  CALL price:        ${call_price:.4f}")
    print(f"  PUT  price:        ${put_price:.4f}")
    print(f"  {'─'*44}")
    print(f"  PUT-CALL PARITY CHECK:")
    parity_check = call_price - put_price - (S - K * np.exp(-r * T))
    print(f"  C - P - (S - Ke^-rT) = {parity_check:.6f}  ({'✓ holds' if abs(parity_check) < 0.001 else '✗ broken'})")
    print(f"  {'─'*44}")
    print(f"  GREEKS           CALL          PUT")
    print(f"  Delta:        {call_greeks['delta']:>8.4f}      {put_greeks['delta']:>8.4f}")
    print(f"  Gamma:        {call_greeks['gamma']:>8.4f}      {put_greeks['gamma']:>8.4f}")
    print(f"  Theta ($/day):{call_greeks['theta']:>8.4f}      {put_greeks['theta']:>8.4f}")
    print(f"  Vega  (1%vol): {call_greeks['vega']:>8.4f}      {put_greeks['vega']:>8.4f}")
    print(f"  Rho   (1%rate):{call_greeks['rho']:>8.4f}      {put_greeks['rho']:>8.4f}")
    print(f"  {'═'*44}\n")

    # ══════════════════════════════════════════════════════════
    # FIGURE 1 — Option Price Surface + Greeks
    # ══════════════════════════════════════════════════════════

    stock_range = np.linspace(S * 0.5, S * 1.5, 300)
    time_range  = np.linspace(0.001, 1.0, 300)
    S_grid, T_grid = np.meshgrid(stock_range, time_range)

    call_surface = np.vectorize(lambda s, t: black_scholes(s, K, t, r, sigma, 'call'))(S_grid, T_grid)
    put_surface  = np.vectorize(lambda s, t: black_scholes(s, K, t, r, sigma, 'put'))(S_grid, T_grid)

    fig = plt.figure(figsize=(18, 10), facecolor='#0D0D0D')
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    ax_call3d = fig.add_subplot(gs[0, 0], projection='3d')
    ax_put3d  = fig.add_subplot(gs[0, 1], projection='3d')
    ax_pnl    = fig.add_subplot(gs[0, 2])
    ax_delta  = fig.add_subplot(gs[1, 0])
    ax_theta  = fig.add_subplot(gs[1, 1])
    ax_vega   = fig.add_subplot(gs[1, 2])

    for ax in [ax_pnl, ax_delta, ax_theta, ax_vega]:
        ax.set_facecolor('#0D0D0D')
        ax.tick_params(colors='#AAAAAA')
        for spine in ax.spines.values():
            spine.set_edgecolor('#333333')

    for ax in [ax_call3d, ax_put3d]:
        ax.set_facecolor('#0D0D0D')
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

    # 3D Call surface
    ax_call3d.plot_surface(S_grid, T_grid, call_surface,
                           cmap='viridis', alpha=0.85, linewidth=0)
    ax_call3d.set_title('Call Price Surface', color='white', fontsize=10, fontweight='bold')
    ax_call3d.set_xlabel('Stock Price', color='#AAAAAA', fontsize=8)
    ax_call3d.set_ylabel('Time (yrs)',  color='#AAAAAA', fontsize=8)
    ax_call3d.set_zlabel('Call Price',  color='#AAAAAA', fontsize=8)
    ax_call3d.tick_params(colors='#AAAAAA', labelsize=7)

    # 3D Put surface
    ax_put3d.plot_surface(S_grid, T_grid, put_surface,
                          cmap='plasma', alpha=0.85, linewidth=0)
    ax_put3d.set_title('Put Price Surface', color='white', fontsize=10, fontweight='bold')
    ax_put3d.set_xlabel('Stock Price', color='#AAAAAA', fontsize=8)
    ax_put3d.set_ylabel('Time (yrs)',  color='#AAAAAA', fontsize=8)
    ax_put3d.set_zlabel('Put Price',   color='#AAAAAA', fontsize=8)
    ax_put3d.tick_params(colors='#AAAAAA', labelsize=7)

    # P&L at expiration
    call_pnl = np.maximum(stock_range - K, 0) - call_price
    put_pnl  = np.maximum(K - stock_range, 0) - put_price

    ax_pnl.plot(stock_range, call_pnl, color=C_CALL, linewidth=2.0, label=f'Call P&L (cost ${call_price:.2f})')
    ax_pnl.plot(stock_range, put_pnl,  color=C_PUT,  linewidth=2.0, label=f'Put P&L  (cost ${put_price:.2f})')
    ax_pnl.axhline(0, color='white', linewidth=0.8, linestyle='--')
    ax_pnl.axvline(S, color=C_NEUTRAL, linewidth=1.0, linestyle=':', label=f'Current price (${S:.2f})')
    ax_pnl.axvline(K, color='white',   linewidth=1.0, linestyle=':', label=f'Strike (${K:.2f})')
    ax_pnl.fill_between(stock_range, call_pnl, 0,
                        where=(call_pnl > 0), alpha=0.15, color=C_CALL)
    ax_pnl.fill_between(stock_range, put_pnl, 0,
                        where=(put_pnl > 0),  alpha=0.15, color=C_PUT)
    ax_pnl.set_title('P&L at Expiration', color='white', fontsize=10, fontweight='bold')
    ax_pnl.set_xlabel('Stock Price at Expiry', color='#AAAAAA', fontsize=9)
    ax_pnl.set_ylabel('Profit / Loss ($)',     color='#AAAAAA', fontsize=9)
    ax_pnl.legend(fontsize=8, framealpha=0.3, labelcolor='white',
                  facecolor='#1A1A1A', edgecolor='#333333')
    ax_pnl.grid(color='#222222', linewidth=0.5)
    ax_pnl.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:.2f}'))
    ax_pnl.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:.0f}'))

    # Delta vs stock price
    call_deltas = [greeks(s, K, T, r, sigma, 'call')['delta'] for s in stock_range]
    put_deltas  = [greeks(s, K, T, r, sigma, 'put')['delta']  for s in stock_range]

    ax_delta.plot(stock_range, call_deltas, color=C_CALL, linewidth=2.0, label='Call Delta')
    ax_delta.plot(stock_range, put_deltas,  color=C_PUT,  linewidth=2.0, label='Put Delta')
    ax_delta.axvline(S, color=C_NEUTRAL, linewidth=1.0, linestyle=':')
    ax_delta.axvline(K, color='white',   linewidth=1.0, linestyle=':')
    ax_delta.set_title('Delta vs Stock Price', color='white', fontsize=10, fontweight='bold')
    ax_delta.set_xlabel('Stock Price', color='#AAAAAA', fontsize=9)
    ax_delta.set_ylabel('Delta',       color='#AAAAAA', fontsize=9)
    ax_delta.legend(fontsize=8, framealpha=0.3, labelcolor='white',
                    facecolor='#1A1A1A', edgecolor='#333333')
    ax_delta.grid(color='#222222', linewidth=0.5)
    ax_delta.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:.0f}'))

    # Theta vs days remaining
    days_range  = np.linspace(1, 365, 300)
    call_thetas = [greeks(S, K, d/365, r, sigma, 'call')['theta'] for d in days_range]
    put_thetas  = [greeks(S, K, d/365, r, sigma, 'put')['theta']  for d in days_range]

    ax_theta.plot(days_range, call_thetas, color=C_CALL, linewidth=2.0, label='Call Theta')
    ax_theta.plot(days_range, put_thetas,  color=C_PUT,  linewidth=2.0, label='Put Theta')
    ax_theta.axvline(T_days, color=C_NEUTRAL, linewidth=1.0, linestyle=':',
                     label=f'Your expiry ({T_days}d)')
    ax_theta.set_title('Theta — Time Decay vs Days Left', color='white', fontsize=10, fontweight='bold')
    ax_theta.set_xlabel('Days to Expiration', color='#AAAAAA', fontsize=9)
    ax_theta.set_ylabel('Theta ($/day)',      color='#AAAAAA', fontsize=9)
    ax_theta.legend(fontsize=8, framealpha=0.3, labelcolor='white',
                    facecolor='#1A1A1A', edgecolor='#333333')
    ax_theta.grid(color='#222222', linewidth=0.5)

    # Vega vs volatility
    vol_range  = np.linspace(0.01, 1.5, 300)
    call_vegas = [greeks(S, K, T, r, v, 'call')['vega'] for v in vol_range]

    ax_vega.plot(vol_range * 100, call_vegas, color=C_BAND, linewidth=2.0)
    ax_vega.axvline(sigma * 100, color=C_NEUTRAL, linewidth=1.0, linestyle=':',
                    label=f'Current vol ({sigma*100:.1f}%)')
    ax_vega.set_title('Vega — Sensitivity to Volatility', color='white', fontsize=10, fontweight='bold')
    ax_vega.set_xlabel('Volatility (%)',  color='#AAAAAA', fontsize=9)
    ax_vega.set_ylabel('Vega ($ per 1%)', color='#AAAAAA', fontsize=9)
    ax_vega.legend(fontsize=8, framealpha=0.3, labelcolor='white',
                   facecolor='#1A1A1A', edgecolor='#333333')
    ax_vega.grid(color='#222222', linewidth=0.5)
    ax_vega.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}%'))

    plt.suptitle(f'Black-Scholes Options Pricing Dashboard — {ticker_symbol}   |   Strike ${K:.2f}   |   {T_days}d expiry',
                 fontsize=13, fontweight='bold', color='white', y=1.01)

    plt.subplots_adjust(left=0.07, right=0.97, top=0.92, bottom=0.08)
    plt.show()