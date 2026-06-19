import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle
import yfinance as yf

plt.style.use('dark_background')

# Color Scheme 
# ____________________________________________________________
C_BULL   = '#00FF9C'   # bright green
C_BEAR   = '#FF4C4C'   # bright red
C_MEDIAN = '#FFD700'   # gold
C_BAND   = '#4488FF'   # blue band
C_PATH   = '#3A6EA8'   # muted path color
C_START  = '#FFFFFF'

TRADING_DAYS = 252

# User inputs 
# ____________________________________________________________
ticker_symbol = input("Enter ticker symbol (Example: AAPL): ").upper().strip()
ticker = yf.Ticker(ticker_symbol)
hist   = ticker.history(period="1y")

if hist.empty:
    print(f"Could not retrieve data for {ticker_symbol}. Check the symbol.")
else:
    closes    = hist['Close']
    log_rets  = np.log(closes / closes.shift(1)).dropna()
    sig       = log_rets.std() * np.sqrt(TRADING_DAYS)   # annualized vol (252-day)
    SO        = closes.iloc[-1]

    print(f"\n  Last close:          ${SO:.2f}")
    print(f"  Annualized vol:      {sig*100:.1f}%\n")

    mu  = float(input("Expected annual return (Example: 0.08 for 8%): "))
    T   = int(input("Trading days to simulate (Example: 252 for 1 year): "))
    N   = int(input("Number of simulations (Example: 10000): "))
    rf  = float(input("Risk-free rate (Example: 0.05 for 5%): "))

    dt  = 1 / TRADING_DAYS

    # GBM Simulation 
    # ____________________________________________________________
    Z             = np.random.standard_normal((T, N))
    daily_returns = np.exp((mu - 0.5 * sig**2) * dt + sig * np.sqrt(dt) * Z)
    price_paths   = np.zeros((T + 1, N))
    price_paths[0] = SO
    for t in range(1, T + 1):
        price_paths[t] = price_paths[t - 1] * daily_returns[t - 1]

    final_prices = price_paths[-1]

    # Percentile bands
    # ______________________________________________________________ 
    p10 = np.percentile(price_paths, 10, axis=1)
    p25 = np.percentile(price_paths, 25, axis=1)
    p50 = np.percentile(price_paths, 50, axis=1)
    p75 = np.percentile(price_paths, 75, axis=1)
    p90 = np.percentile(price_paths, 90, axis=1)

    # Risk metrics 
    # _____________________________________________________________
    returns_pct    = (final_prices - SO) / SO
    VaR_95         = np.percentile(returns_pct, 5) * SO          # dollar VaR
    CVaR_95        = returns_pct[returns_pct <= np.percentile(returns_pct, 5)].mean() * SO
    pct_above      = np.mean(final_prices > SO) * 100
    sharpe_approx  = (mu - rf) / sig

    stats_text = (
        f"  Ticker:            {ticker_symbol}\n"
        f"  Starting price:    ${SO:.2f}\n"
        f"  Median outcome:    ${p50[-1]:.2f}\n"
        f"  Bull case (90th):  ${p90[-1]:.2f}\n"
        f"  Bear case (10th):  ${p10[-1]:.2f}\n"
        f"  Paths above start: {pct_above:.1f}%\n"
        f"  ──────────────────────────\n"
        f"  VaR (95%):        -${abs(VaR_95):.2f}\n"
        f"  CVaR (95%):       -${abs(CVaR_95):.2f}\n"
        f"  Approx. Sharpe:    {sharpe_approx:.2f}\n"
        f"  Annualized vol:    {sig*100:.1f}%"
    )
 
    # FIGURE 1 — GBM Simulation
    # ____________________________________________________________
    fig = plt.figure(figsize=(16, 9), facecolor='#0D0D0D')
    gs  = gridspec.GridSpec(1, 2, width_ratios=[2.2, 1], wspace=0.06)

    ax_sim  = fig.add_subplot(gs[0])
    ax_hist = fig.add_subplot(gs[1])

    for ax in [ax_sim, ax_hist]:
        ax.set_facecolor('#0D0D0D')
        ax.tick_params(colors='#AAAAAA')
        for spine in ax.spines.values():
            spine.set_edgecolor('#333333')

    # Left panel: price paths 
    # ___________________________________________________________
    ax_sim.plot(price_paths[:, :200], alpha=0.08, linewidth=0.5, color=C_PATH)
    ax_sim.fill_between(range(T + 1), p25, p75,
                        alpha=0.18, color=C_BAND, label='50% confidence band')
    ax_sim.plot(p90, color=C_BULL,   linewidth=2.0, linestyle='--', label='Bull case (90th %)')
    ax_sim.plot(p50, color=C_MEDIAN, linewidth=2.2,                 label='Median path (50th %)')
    ax_sim.plot(p10, color=C_BEAR,   linewidth=2.0, linestyle='--', label='Bear case (10th %)')
    ax_sim.axhline(SO, color=C_START, linewidth=1.2, linestyle=':',
                   label=f'Starting price (${SO:.2f})')

    # VaR price level line
    var_price = SO + VaR_95
    ax_sim.axhline(var_price, color='orange', linewidth=1.0, linestyle='-.',
                   label=f'VaR 95% floor (${var_price:.2f})')

    ax_sim.text(0.01, 0.98, stats_text,
                transform=ax_sim.transAxes, fontsize=8.5,
                verticalalignment='top', fontfamily='monospace',
                color='#DDDDDD',
                bbox=dict(boxstyle='round,pad=0.6', facecolor='#1A1A1A',
                          alpha=0.85, edgecolor='#444444'))

    ax_sim.set_title(f'Monte Carlo Simulation: {ticker_symbol}   |   GBM   |   {N:,} paths   |   {T}d horizon',
                     fontsize=12, fontweight='bold', color='white', pad=12)
    ax_sim.set_xlabel('Trading Days', color='#AAAAAA', fontsize=10)
    ax_sim.set_ylabel('Price ($)', color='#AAAAAA', fontsize=10)
    ax_sim.legend(loc='upper right', fontsize=8.5, framealpha=0.3,
                  labelcolor='white', facecolor='#1A1A1A', edgecolor='#333333')
    ax_sim.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax_sim.grid(axis='y', color='#222222', linewidth=0.6)

    # Right panel: final price distribution 
    #_____________________________________________________________
    ax_hist.hist(final_prices, bins=80, orientation='horizontal',
                 color=C_BAND, alpha=0.7, edgecolor='none')
    ax_hist.axhline(p50[-1],   color=C_MEDIAN, linewidth=1.8, linestyle='-')
    ax_hist.axhline(p90[-1],   color=C_BULL,   linewidth=1.4, linestyle='--')
    ax_hist.axhline(p10[-1],   color=C_BEAR,   linewidth=1.4, linestyle='--')
    ax_hist.axhline(var_price, color='orange',  linewidth=1.2, linestyle='-.')
    ax_hist.axhline(SO,        color=C_START,   linewidth=1.2, linestyle=':')

    ax_hist.set_xlabel('Frequency', color='#AAAAAA', fontsize=9)
    ax_hist.set_title('Final Price\nDistribution', fontsize=10,
                       color='white', fontweight='bold')
    ax_hist.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax_hist.yaxis.tick_right()
    ax_hist.tick_params(axis='y', labelcolor='#AAAAAA')
    ax_hist.grid(axis='x', color='#222222', linewidth=0.6)

    plt.suptitle('Geometric Brownian Motion   |   Quantitative Risk Dashboard',
                 fontsize=11, color='#666666', y=0.01)
    plt.subplots_adjust(left=0.06, right=0.97, top=0.93, bottom=0.08,wspace=0.06)

    # ____________________________________________________________
    # FIGURE 2 — π Monte Carlo Estimator
    # ____________________________________________________________
    N_PI     = 50_000
    x_pts    = np.random.uniform(-1, 1, N_PI)
    y_pts    = np.random.uniform(-1, 1, N_PI)
    dist     = x_pts**2 + y_pts**2
    inside   = dist <= 1.0
    pi_est   = 4 * np.sum(inside) / N_PI
    error    = abs(pi_est - np.pi)

    # Running estimate convergence
    running_pi = 4 * np.cumsum(inside) / np.arange(1, N_PI + 1)

    fig2, (ax_scatter, ax_conv) = plt.subplots(1, 2, figsize=(14, 6),
                                                facecolor='#0D0D0D')
    for ax in [ax_scatter, ax_conv]:
        ax.set_facecolor('#0D0D0D')
        ax.tick_params(colors='#AAAAAA')
        for spine in ax.spines.values():
            spine.set_edgecolor('#333333')

    # Scatter
    ax_scatter.scatter(x_pts[ inside], y_pts[ inside], s=0.4, color=C_BULL,  alpha=0.5)
    ax_scatter.scatter(x_pts[~inside], y_pts[~inside], s=0.4, color=C_BEAR,  alpha=0.3)
    circle = Circle((0, 0), 1, fill=False, edgecolor='white', linewidth=1.2)
    ax_scatter.add_patch(circle)
    ax_scatter.set_aspect('equal')
    ax_scatter.set_title(f'π Monte Carlo Estimator   |   π ≈ {pi_est:.5f}   |   error = {error:.5f}',
                          fontsize=11, fontweight='bold', color='white')
    ax_scatter.set_xlabel('x', color='#AAAAAA')
    ax_scatter.set_ylabel('y', color='#AAAAAA')
    ax_scatter.text(0.02, 0.97,
                    f"Points sampled:   {N_PI:,}\nInside circle:    {inside.sum():,}\nπ estimate:       {pi_est:.6f}\nTrue π:           {np.pi:.6f}\nError:            {error:.6f}",
                    transform=ax_scatter.transAxes, fontsize=9, verticalalignment='top',
                    fontfamily='monospace', color='#DDDDDD',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='#1A1A1A',
                              alpha=0.85, edgecolor='#444444'))

    # Convergence
    ax_conv.plot(running_pi, color=C_MEDIAN, linewidth=1.2, alpha=0.9)
    ax_conv.axhline(np.pi, color=C_BULL, linewidth=1.5, linestyle='--', label=f'True π = {np.pi:.5f}')
    ax_conv.set_title('Convergence to π', fontsize=11, fontweight='bold', color='white')
    ax_conv.set_xlabel('Number of Samples', color='#AAAAAA', fontsize=10)
    ax_conv.set_ylabel('π Estimate',        color='#AAAAAA', fontsize=10)
    ax_conv.legend(fontsize=9, framealpha=0.3, labelcolor='white',
                   facecolor='#1A1A1A', edgecolor='#333333')
    ax_conv.grid(color='#222222', linewidth=0.5)
    ax_conv.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))

    plt.suptitle('Monte Carlo Method — Estimating π via Random Sampling',
                 fontsize=12, color='white', fontweight='bold', y=1.01)
    plt.subplots_adjust(left=0.07, right=0.97, top=0.93, bottom=0.08, wspace=0.06)
    plt.show()