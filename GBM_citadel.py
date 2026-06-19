import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle
from matplotlib.ticker import AutoMinorLocator
import yfinance as yf

# ══════════════════════════════════════════════════════════════════════════════
# CITADEL TRADING DESK PALETTE
# ══════════════════════════════════════════════════════════════════════════════
BG_PRIMARY    = '#0B0E14'   # near-black navy — main background
BG_PANEL      = '#0F1520'   # slightly lighter navy — panel/axes background
BG_CARD       = '#131C2E'   # stat card background
GRID_COLOR    = '#1C2840'   # subtle grid lines
SPINE_COLOR   = '#253352'   # axis border

C_BULL        = '#00E5C0'   # teal-green  — bull / upside
C_BEAR        = '#FF3B5C'   # hot red     — bear / downside
C_MEDIAN      = '#F5C842'   # amber gold  — median
C_BAND_FILL   = '#2A6AFF'   # electric blue — confidence band fill
C_PATH        = '#1E3A6E'   # muted navy path
C_START       = '#B0C4DE'   # steel blue  — starting price
C_VAR         = '#FF8C00'   # dark orange — VaR line
C_TEXT        = '#D4E0F0'   # primary text
C_TEXT_DIM    = '#5A7A9F'   # dimmed label text
C_ACCENT      = '#3A8FFF'   # accent blue

FONT_MONO     = 'monospace'
TRADING_DAYS  = 252

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL STYLE SETUP
# ══════════════════════════════════════════════════════════════════════════════
plt.rcParams.update({
    'font.family':        'monospace',
    'axes.labelcolor':    C_TEXT_DIM,
    'xtick.color':        C_TEXT_DIM,
    'ytick.color':        C_TEXT_DIM,
    'xtick.labelsize':    8,
    'ytick.labelsize':    8,
    'axes.titlesize':     10,
    'axes.titleweight':   'bold',
    'axes.titlecolor':    C_TEXT,
    'figure.facecolor':   BG_PRIMARY,
    'axes.facecolor':     BG_PANEL,
    'axes.edgecolor':     SPINE_COLOR,
    'axes.grid':          True,
    'grid.color':         GRID_COLOR,
    'grid.linewidth':     0.5,
    'grid.alpha':         1.0,
    'legend.framealpha':  0.25,
    'legend.edgecolor':   SPINE_COLOR,
    'legend.facecolor':   BG_CARD,
    'legend.labelcolor':  C_TEXT,
    'legend.fontsize':    8,
})

def style_ax(ax):
    """Apply Citadel panel styling to an axis."""
    ax.set_facecolor(BG_PANEL)
    ax.tick_params(which='both', colors=C_TEXT_DIM, length=3, width=0.6)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(which='minor', length=1.5, color=SPINE_COLOR)
    for spine in ax.spines.values():
        spine.set_edgecolor(SPINE_COLOR)
        spine.set_linewidth(0.8)
    ax.grid(which='major', color=GRID_COLOR,  linewidth=0.5)
    ax.grid(which='minor', color=BG_PANEL,    linewidth=0.3, alpha=0.5)

def header_bar(fig, y_pos, text):
    """Draw a thin accent rule with a label — Bloomberg-style section header."""
    fig.text(0.5, y_pos, text, ha='center', va='center',
             fontsize=9, color=C_ACCENT, fontfamily=FONT_MONO,
             fontweight='bold')

# ══════════════════════════════════════════════════════════════════════════════
# DATA COLLECTION
# ══════════════════════════════════════════════════════════════════════════════
ticker_symbol = input("Enter ticker symbol (e.g. AAPL): ").upper().strip()
ticker = yf.Ticker(ticker_symbol)
hist   = ticker.history(period="1y")

if hist.empty:
    print(f"[ERR] Could not retrieve data for {ticker_symbol}. Verify symbol.")
else:
    closes   = hist['Close']
    log_rets = np.log(closes / closes.shift(1)).dropna()
    sig      = log_rets.std() * np.sqrt(TRADING_DAYS)
    SO       = closes.iloc[-1]

    print(f"\n  ┌─ MARKET DATA ──────────────────────────┐")
    print(f"   │  Last close:       ${SO:>10.2f}       │")
    print(f"   │  Annualized vol:   {sig*100:>9.1f}%       │")
    print(f"   └───────────────────────────────────────┘\n")

    mu  = float(input("Expected annual return (e.g. 0.08 for 8%): "))
    T   = int(input("Trading days to simulate (e.g. 252 for 1 year): "))
    N   = int(input("Number of simulations (e.g. 10000): "))
    rf  = float(input("Risk-free rate (e.g. 0.05 for 5%): "))

    dt  = 1 / TRADING_DAYS

    # ══════════════════════════════════════════════════════════════════════════
    # GBM SIMULATION
    # ══════════════════════════════════════════════════════════════════════════
    Z              = np.random.standard_normal((T, N))
    daily_returns  = np.exp((mu - 0.5 * sig**2) * dt + sig * np.sqrt(dt) * Z)
    price_paths    = np.zeros((T + 1, N))
    price_paths[0] = SO
    for t in range(1, T + 1):
        price_paths[t] = price_paths[t - 1] * daily_returns[t - 1]

    final_prices = price_paths[-1]

    # Percentile bands
    p10 = np.percentile(price_paths, 10, axis=1)
    p25 = np.percentile(price_paths, 25, axis=1)
    p50 = np.percentile(price_paths, 50, axis=1)
    p75 = np.percentile(price_paths, 75, axis=1)
    p90 = np.percentile(price_paths, 90, axis=1)

    # Risk metrics
    returns_pct   = (final_prices - SO) / SO
    VaR_95        = np.percentile(returns_pct, 5) * SO
    CVaR_95       = returns_pct[returns_pct <= np.percentile(returns_pct, 5)].mean() * SO
    pct_above     = np.mean(final_prices > SO) * 100
    sharpe_approx = (mu - rf) / sig
    var_price     = SO + VaR_95

    # Stat card — Bloomberg terminal style
    col1 = (
        f"  TICKER        {ticker_symbol}\n"
        f"  ENTRY         ${SO:.2f}\n"
        f"  HORIZON       {T}d\n"
        f"  SIMS          {N:,}\n"
        f"  ─────────────────────\n"
        f"  P50 (MEDIAN)  ${p50[-1]:.2f}\n"
        f"  P90 (BULL)    ${p90[-1]:.2f}\n"
        f"  P10 (BEAR)    ${p10[-1]:.2f}\n"
        f"  PATHS > S0    {pct_above:.1f}%\n"
        f"  ─────────────────────\n"
        f"  VaR  95%     -${abs(VaR_95):.2f}\n"
        f"  CVaR 95%     -${abs(CVaR_95):.2f}\n"
        f"  SHARPE        {sharpe_approx:.2f}\n"
        f"  ANN VOL       {sig*100:.1f}%\n"
        f"  EXP RETURN    {mu*100:.1f}%"
    )

    # ══════════════════════════════════════════════════════════════════════════
    # FIGURE 1 — GBM SIMULATION DASHBOARD
    # ══════════════════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(18, 9), facecolor=BG_PRIMARY)
    fig.subplots_adjust(left=0.055, right=0.97, top=0.90, bottom=0.09, wspace=0.04)

    # Title bar
    fig.text(0.5, 0.965,
             f'MONTE CARLO SIMULATION  ·  {ticker_symbol}  ·  GBM  ·  {N:,} PATHS  ·  {T}d HORIZON',
             ha='center', va='top', fontsize=11, fontweight='bold',
             color=C_TEXT, fontfamily=FONT_MONO)
    fig.text(0.5, 0.935,
             'GEOMETRIC BROWNIAN MOTION   |   QUANTITATIVE RISK DASHBOARD',
             ha='center', va='top', fontsize=8, color=C_TEXT_DIM, fontfamily=FONT_MONO)

    # Thin accent rule under title
    line = plt.Line2D([0.04, 0.96], [0.925, 0.925],
                      transform=fig.transFigure, color=C_ACCENT,
                      linewidth=0.8, alpha=0.6)
    fig.add_artist(line)

    gs      = gridspec.GridSpec(1, 2, width_ratios=[2.4, 1], figure=fig)
    ax_sim  = fig.add_subplot(gs[0])
    ax_hist = fig.add_subplot(gs[1])

    style_ax(ax_sim)
    style_ax(ax_hist)

    # ── Simulation panel ─────────────────────────────────────────────────────
    ax_sim.plot(price_paths[:, :300], alpha=0.04, linewidth=0.4, color=C_PATH, rasterized=True)
    ax_sim.fill_between(range(T + 1), p25, p75,
                        alpha=0.12, color=C_BAND_FILL, label='IQR (25–75%)', zorder=2)
    ax_sim.plot(p90, color=C_BULL,   linewidth=1.8, linestyle='--',
                label=f'Bull P90  ${p90[-1]:,.2f}', zorder=4)
    ax_sim.plot(p50, color=C_MEDIAN, linewidth=2.0,
                label=f'Median P50  ${p50[-1]:,.2f}', zorder=5)
    ax_sim.plot(p10, color=C_BEAR,   linewidth=1.8, linestyle='--',
                label=f'Bear P10  ${p10[-1]:,.2f}', zorder=4)
    ax_sim.axhline(SO,        color=C_START, linewidth=1.0, linestyle=':',
                   label=f'Entry  ${SO:.2f}', zorder=3)
    ax_sim.axhline(var_price, color=C_VAR,   linewidth=0.9, linestyle='-.',
                   label=f'VaR 95% Floor  ${var_price:.2f}', zorder=3)

    # Stat card
    ax_sim.text(0.013, 0.975, col1,
                transform=ax_sim.transAxes, fontsize=8, verticalalignment='top',
                fontfamily=FONT_MONO, color=C_TEXT,
                bbox=dict(boxstyle='round,pad=0.55', facecolor=BG_CARD,
                          alpha=0.92, edgecolor=SPINE_COLOR, linewidth=0.8))

    ax_sim.set_xlabel('TRADING DAYS', fontsize=8, labelpad=6)
    ax_sim.set_ylabel('PRICE  ($)', fontsize=8, labelpad=6)
    ax_sim.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax_sim.legend(loc='upper right', ncol=1,
                  handlelength=1.8, handletextpad=0.6, borderpad=0.7)

    # ── Distribution panel ───────────────────────────────────────────────────
    ax_hist.hist(final_prices, bins=90, orientation='horizontal',
                 color=C_BAND_FILL, alpha=0.55, edgecolor='none', rasterized=True)

    # Percentile markers with labels on right
    for price, color, lw, ls, label in [
        (p90[-1], C_BULL,   1.4, '--', 'P90'),
        (p50[-1], C_MEDIAN, 1.8, '-',  'P50'),
        (p10[-1], C_BEAR,   1.4, '--', 'P10'),
        (var_price, C_VAR,  1.0, '-.', 'VaR'),
        (SO,      C_START,  1.0, ':',  'S0'),
    ]:
        ax_hist.axhline(price, color=color, linewidth=lw, linestyle=ls)
        ax_hist.text(ax_hist.get_xlim()[1] if ax_hist.get_xlim()[1] > 0 else 1,
                     price, f' {label}', color=color, fontsize=7,
                     va='center', fontfamily=FONT_MONO)

    ax_hist.set_xlabel('FREQUENCY', fontsize=8, labelpad=6)
    ax_hist.set_title('FINAL PRICE\nDISTRIBUTION', fontsize=9, pad=8)
    ax_hist.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax_hist.yaxis.tick_right()
    ax_hist.yaxis.set_label_position('right')
    ax_hist.tick_params(axis='y', labelcolor=C_TEXT_DIM)

    # ══════════════════════════════════════════════════════════════════════════
    # FIGURE 2 — π MONTE CARLO ESTIMATOR
    # ══════════════════════════════════════════════════════════════════════════
    N_PI       = 50_000
    x_pts      = np.random.uniform(-1, 1, N_PI)
    y_pts      = np.random.uniform(-1, 1, N_PI)
    dist       = x_pts**2 + y_pts**2
    inside     = dist <= 1.0
    pi_est     = 4 * np.sum(inside) / N_PI
    error      = abs(pi_est - np.pi)
    running_pi = 4 * np.cumsum(inside) / np.arange(1, N_PI + 1)

    fig2 = plt.figure(figsize=(15, 6.5), facecolor=BG_PRIMARY)
    fig2.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.10, wspace=0.08)

    fig2.text(0.5, 0.965,
              'MONTE CARLO METHOD  ·  ESTIMATING π  VIA RANDOM SAMPLING',
              ha='center', va='top', fontsize=11, fontweight='bold',
              color=C_TEXT, fontfamily=FONT_MONO)
    line2 = plt.Line2D([0.04, 0.96], [0.925, 0.925],
                       transform=fig2.transFigure, color=C_ACCENT,
                       linewidth=0.8, alpha=0.6)
    fig2.add_artist(line2)

    ax_scatter, ax_conv = fig2.subplots(1, 2)
    style_ax(ax_scatter)
    style_ax(ax_conv)

    # Scatter — inside vs outside unit circle
    ax_scatter.scatter(x_pts[ inside], y_pts[ inside],
                       s=0.3, color=C_BULL,  alpha=0.35, rasterized=True)
    ax_scatter.scatter(x_pts[~inside], y_pts[~inside],
                       s=0.3, color=C_BEAR,  alpha=0.20, rasterized=True)
    circle = Circle((0, 0), 1, fill=False, edgecolor=C_START, linewidth=1.0)
    ax_scatter.add_patch(circle)
    ax_scatter.set_aspect('equal')
    ax_scatter.set_title(f'π ≈ {pi_est:.5f}   |   ERROR = {error:.5f}', pad=8)
    ax_scatter.set_xlabel('x', fontsize=8)
    ax_scatter.set_ylabel('y', fontsize=8)

    pi_card = (
        f"  SAMPLES    {N_PI:,}\n"
        f"  INSIDE     {inside.sum():,}\n"
        f"  π EST      {pi_est:.6f}\n"
        f"  TRUE π     {np.pi:.6f}\n"
        f"  ERROR      {error:.6f}"
    )
    ax_scatter.text(0.02, 0.975, pi_card,
                    transform=ax_scatter.transAxes, fontsize=8,
                    verticalalignment='top', fontfamily=FONT_MONO, color=C_TEXT,
                    bbox=dict(boxstyle='round,pad=0.55', facecolor=BG_CARD,
                              alpha=0.92, edgecolor=SPINE_COLOR, linewidth=0.8))

    # Convergence plot
    ax_conv.plot(running_pi, color=C_MEDIAN, linewidth=1.0, alpha=0.9, rasterized=True)
    ax_conv.axhline(np.pi, color=C_BULL, linewidth=1.4, linestyle='--',
                    label=f'True π = {np.pi:.5f}')
    ax_conv.set_title('CONVERGENCE TO π', pad=8)
    ax_conv.set_xlabel('NUMBER OF SAMPLES', fontsize=8)
    ax_conv.set_ylabel('π ESTIMATE',        fontsize=8)
    ax_conv.legend()
    ax_conv.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))

    plt.show()
