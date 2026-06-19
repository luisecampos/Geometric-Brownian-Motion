"""
WTI Crude Oil — Ornstein-Uhlenbeck Mean-Reversion Model
=========================================================
Models WTI crude oil prices using the Ornstein-Uhlenbeck (O-U) stochastic
process and compares simulated forecasts against a synthetic futures strip.

The O-U process is the industry-standard model for commodity prices because,
unlike GBM, it captures mean-reversion — the tendency of prices to be pulled
back toward a long-run equilibrium driven by supply/demand fundamentals.

SDE:  dS = κ(μ - S)dt + σ dW

Parameters:
    κ  (kappa)  — mean-reversion speed (how fast price snaps back to μ)
    μ  (mu)     — long-run equilibrium price
    σ  (sigma)  — instantaneous volatility

Author:  Luis (Energy IB Recruiting Project)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

# ── 1. MODEL PARAMETERS ──────────────────────────────────────────────────────

S0    = 72.0    # Spot price today ($/bbl)
mu    = 72.0    # Long-run mean / equilibrium price ($/bbl)
kappa = 0.40    # Mean-reversion speed (per year); higher = faster reversion
sigma = 0.30    # Annual volatility (30%)

T     = 2.0     # Forecast horizon (years)
dt    = 1/252   # Daily time step (trading days)
N     = int(T / dt)   # Number of steps
n_paths = 500   # Monte Carlo paths

np.random.seed(42)  # Reproducibility


# ── 2. SIMULATE O-U PATHS ────────────────────────────────────────────────────

def simulate_ou(S0, mu, kappa, sigma, T, dt, n_paths):
    """
    Simulate n_paths of the Ornstein-Uhlenbeck process using the
    exact discrete-time solution (no Euler approximation error):

        S(t+dt) = S(t)*e^(-κdt) + μ(1 - e^(-κdt)) + σ*sqrt((1-e^(-2κdt))/(2κ)) * Z

    Returns: paths array of shape (N+1, n_paths)
    """
    N = int(T / dt)
    paths = np.zeros((N + 1, n_paths))
    paths[0] = S0

    e_kdt   = np.exp(-kappa * dt)
    std_dev  = sigma * np.sqrt((1 - np.exp(-2 * kappa * dt)) / (2 * kappa))

    for t in range(1, N + 1):
        Z = np.random.standard_normal(n_paths)
        paths[t] = paths[t-1] * e_kdt + mu * (1 - e_kdt) + std_dev * Z

    return paths


paths = simulate_ou(S0, mu, kappa, sigma, T, dt, n_paths)
time  = np.linspace(0, T, N + 1)   # Time axis in years


# ── 3. SUMMARY STATISTICS ────────────────────────────────────────────────────

mean_path   = paths.mean(axis=1)
pct_5       = np.percentile(paths, 5,  axis=1)
pct_25      = np.percentile(paths, 25, axis=1)
pct_75      = np.percentile(paths, 75, axis=1)
pct_95      = np.percentile(paths, 95, axis=1)

# Analytical O-U mean and variance at each time step
analytical_mean = mu + (S0 - mu) * np.exp(-kappa * time)
analytical_var  = (sigma**2 / (2 * kappa)) * (1 - np.exp(-2 * kappa * time))
analytical_std  = np.sqrt(analytical_var)


# ── 4. SYNTHETIC FUTURES STRIP ───────────────────────────────────────────────
#
# In practice you'd pull this from CME Group or Bloomberg.
# Here we build a realistic synthetic strip: slight contango near-term,
# then modest backwardation further out — typical WTI structure.

futures_months = np.array([1, 2, 3, 6, 9, 12, 18, 24])          # months out
futures_prices = np.array([72.50, 72.80, 73.10, 73.40,
                            72.90, 72.20, 71.50, 71.00])          # $/bbl
futures_times  = futures_months / 12                              # convert to years

# O-U model "fair value" at each futures tenor
ou_futures_fair = mu + (S0 - mu) * np.exp(-kappa * futures_times)


# ── 5. ANALYTICS PRINTOUT ────────────────────────────────────────────────────

print("=" * 60)
print("  WTI CRUDE OIL — O-U MEAN REVERSION MODEL")
print("=" * 60)
print(f"\n  Model Parameters")
print(f"  {'Spot price (S0):':<30} ${S0:.2f}/bbl")
print(f"  {'Long-run mean (μ):':<30} ${mu:.2f}/bbl")
print(f"  {'Mean-reversion speed (κ):':<30} {kappa:.2f} per year")
print(f"  {'Volatility (σ):':<30} {sigma*100:.0f}% annualized")
print(f"  {'Forecast horizon:':<30} {T:.0f} years")
print(f"  {'Simulated paths:':<30} {n_paths:,}")

print(f"\n  Forecast Summary (end of {T:.0f}Y)")
print(f"  {'Simulated mean:':<30} ${mean_path[-1]:.2f}/bbl")
print(f"  {'Analytical mean:':<30} ${analytical_mean[-1]:.2f}/bbl")
print(f"  {'5th percentile:':<30} ${pct_5[-1]:.2f}/bbl")
print(f"  {'95th percentile:':<30} ${pct_95[-1]:.2f}/bbl")

print(f"\n  Futures Strip vs. O-U Fair Value")
print(f"  {'Tenor':<12} {'Futures ($)':<16} {'O-U Fair ($)':<16} {'Spread ($)'}")
print(f"  {'-'*54}")
for m, fp, oup in zip(futures_months, futures_prices, ou_futures_fair):
    spread = fp - oup
    print(f"  {f'{m}M':<12} ${fp:<15.2f} ${oup:<15.2f} {spread:+.2f}")

print("\n" + "=" * 60)


# ── 6. VISUALIZATION ─────────────────────────────────────────────────────────

fig = plt.figure(figsize=(14, 10), facecolor="#0f1117")
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.30)

# ── Color palette ──
C_TEAL   = "#1aefbc"
C_AMBER  = "#f5a623"
C_CORAL  = "#f06060"
C_GRAY   = "#8a8fa8"
C_WHITE  = "#e8eaf0"
BG       = "#0f1117"
CARD_BG  = "#181c28"

def style_ax(ax, title):
    ax.set_facecolor(CARD_BG)
    ax.tick_params(colors=C_GRAY, labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a2e42")
    ax.set_title(title, color=C_WHITE, fontsize=11, fontweight="bold", pad=10)
    ax.yaxis.label.set_color(C_GRAY)
    ax.xaxis.label.set_color(C_GRAY)


# ── Panel 1: Simulated paths + confidence bands ───────────────────────────
ax1 = fig.add_subplot(gs[0, :])
style_ax(ax1, "WTI O-U Simulation — 500 Paths with Confidence Bands")

# Draw 80 faint sample paths
for i in range(80):
    ax1.plot(time, paths[:, i], color=C_TEAL, alpha=0.06, linewidth=0.6)

# Confidence bands
ax1.fill_between(time, pct_5,  pct_95, color=C_TEAL,  alpha=0.10, label="5–95th pct")
ax1.fill_between(time, pct_25, pct_75, color=C_TEAL,  alpha=0.18, label="25–75th pct")

# Mean path and long-run mean
ax1.plot(time, mean_path,      color=C_TEAL,  linewidth=2.0, label="Simulated mean")
ax1.plot(time, analytical_mean,color=C_AMBER, linewidth=1.5,
         linestyle="--", label="Analytical O-U mean")
ax1.axhline(mu, color=C_GRAY,  linewidth=1.0, linestyle=":",  label=f"Long-run mean μ = ${mu}")
ax1.axhline(S0, color=C_WHITE, linewidth=0.8, linestyle=":",  alpha=0.4)

# Futures strip overlay
ax1.scatter(futures_times, futures_prices, color=C_AMBER,
            zorder=5, s=60, label="Futures strip (CME)")
ax1.plot(futures_times, futures_prices, color=C_AMBER,
         linewidth=1.2, linestyle="-", alpha=0.7)

ax1.set_xlabel("Time (years)")
ax1.set_ylabel("WTI Price ($/bbl)")
ax1.legend(fontsize=8, framealpha=0.2, labelcolor=C_WHITE, loc="upper right")
ax1.set_xlim(0, T)


# ── Panel 2: Futures strip vs. O-U fair value ─────────────────────────────
ax2 = fig.add_subplot(gs[1, 0])
style_ax(ax2, "Futures Strip vs. O-U Fair Value")

x = np.arange(len(futures_months))
w = 0.35
bars1 = ax2.bar(x - w/2, futures_prices,  width=w, color=C_AMBER, alpha=0.85, label="Futures")
bars2 = ax2.bar(x + w/2, ou_futures_fair, width=w, color=C_TEAL,  alpha=0.85, label="O-U Fair Value")

ax2.set_xticks(x)
ax2.set_xticklabels([f"{m}M" for m in futures_months])
ax2.set_ylabel("Price ($/bbl)")
ax2.legend(fontsize=8, framealpha=0.2, labelcolor=C_WHITE)
ax2.set_ylim(min(futures_prices.min(), ou_futures_fair.min()) - 2,
             max(futures_prices.max(), ou_futures_fair.max()) + 2)


# ── Panel 3: Spread (futures − O-U fair value) ────────────────────────────
ax3 = fig.add_subplot(gs[1, 1])
style_ax(ax3, "Futures vs. O-U Spread ($/bbl)")

spreads = futures_prices - ou_futures_fair
colors  = [C_TEAL if s >= 0 else C_CORAL for s in spreads]
ax3.bar([f"{m}M" for m in futures_months], spreads, color=colors, alpha=0.85)
ax3.axhline(0, color=C_GRAY, linewidth=0.8, linestyle="--")
ax3.set_ylabel("Spread = Futures − O-U ($/bbl)")

for i, (label, val) in enumerate(zip([f"{m}M" for m in futures_months], spreads)):
    ax3.text(i, val + (0.05 if val >= 0 else -0.12),
             f"{val:+.2f}", ha="center", va="bottom", fontsize=8, color=C_WHITE)

# ── Title ──
fig.suptitle("WTI Crude Oil  |  Ornstein-Uhlenbeck Mean-Reversion Model",
             color=C_WHITE, fontsize=14, fontweight="bold", y=0.98)

plt.savefig("wti_ou_model_chart.png",
            dpi=150, bbox_inches="tight", facecolor=BG)
plt.show()
print("\n  Chart saved → wti_ou_model_chart.png")
