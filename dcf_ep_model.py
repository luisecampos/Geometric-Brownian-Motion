"""
E&P DCF + Sensitivity Model  —  Universal E&P Edition
Author : Luis Campos — Prysmix Equity Research
Usage  : python dcf_ep_model.py
         → prompts for ticker, pulls fundamentals live, asks only for
           operating inputs that cannot be derived from market data.
"""

# ─────────────────────────────────────────────
# SECTION 1: IMPORTS
# ─────────────────────────────────────────────

import os
import sys
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# SECTION 2: E&P TICKER REGISTRY
# ─────────────────────────────────────────────

# Curated list of U.S.-listed E&P companies (upstream oil & gas producers).
# Midstream, refiners, and oilfield services are intentionally excluded —
# they have fundamentally different valuation frameworks (EV/EBITDA multiples,
# volume-based fee contracts) that are incompatible with a reserve-based DCF.
EP_TICKERS = {
    # ── Large-cap Permian / diversified E&P ──
    "DVN":  "Devon Energy",
    "PXD":  "Pioneer Natural Resources",
    "COP":  "ConocoPhillips",
    "EOG":  "EOG Resources",
    "OXY":  "Occidental Petroleum",
    "CVX":  "Chevron",
    "XOM":  "ExxonMobil",
    "APA":  "APA Corporation",
    "FANG": "Diamondback Energy",
    "MTDR": "Matador Resources",
    "CTRA": "Coterra Energy",
    "MRO":  "Marathon Oil",
    "HES":  "Hess Corporation",
    # ── Mid-cap E&P ──
    "CHRD": "Chord Energy",
    "SM":   "SM Energy",
    "CIVI": "Civitas Resources",
    "PR":   "Permian Resources",
    "PDCE": "PDC Energy",
    "CPE":  "Callon Petroleum",
    "REI":  "Ring Energy",
    "MGY":  "Magnolia Oil & Gas",
    "NOG":  "Northern Oil and Gas",
    "BATL": "Battalion Oil",
    # ── Offshore / deepwater E&P ──
    "RIG":  "Transocean",
    "VAL":  "Valaris",
    "SLB":  "SLB (Schlumberger)",
    "CNQ":  "Canadian Natural Resources",
    "CVE":  "Cenovus Energy",
    # ── Natural gas focused ──
    "EQT":  "EQT Corporation",
    "AR":   "Antero Resources",
    "SWN":  "Southwestern Energy",
    "RRC":  "Range Resources",
    "GPOR": "Gulfport Energy",
    "CNX":  "CNX Resources",
    # ── International E&P ──
    "E":    "Eni SpA",
    "BP":   "BP plc",
    "SHEL": "Shell plc",
    "TTE":  "TotalEnergies",
    "PBR":  "Petrobras",
    "EC":   "Ecopetrol",
}

# Sensible production defaults by company size tier (MBoe/d).
# Used only if the user skips manual entry.
# Source: most recent public guidance / 10-K filings.
PRODUCTION_DEFAULTS = {
    "DVN": 650, "PXD": 720, "COP": 1800, "EOG": 950, "OXY": 1200,
    "CVX": 3300, "XOM": 3700, "APA": 360, "FANG": 460, "MTDR": 175,
    "CTRA": 640, "MRO": 390, "HES": 460, "CHRD": 290, "SM": 160,
    "CIVI": 200, "PR": 280, "EQT": 600, "AR": 340, "SWN": 420,
    "RRC": 220, "CNX": 160, "GPOR": 130, "CPE": 110, "REI": 20,
    "MGY": 85,  "NOG": 120, "BP": 2300, "SHEL": 2900, "TTE": 2400,
    "PBR": 2800, "EC": 700, "CNQ": 1400, "CVE": 800, "E": 1600,
}

# Oil/gas production mix by ticker (oil fraction, 0–1).
# The remainder is treated as gas/NGL.
OIL_MIX_DEFAULTS = {
    "DVN": 0.60, "PXD": 0.75, "COP": 0.52, "EOG": 0.70, "OXY": 0.65,
    "CVX": 0.55, "XOM": 0.53, "APA": 0.62, "FANG": 0.78, "MTDR": 0.65,
    "CTRA": 0.55, "MRO": 0.62, "HES": 0.68, "CHRD": 0.72, "SM": 0.68,
    "CIVI": 0.70, "PR": 0.72, "EQT": 0.05, "AR": 0.20, "SWN": 0.08,
    "RRC": 0.18, "CNX": 0.10, "GPOR": 0.12, "CPE": 0.68, "MGY": 0.65,
    "BP": 0.55, "SHEL": 0.50, "TTE": 0.52, "PBR": 0.72, "EC": 0.70,
    "CNQ": 0.60, "CVE": 0.62, "E": 0.54, "NOG": 0.65, "REI": 0.75,
}


# ─────────────────────────────────────────────
# SECTION 3: USER INPUT INTERFACE
# ─────────────────────────────────────────────

def prompt_ticker() -> str:
    """
    Ask the user for an E&P ticker and validate it against the registry.
    Loops until a valid ticker is entered or the user types 'list' to see options.
    """
    print("\n" + "=" * 62)
    print("  E&P DCF + SENSITIVITY MODEL — Universal Edition")
    print("  Prysmix Equity Research  |  Luis Campos")
    print("=" * 62)
    print("\n  Supported universe: U.S.-listed E&P upstream producers.")
    print("  Type  LIST  to see all supported tickers.\n")

    while True:
        raw = input("  Enter ticker symbol: ").strip().upper()

        if raw == "LIST":
            print("\n  ── Supported E&P Tickers ──")
            for t, name in sorted(EP_TICKERS.items()):
                print(f"    {t:<6}  {name}")
            print()
            continue

        if raw in EP_TICKERS:
            print(f"\n  ✓  {raw} — {EP_TICKERS[raw]}")
            return raw

        # Soft match: let the user proceed with an unrecognized ticker
        # (useful for newly listed companies not yet in the registry)
        print(f"\n  '{raw}' is not in the verified E&P registry.")
        confirm = input("  Proceed anyway? This model is calibrated for upstream E&P only. (y/n): ").strip().lower()
        if confirm == "y":
            EP_TICKERS[raw] = raw   # Register it temporarily
            PRODUCTION_DEFAULTS[raw] = None
            OIL_MIX_DEFAULTS[raw]    = None
            return raw
        print()


def _ask_float(prompt: str, default: float, lo: float, hi: float) -> float:
    """
    Prompt for a float with a pre-filled default. Validates range.
    Pressing Enter accepts the default without typing anything.
    """
    while True:
        raw = input(f"  {prompt} [{default}]: ").strip()
        if raw == "":
            return default
        try:
            val = float(raw)
            if lo <= val <= hi:
                return val
            print(f"    ✗  Must be between {lo} and {hi}. Try again.")
        except ValueError:
            print("    ✗  Please enter a number.")


def collect_operating_inputs(ticker: str) -> dict:
    """
    Gather the operating assumptions that CANNOT be derived from market data.
    Everything yfinance can supply is pulled automatically in fetch_live_data().

    The only inputs we ask for:
      1. Production (MBoe/d)  — from 10-K / earnings guidance
      2. Oil/gas mix          — from 10-K production breakdown
      3. Production growth    — analyst estimate or company guidance
      4. WACC override        — optional; model computes a default from beta

    All other cost-structure assumptions use sector-calibrated defaults that
    can be reviewed and changed in the COST_PROFILES dict below.
    """
    default_prod    = PRODUCTION_DEFAULTS.get(ticker) or 300
    default_oil_mix = OIL_MIX_DEFAULTS.get(ticker)   or 0.60

    print("\n  ── Operating Inputs (press Enter to accept defaults) ──\n")
    prod     = _ask_float("Current production (MBoe/d)", default_prod, 1, 10_000)
    oil_mix  = _ask_float("Oil/NGL fraction of production  (0.0–1.0)", default_oil_mix, 0.0, 1.0)
    growth   = _ask_float("Annual production growth rate   (e.g. 0.03 = 3%)", 0.03, -0.10, 0.30)

    print("\n  ── Commodity Assumptions ──\n")
    oil_base = _ask_float("Base case WTI oil price  ($/bbl)", 72.0, 20.0, 200.0)
    gas_base = _ask_float("Base case Henry Hub gas  ($/mcf)", 2.50,  0.5,  20.0)

    print("\n  ── Valuation Parameters ──\n")
    wacc     = _ask_float("WACC  (e.g. 0.095 = 9.5%)", 0.095, 0.04, 0.25)
    tg       = _ask_float("Terminal growth rate  (e.g. 0.025 = 2.5%)", 0.025, 0.00, 0.06)

    return {
        "prod":     prod,
        "oil_mix":  oil_mix,
        "growth":   growth,
        "oil_base": oil_base,
        "gas_base": gas_base,
        "wacc":     wacc,
        "tg":       tg,
    }


# ─────────────────────────────────────────────
# SECTION 4: LIVE FUNDAMENTAL DATA PULL
# ─────────────────────────────────────────────

def fetch_live_data(ticker: str) -> dict:
    """
    Pull market price, shares outstanding, total debt, and cash from yfinance.

    Net Debt = Total Debt - Cash & Equivalents
    This is automatically derived so the user never has to input it manually —
    balance sheet data updates every quarter via yfinance.

    Falls back to conservative estimates if the network is unavailable.
    """
    print(f"\n[DATA] Pulling live fundamentals for {ticker}...")
    try:
        stk  = yf.Ticker(ticker)
        info = stk.info

        price  = info.get("currentPrice") or info.get("regularMarketPrice") or 0.0
        mcap   = (info.get("marketCap") or 0) / 1e9
        ev     = (info.get("enterpriseValue") or 0) / 1e9

        # Shares: prefer diluted from info, fall back to basicShares
        shares = (
            info.get("sharesOutstanding") or
            info.get("impliedSharesOutstanding") or 0
        ) / 1e6  # → millions

        # Net Debt: pull from balance sheet if available
        bs = stk.balance_sheet
        if bs is not None and not bs.empty:
            def _get(row):
                matches = [c for c in bs.index if row.lower() in c.lower()]
                return float(bs.loc[matches[0]].iloc[0]) if matches else 0.0
            total_debt = _get("total debt") or _get("long term debt")
            cash       = _get("cash and cash equivalents") or _get("cash")
            net_debt   = (total_debt - cash) / 1e9
        else:
            net_debt = ev - mcap if ev and mcap else 0.0

        if not price:
            raise ValueError("No price returned from yfinance")

        print(f"  Price:          ${price:.2f}")
        print(f"  Market Cap:     ${mcap:.1f}B")
        print(f"  Enterprise Val: ${ev:.1f}B")
        print(f"  Shares Out:     {shares:.0f}M")
        print(f"  Net Debt:       ${net_debt:.2f}B")

        return {
            "price":    price,
            "mcap":     mcap,
            "ev":       ev,
            "shares_m": shares if shares > 0 else 500,
            "net_debt": net_debt,
        }

    except Exception as exc:
        print(f"  [WARN] Live pull failed ({type(exc).__name__}). Using conservative fallback.")
        return {"price": 40.0, "mcap": 20.0, "ev": 25.0, "shares_m": 500, "net_debt": 5.0}


# ─────────────────────────────────────────────
# SECTION 5: REVENUE ENGINE
# ─────────────────────────────────────────────

def calc_annual_revenue(oil_price: float, gas_price: float,
                         prod_mboed: float, oil_mix: float,
                         growth_rate: float, year: int) -> float:
    """
    Revenue = (oil volume × oil price) + (gas volume × gas price)

    All company-specific parameters are now passed as arguments —
    no global variables. This makes the function safe to call
    concurrently for multiple companies without state collisions.
    """
    gas_mix    = 1.0 - oil_mix
    prod       = prod_mboed * ((1 + growth_rate) ** year)
    annual_boe = prod * 1_000 * 365

    oil_rev = annual_boe * oil_mix * oil_price
    gas_rev = annual_boe * gas_mix * gas_price

    return (oil_rev + gas_rev) / 1e9


# ─────────────────────────────────────────────
# SECTION 6: FCF ENGINE
# ─────────────────────────────────────────────

# Cost structure profiles calibrated to company size tier.
# Large-caps run lower LOE/BOE from scale advantages; small-caps pay more.
# All figures are expressed as % of revenue.
COST_PROFILES = {
    "large":  {"loe": 0.15, "gather": 0.06, "gna": 0.03, "dda": 0.20, "capex": 0.35},
    "mid":    {"loe": 0.18, "gather": 0.07, "gna": 0.04, "dda": 0.22, "capex": 0.38},
    "small":  {"loe": 0.22, "gather": 0.09, "gna": 0.06, "dda": 0.25, "capex": 0.42},
}

def _size_tier(mcap_b: float) -> str:
    """Classify company by market cap into cost tier."""
    if mcap_b >= 20:   return "large"
    if mcap_b >= 3:    return "mid"
    return "small"

def calc_fcf(revenue_B: float, mcap_b: float, tax_rate: float = 0.21) -> dict:
    """
    Build the FCF bridge from Revenue → Unlevered Free Cash Flow.
    Cost percentages scale automatically with company size tier so that
    a small-cap E&P (higher per-BOE costs) is valued differently from
    a major (scale efficiencies).
    """
    costs  = COST_PROFILES[_size_tier(mcap_b)]

    loe      = revenue_B * costs["loe"]
    gather   = revenue_B * costs["gather"]
    gna      = revenue_B * costs["gna"]
    dda      = revenue_B * costs["dda"]
    capex    = revenue_B * costs["capex"]

    ebitda   = revenue_B - loe - gather - gna
    ebit     = ebitda - dda
    nopat    = ebit * (1 - tax_rate)
    fcf      = nopat + dda - capex

    return {
        "revenue": revenue_B, "ebitda": ebitda, "ebit": ebit,
        "nopat": nopat, "capex": capex, "dda": dda, "fcf": fcf,
        "margin": fcf / revenue_B if revenue_B > 0 else 0,
    }


# ─────────────────────────────────────────────
# SECTION 7: DCF ENGINE
# ─────────────────────────────────────────────

MODEL_YEARS = 5

def run_dcf(oil_price: float, gas_price: float, wacc: float,
            terminal_growth: float, inputs: dict, live: dict,
            label: str = "Base") -> dict:
    """
    Core DCF engine — fully parameterized, no global state.

    All company-specific values (production, shares, net debt, cost tier)
    are passed through `inputs` and `live` dicts. This means the same
    function can value any E&P company without modification.
    """
    if label:
        print(f"\n[DCF] Running scenario: {label}")

    fcf_list, pv_list = [], []

    for yr in range(1, MODEL_YEARS + 1):
        rev = calc_annual_revenue(
            oil_price, gas_price,
            inputs["prod"], inputs["oil_mix"], inputs["growth"], yr
        )
        fin    = calc_fcf(rev, live["mcap"])
        fcf    = fin["fcf"]
        pv     = fcf / ((1 + wacc) ** yr)
        fcf_list.append(fcf)
        pv_list.append(pv)
        if label:
            print(f"  Year {yr}: Rev ${rev:.2f}B  |  FCF ${fcf:.2f}B  |  PV ${pv:.2f}B")

    # Terminal value (Gordon Growth Model)
    tv    = fcf_list[-1] * (1 + terminal_growth) / (wacc - terminal_growth)
    pv_tv = tv / ((1 + wacc) ** MODEL_YEARS)

    ev          = sum(pv_list) + pv_tv
    equity_val  = ev - live["net_debt"]
    nav_share   = (equity_val * 1e9) / (live["shares_m"] * 1e6)
    tv_pct      = pv_tv / ev * 100 if ev else 0

    if label:
        print(f"\n  ── {label} Results ──")
        print(f"  PV of FCFs:        ${sum(pv_list):.2f}B")
        print(f"  PV Terminal Value: ${pv_tv:.2f}B  ({tv_pct:.1f}% of EV)")
        print(f"  Enterprise Value:  ${ev:.2f}B")
        print(f"  Net Debt:          ${live['net_debt']:.2f}B")
        print(f"  Equity Value:      ${equity_val:.2f}B")
        print(f"  NAV / Share:       ${nav_share:.2f}")

    return {
        "label": label, "oil_price": oil_price, "gas_price": gas_price,
        "wacc": wacc, "pv_fcfs": sum(pv_list), "pv_tv": pv_tv,
        "tv_pct": tv_pct, "ev": ev, "equity_value": equity_val,
        "nav_share": nav_share, "fcf_list": fcf_list,
    }


# ─────────────────────────────────────────────
# SECTION 8: SCENARIO ENGINE
# ─────────────────────────────────────────────

def run_scenarios(inputs: dict, live: dict) -> tuple:
    """
    Run Bear / Base / Bull using the user's base-case commodity assumptions
    as the anchor, then stress-test symmetrically around them.

    Bear  = oil  −$17,  gas  −$0.50,  WACC +200bps,  TG  −50bps
    Base  = user inputs as-is
    Bull  = oil  +$18,  gas  +$0.75,  WACC −150bps,  TG  +50bps

    The deltas are calibrated to represent roughly one commodity cycle
    leg, consistent with the historical WTI range of ~$55–$90.
    """
    oil_b = inputs["oil_base"]
    gas_b = inputs["gas_base"]
    wacc  = inputs["wacc"]
    tg    = inputs["tg"]

    scenarios = [
        {"name": "Bear", "oil": max(oil_b - 17, 30), "gas": max(gas_b - 0.50, 1.0),
         "wacc": wacc + 0.020, "tg": tg - 0.005},
        {"name": "Base", "oil": oil_b,     "gas": gas_b,
         "wacc": wacc,         "tg": tg},
        {"name": "Bull", "oil": oil_b + 18, "gas": gas_b + 0.75,
         "wacc": wacc - 0.015, "tg": tg + 0.005},
    ]

    results = []
    for s in scenarios:
        r = run_dcf(s["oil"], s["gas"], s["wacc"], s["tg"], inputs, live, s["name"])
        results.append(r)

    df = pd.DataFrame([{
        "Scenario":          r["label"],
        "Oil ($/bbl)":       r["oil_price"],
        "Gas ($/mcf)":       r["gas_price"],
        "WACC":              f"{r['wacc']*100:.1f}%",
        "EV ($B)":           round(r["ev"], 2),
        "Equity Val ($B)":   round(r["equity_value"], 2),
        "NAV/Share ($)":     round(r["nav_share"], 2),
        "TV % of EV":        f"{r['tv_pct']:.1f}%",
    } for r in results])

    print("\n\n── SCENARIO SUMMARY ──")
    print(df.to_string(index=False))
    return df, results


# ─────────────────────────────────────────────
# SECTION 9: TORNADO CHART
# ─────────────────────────────────────────────

def build_tornado_chart(base_result: dict, inputs: dict,
                         live: dict, ticker: str, company_name: str):
    """
    Tornado chart — shocks each variable independently vs. the base case.
    Now fully parameterized: shock ranges scale relative to the user's
    base-case assumptions rather than being DVN-hardcoded.
    """
    base_nav  = base_result["nav_share"]
    oil_b     = inputs["oil_base"]
    gas_b     = inputs["gas_base"]
    wacc_b    = inputs["wacc"]
    tg_b      = inputs["tg"]
    prod_b    = inputs["prod"]

    # Shock ranges: ±25% on commodities, ±200/150bps on rates
    shocks = [
        ("WTI Oil Price ($/bbl)",    oil_b * 0.75,  oil_b * 1.25,  "oil"),
        ("WACC (%)",                 wacc_b + 0.02, wacc_b - 0.015,"wacc"),
        (f"Henry Hub Gas ($/mcf)",   gas_b * 0.80,  gas_b * 1.30,  "gas"),
        ("Terminal Growth Rate (%)", tg_b - 0.005,  tg_b + 0.005,  "tg"),
        ("Production (MBoe/d)",      prod_b * 0.92, prod_b * 1.08, "prod"),
        ("Capex Intensity",          None,           None,          "capex"),
    ]

    low_navs, high_navs, labels = [], [], []

    for label, lo_val, hi_val, vtype in shocks:
        inp_lo = dict(inputs)
        inp_hi = dict(inputs)

        if vtype == "oil":
            r_lo = run_dcf(lo_val,  gas_b,  wacc_b, tg_b,          inp_lo, live, "")
            r_hi = run_dcf(hi_val,  gas_b,  wacc_b, tg_b,          inp_hi, live, "")
        elif vtype == "gas":
            r_lo = run_dcf(oil_b, lo_val,   wacc_b, tg_b,          inp_lo, live, "")
            r_hi = run_dcf(oil_b, hi_val,   wacc_b, tg_b,          inp_hi, live, "")
        elif vtype == "wacc":
            r_lo = run_dcf(oil_b, gas_b,    lo_val, tg_b,          inp_lo, live, "")
            r_hi = run_dcf(oil_b, gas_b,    hi_val, tg_b,          inp_hi, live, "")
        elif vtype == "tg":
            r_lo = run_dcf(oil_b, gas_b,    wacc_b, lo_val,        inp_lo, live, "")
            r_hi = run_dcf(oil_b, gas_b,    wacc_b, hi_val,        inp_hi, live, "")
        elif vtype == "prod":
            inp_lo["prod"] = lo_val
            inp_hi["prod"] = hi_val
            r_lo = run_dcf(oil_b, gas_b, wacc_b, tg_b, inp_lo, live, "")
            r_hi = run_dcf(oil_b, gas_b, wacc_b, tg_b, inp_hi, live, "")
        elif vtype == "capex":
            # Stress capex intensity ±4pp of revenue
            tier   = _size_tier(live["mcap"])
            base_c = COST_PROFILES[tier]["capex"]
            lo_c   = COST_PROFILES[tier].copy(); lo_c["capex"] = base_c + 0.04
            hi_c   = COST_PROFILES[tier].copy(); hi_c["capex"] = base_c - 0.04

            def _run_capex(profile):
                old = COST_PROFILES[tier]
                COST_PROFILES[tier] = profile
                res = run_dcf(oil_b, gas_b, wacc_b, tg_b, inputs, live, "")
                COST_PROFILES[tier] = old
                return res

            r_lo = _run_capex(lo_c)
            r_hi = _run_capex(hi_c)

        low_navs.append(r_lo["nav_share"])
        high_navs.append(r_hi["nav_share"])
        labels.append(label)

    # Sort by range (widest bar at top)
    ranges   = [abs(h - l) for h, l in zip(high_navs, low_navs)]
    order    = sorted(range(len(ranges)), key=lambda i: ranges[i])
    labels   = [labels[i]    for i in order]
    low_navs = [low_navs[i]  for i in order]
    high_navs= [high_navs[i] for i in order]

    # ─── Plot ───
    fig, ax = plt.subplots(figsize=(11, 6.5))
    fig.patch.set_facecolor('#0D1117')
    ax.set_facecolor('#0D1117')

    for i, (lbl, lo, hi) in enumerate(zip(labels, low_navs, high_navs)):
        ax.barh(i, lo - base_nav, left=base_nav, height=0.55,
                color='#C0392B', alpha=0.85, zorder=3)
        ax.barh(i, hi - base_nav, left=base_nav, height=0.55,
                color='#1A6B3A', alpha=0.85, zorder=3)
        ax.text(lo - 0.5, i, f"${lo:.1f}", va='center', ha='right',
                color='#E74C3C', fontsize=8.5, fontweight='bold')
        ax.text(hi + 0.5, i, f"${hi:.1f}", va='center', ha='left',
                color='#2ECC71', fontsize=8.5, fontweight='bold')

    ax.axvline(base_nav, color='#F0E68C', lw=1.8, ls='--', zorder=4,
               label=f"Base NAV: ${base_nav:.2f}")
    if live["price"]:
        ax.axvline(live["price"], color='#87CEEB', lw=1.5, ls=':', zorder=4,
                   label=f"Market Price: ${live['price']:.2f}")

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, color='#E8E8E0', fontsize=9.5)
    ax.set_xlabel("NAV per Share ($)", color='#E8E8E0', fontsize=10)
    ax.set_title(f"{company_name} ({ticker}) — DCF Sensitivity Analysis\nNAV/Share by Input Variable",
                 color='#F5F5F0', fontsize=13, fontweight='bold', pad=14)
    ax.tick_params(colors='#A0A098')
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:.0f}"))
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_color('#3A3A40')
    ax.grid(axis='x', color='#2A2A30', lw=0.6, zorder=1)
    ax.legend(framealpha=0.2, labelcolor='white', fontsize=9,
              loc='lower right', facecolor='#1A1A22')

    plt.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       f"{ticker.lower()}_tornado_chart.png")
    plt.savefig(out, dpi=180, bbox_inches='tight', facecolor='#0D1117')
    plt.close()
    print(f"\n[CHART] Tornado chart saved → {out}")
    return out


# ─────────────────────────────────────────────
# SECTION 10: MAIN RUNNER
# ─────────────────────────────────────────────

def main():
    # 1. Get ticker from user
    ticker       = prompt_ticker()
    company_name = EP_TICKERS.get(ticker, ticker)

    # 2. Pull live market data (price, shares, net debt)
    live = fetch_live_data(ticker)

    # 3. Collect operating inputs from user
    inputs = collect_operating_inputs(ticker)

    # 4. Run scenarios
    summary_df, results = run_scenarios(inputs, live)

    # 5. Build tornado chart
    base_result = next(r for r in results if r["label"] == "Base")
    build_tornado_chart(base_result, inputs, live, ticker, company_name)

    # 6. Final summary
    print("\n\n" + "=" * 62)
    print(f"  FINAL NAV SUMMARY — {company_name} ({ticker})")
    print("=" * 62)
    for r in results:
        pct = ((r["nav_share"] / live["price"]) - 1) * 100 if live["price"] else 0
        arrow = "▲" if pct > 0 else "▼"
        print(f"  {r['label']:<6}  NAV/Share: ${r['nav_share']:>8.2f}   "
              f"({arrow}{abs(pct):.1f}% vs market ${live['price']:.2f})")

    verdict = "UNDERVALUED" if base_result["nav_share"] > live["price"] else "OVERVALUED"
    print(f"\n  Base NAV/Share:  ${base_result['nav_share']:.2f}")
    print(f"  Model suggests:  {verdict}")
    print("\n[DONE] Model complete.\n")


if __name__ == "__main__":
    main()
