"""
SEC EDGAR 10-K Scraper — Interactive Version
=============================================
Asks you which tickers to pull, fetches 10-K data from the SEC,
saves CSVs, then launches a visual dashboard in your browser.

Run:
  python edgar_scraper.py
"""

import requests
import pandas as pd
import time
import os
import json
import webbrowser

USER_AGENT = os.environ.get("SEC_USER_AGENT", "")  # your name + email, e.g. "Your Name your@email.com"

CONCEPTS = {
    "Revenues":                        "us-gaap/Revenues",
    "RevenueFromContractWithCustomer": "us-gaap/RevenueFromContractWithCustomerExcludingAssessedTax",
    "NetIncomeLoss":                   "us-gaap/NetIncomeLoss",
    "OperatingIncomeLoss":             "us-gaap/OperatingIncomeLoss",
    "Assets":                          "us-gaap/Assets",
    "LongTermDebt":                    "us-gaap/LongTermDebt",
    "LongTermDebtNoncurrent":            "us-gaap/LongTermDebtNoncurrent",
    "CashAndEquivalents":              "us-gaap/CashAndCashEquivalentsAtCarryingValue",
    "OperatingCashFlow":               "us-gaap/NetCashProvidedByUsedInOperatingActivities",
    "CapEx":                           "us-gaap/PaymentsToAcquirePropertyPlantAndEquipment",
    "ShareholdersEquity":              "us-gaap/StockholdersEquity",
    "InterestExpense":                 "us-gaap/InterestExpense",
    "DepreciationAndAmortization":     "us-gaap/DepreciationDepletionAndAmortization",
}

BASE_URL          = "https://data.sec.gov"
COMPANY_FACTS_URL = BASE_URL + "/api/xbrl/companyfacts/CIK{cik}.json"
SUBMISSIONS_URL   = BASE_URL + "/submissions/CIK{cik}.json"
TICKER_MAP_URL    = "https://www.sec.gov/files/company_tickers.json"


# ─── helpers ─────────────────────────────────────────────────────────────────

def headers():
    if not USER_AGENT:
        raise RuntimeError('Set SEC_USER_AGENT to your name and email, e.g. "Your Name your@email.com".')
    return {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"}

def get_cik_map():
    print("  Fetching ticker → CIK map from SEC...")
    r = requests.get(TICKER_MAP_URL, headers=headers())
    r.raise_for_status()
    raw = r.json()
    return {v["ticker"].upper(): str(v["cik_str"]).zfill(10) for v in raw.values()}

def get_company_facts(cik):
    r = requests.get(COMPANY_FACTS_URL.format(cik=cik), headers=headers())
    if r.status_code == 404:
        return {}
    r.raise_for_status()
    return r.json()

def get_submission_metadata(cik):
    r = requests.get(SUBMISSIONS_URL.format(cik=cik), headers=headers())
    r.raise_for_status()
    return r.json()

def _period_days(start, end):
    """Return number of days between two YYYY-MM-DD strings. Returns None if missing."""
    if not start or not end:
        return None
    try:
        from datetime import date
        s = date.fromisoformat(start)
        e = date.fromisoformat(end)
        return (e - s).days
    except Exception:
        return None


def _amendment_rank(form):
    """10-K/A (amendment) is more authoritative than 10-K for the same period."""
    return 1 if form == "10-K/A" else 0


def extract_annual_values(facts, concept_path):
    """
    Precisely select one value per fiscal year from EDGAR XBRL data.

    Selection rules applied in order:
      1. Accept only 10-K and 10-K/A form types.
      2. Require a 'start' date — entries without one are likely point-in-time
         balance sheet snapshots filed inside an annual report but covering a
         different sub-period (e.g. a quarterly comparison column).
      3. Verify the period is a full fiscal year: between 340 and 380 days.
         This eliminates stub periods, transitions, and quarterly carve-outs
         that occasionally appear tagged as 10-K entries in XBRL.
      4. Key by fiscal year-end date (full YYYY-MM-DD), not just the year.
         Two 10-K entries with different end dates are different fiscal years,
         even if the calendar year is the same (e.g. a company that shifted its
         fiscal year end mid-year will have two valid entries for one calendar year).
      5. For duplicate keys (same fiscal year-end), prefer:
           a. 10-K/A over 10-K  (amendment supersedes original)
           b. Most recently *filed* date  (latest restatement wins)
      6. Sort output chronologically by fiscal year-end date.
    """
    namespace, concept = concept_path.split("/")
    try:
        data_points = facts["facts"][namespace][concept]["units"]["USD"]
    except KeyError:
        return []

    # Map: fiscal_end_date → best candidate so far
    best: dict = {}

    for p in data_points:
        form  = p.get("form", "")
        start = p.get("start", "")
        end   = p.get("end", "")
        filed = p.get("filed", "")
        val   = p.get("val")

        # Rule 1: annual filing forms only
        if form not in ("10-K", "10-K/A"):
            continue

        # Rule 2: must have a start date (flow metric over a period)
        # Balance-sheet point-in-time items (Assets, Debt, Cash) don't have
        # 'start' in XBRL — we relax this rule for them specifically below.
        # We detect flow vs. stock by whether ANY entry for this concept has 'start'.
        # For now record start as-is; we filter after.

        # Rule 3: full-year period check (only when start is present)
        if start:
            days = _period_days(start, end)
            if days is None or not (330 <= days <= 390):
                continue  # stub, quarterly carve-out, or transition period

        if not end:
            continue

        key = end  # fiscal year-end as the canonical identifier

        existing = best.get(key)
        if existing is None:
            best[key] = p
        else:
            # Rule 5a: prefer amendment
            if _amendment_rank(form) > _amendment_rank(existing.get("form", "")):
                best[key] = p
                continue
            if _amendment_rank(form) < _amendment_rank(existing.get("form", "")):
                continue
            # Rule 5b: prefer most recently filed
            if filed > existing.get("filed", ""):
                best[key] = p

    # If no entries had 'start' (pure balance-sheet concept), re-run without
    # the period-length filter so we still capture point-in-time values.
    if not best:
        for p in data_points:
            form  = p.get("form", "")
            end   = p.get("end", "")
            filed = p.get("filed", "")

            if form not in ("10-K", "10-K/A") or not end:
                continue

            key = end
            existing = best.get(key)
            if existing is None:
                best[key] = p
            else:
                if _amendment_rank(form) > _amendment_rank(existing.get("form", "")):
                    best[key] = p
                    continue
                if _amendment_rank(form) < _amendment_rank(existing.get("form", "")):
                    continue
                if filed > existing.get("filed", ""):
                    best[key] = p

    results = []
    for end_date, p in sorted(best.items()):
        results.append({
            "year":        end_date[:4],          # calendar year of period end
            "fiscal_end":  end_date,              # full date — more precise
            "value":       p.get("val"),
            "filed":       p.get("filed", ""),
            "form":        p.get("form", ""),
            "start":       p.get("start", ""),
        })

    return results

def scrape_company(ticker, cik):
    print(f"\n  [{ticker}] Fetching metadata...")
    try:
        meta = get_submission_metadata(cik)
        company_name = meta.get("name", ticker)
        sic = meta.get("sicDescription", "")
        print(f"  [{ticker}] {company_name}  |  {sic}")
    except Exception as e:
        company_name = ticker
        print(f"  [{ticker}] Metadata error: {e}")

    time.sleep(0.15)
    try:
        facts = get_company_facts(cik)
    except Exception as e:
        print(f"  [{ticker}] Facts error: {e}")
        return pd.DataFrame()

    if not facts:
        return pd.DataFrame()

    rows = []
    for label, path in CONCEPTS.items():
        for pt in extract_annual_values(facts, path):
            rows.append({
                "ticker":      ticker,
                "company":     company_name,
                "year":        pt["year"],
                "fiscal_end":  pt["fiscal_end"],
                "metric":      label,
                "value":       pt["value"],
                "value_mm":    round(pt["value"] / 1_000_000, 1) if pt["value"] is not None else None,
                "filed":       pt["filed"],
                "form":        pt["form"],
                "period_days": _period_days(pt.get("start", ""), pt["fiscal_end"]),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        print(f"  [{ticker}] No data extracted")
        return df

    years      = df["year"].nunique()
    amendments = (df["form"] == "10-K/A").sum()
    print(f"  [{ticker}] {len(df)} data points | {years} fiscal years | "
          f"{amendments} amended entries used")

    # Quality spot-check: warn if any period looks off
    odd = df[(df["period_days"].notna()) &
             ((df["period_days"] < 330) | (df["period_days"] > 390))]
    if not odd.empty:
        print(f"  [{ticker}] ⚠  {len(odd)} entries with unusual period length — review manually")

    return df

def pivot_wide(df):
    if df.empty:
        return df
    p = df.pivot_table(index=["ticker","company","year"], columns="metric", values="value_mm", aggfunc="first").reset_index()
    p.columns.name = None
    return p

def derive_metrics(df):
    if df.empty:
        return df
    if "Revenues" in df.columns:
        df["Revenue_mm"] = df["Revenues"]
    elif "RevenueFromContractWithCustomer" in df.columns:
        df["Revenue_mm"] = df["RevenueFromContractWithCustomer"]
    if "OperatingCashFlow" in df.columns and "CapEx" in df.columns:
        df["FreeCashFlow_mm"] = df["OperatingCashFlow"] - df["CapEx"].abs()
    if "NetIncomeLoss" in df.columns and "Revenue_mm" in df.columns:
        df["NetMargin_pct"] = (df["NetIncomeLoss"] / df["Revenue_mm"] * 100).round(1)
    if "LongTermDebt" in df.columns and "CashAndEquivalents" in df.columns:
        df["NetDebt_mm"] = df["LongTermDebt"] - df["CashAndEquivalents"]
    return df


# ─── user input ──────────────────────────────────────────────────────────────

def ask_tickers():
    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║         SEC EDGAR 10-K Scraper  ·  Luis Campos       ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()
    print("  Enter tickers separated by commas or spaces.")
    print("  Example:  RIG, VAL, SLB, HP")
    print("  Press Enter with no input to use default energy set.")
    print()

    raw = input("  Tickers → ").strip()

    if not raw:
        defaults = ["RIG", "VAL", "SLB", "HP", "NE"]
        print(f"\n  No input — using defaults: {', '.join(defaults)}")
        return defaults

    # Accept comma or space separated, strip extras, uppercase
    tickers = [t.strip().upper() for t in raw.replace(",", " ").split() if t.strip()]
    print(f"\n  Got {len(tickers)} ticker(s): {', '.join(tickers)}")
    return tickers


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    tickers = ask_tickers()

    print()
    print("─" * 54)
    print("  Loading SEC ticker map...")
    ticker_map = get_cik_map()

    all_dfs = []
    not_found = []

    for ticker in tickers:
        cik = ticker_map.get(ticker)
        if not cik:
            print(f"\n  [{ticker}] ✗ Not found in SEC database — skipping")
            not_found.append(ticker)
            continue
        df = scrape_company(ticker, cik)
        if not df.empty:
            all_dfs.append(df)
        time.sleep(0.4)

    if not_found:
        print(f"\n  Skipped (not in SEC): {', '.join(not_found)}")

    if not all_dfs:
        print("\n  No data retrieved. Check your User-Agent and internet connection.")
        return

    tidy_df = pd.concat(all_dfs, ignore_index=True)
    wide_df  = derive_metrics(pivot_wide(tidy_df))

    tidy_df.to_csv("edgar_10k_tidy.csv", index=False)
    wide_df.to_csv("edgar_10k_data.csv",  index=False)

    print()
    print("─" * 54)
    print(f"  ✓ Saved edgar_10k_tidy.csv")
    print(f"  ✓ Saved edgar_10k_data.csv")
    print(f"  Companies: {tidy_df['ticker'].nunique()}")
    print(f"  Year range: {tidy_df['year'].min()} – {tidy_df['year'].max()}")
    print()

    # Build and open the dashboard
    print("  Building dashboard...")
    build_dashboard(tidy_df)
    print("  ✓ Dashboard saved → edgar_dashboard.html")
    print("  Opening in browser...")
    webbrowser.open("edgar_dashboard.html")

# ─── dashboard builder ────────────────────────────────────────────────────────

def build_dashboard(tidy_df):
    """Convert tidy dataframe into a Bloomberg-themed HTML dashboard."""

    companies = tidy_df["ticker"].unique().tolist()
    n = len(companies)

    company_data = {}
    for ticker in companies:
        cdf = tidy_df[tidy_df["ticker"] == ticker].copy()
        name = cdf["company"].iloc[0] if len(cdf) else ticker

        def get_series(metric, _cdf=cdf):
            m = _cdf[_cdf["metric"] == metric].sort_values("fiscal_end")
            if m.empty and metric == "Revenues":
                m = _cdf[_cdf["metric"] == "RevenueFromContractWithCustomer"].sort_values("fiscal_end")
            if m.empty and metric == "LongTermDebt":
                m = _cdf[_cdf["metric"] == "LongTermDebtNoncurrent"].sort_values("fiscal_end")
            years  = m["year"].tolist()
            values = [round(v, 1) if v is not None else None for v in m["value_mm"].tolist()]
            return {"years": years, "values": values}

        def latest(metric, _cdf=cdf):
            m = _cdf[_cdf["metric"] == metric].sort_values("fiscal_end")
            if m.empty and metric == "Revenues":
                m = _cdf[_cdf["metric"] == "RevenueFromContractWithCustomer"].sort_values("fiscal_end")
            if m.empty and metric == "LongTermDebt":
                m = _cdf[_cdf["metric"] == "LongTermDebtNoncurrent"].sort_values("fiscal_end")
            if m.empty: return None
            v = m.iloc[-1]["value_mm"]
            return round(v, 1) if v is not None else None

        company_data[ticker] = {
            "name":      name,
            "ticker":    ticker,
            "revenue":   get_series("Revenues"),
            "netIncome": get_series("NetIncomeLoss"),
            "cfo":       get_series("OperatingCashFlow"),
            "debt":      get_series("LongTermDebt"),
            "snap": {
                "REVENUE":      latest("Revenues"),
                "NET INCOME":   latest("NetIncomeLoss"),
                "OPER CF":      latest("OperatingCashFlow"),
                "LT DEBT":      latest("LongTermDebt"),
                "CASH":         latest("CashAndEquivalents"),
                "TOTAL ASSETS": latest("Assets"),
            }
        }

    import json
    data_json    = json.dumps(company_data)
    tickers_json = json.dumps(companies)

    if n == 1:
        grid_cols = "1fr"
    elif n == 2:
        grid_cols = "1fr 1fr"
    else:
        grid_cols = "repeat(3, 1fr)"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>EDGAR FA | Bloomberg Terminal</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: #000;
    color: #fff;
    font-family: 'Courier New', Courier, monospace;
    font-size: 13px;
    min-height: 100vh;
  }}

  /* Only orange: top bar + logo + F-key numbers + source label */
  .bbg-top {{
    background: #FF6200;
    color: #000;
    font-weight: 700;
    font-size: 11px;
    letter-spacing: 0.06em;
    padding: 3px 10px;
    display: flex;
    justify-content: space-between;
  }}
  .bbg-header {{
    background: #000;
    border-bottom: 1px solid #2a2a2a;
    padding: 6px 10px;
    display: flex;
    align-items: center;
  }}
  .bbg-logo {{
    background: #FF6200;
    color: #000;
    font-weight: 900;
    font-size: 20px;
    padding: 2px 10px 2px 8px;
    margin-right: 14px;
    flex-shrink: 0;
  }}
  .bbg-cmd  {{ color: #FF6200; font-size: 14px; font-weight: 700; margin-right: 18px; flex-shrink: 0; }}
  .bbg-desc {{ color: #888; font-size: 11px; flex: 1; }}
  .bbg-clock {{ text-align: right; flex-shrink: 0; }}
  .bbg-clock .city {{ font-size: 9px; color: #555; }}
  .bbg-clock .time {{ font-size: 15px; color: #fff; font-weight: 700; letter-spacing: 0.04em; }}

  /* Menu bar — grey keys, white text, active = white bg black text */
  .bbg-menu {{
    background: #1a1a1a;
    border-bottom: 1px solid #2a2a2a;
    display: flex;
    align-items: stretch;
  }}
  .menu-key {{
    color: #bbb;
    font-size: 11px;
    font-weight: 700;
    padding: 5px 14px;
    cursor: pointer;
    border-right: 1px solid #2a2a2a;
    white-space: nowrap;
    letter-spacing: 0.04em;
    display: flex;
    align-items: center;
    gap: 5px;
    transition: background 0.1s, color 0.1s;
    user-select: none;
  }}
  .menu-key:hover  {{ background: #2a2a2a; color: #fff; }}
  .menu-key.active {{ background: #000; color: #fff; border-bottom: 2px solid #FF6200; }}
  .menu-key .fnum  {{ color: #FF6200; font-size: 9px; }}
  .menu-right {{ margin-left: auto; display: flex; }}

  /* Security strip */
  .sec-strip {{
    background: #0a0a0a;
    border-bottom: 1px solid #1a1a1a;
    padding: 4px 10px;
    display: flex;
    align-items: center;
    gap: 2px;
    overflow-x: auto;
  }}
  .sec-label {{ color: #444; font-size: 10px; margin-right: 10px; white-space: nowrap; }}
  .sec-item {{
    display: flex;
    align-items: baseline;
    gap: 6px;
    padding: 2px 10px;
    border: 1px solid #222;
    white-space: nowrap;
  }}
  .sec-sym  {{ font-size: 12px; font-weight: 700; }}
  .sec-exch {{ font-size: 9px; color: #444; }}

  /* Panel grid */
  .panel-grid {{
    display: grid;
    grid-template-columns: {grid_cols};
    gap: 0;
    background: #111;
  }}
  .co-panel {{
    background: #000;
    border-right: 1px solid #1a1a1a;
    border-bottom: 1px solid #1a1a1a;
    display: none;
    flex-direction: column;
  }}
  .co-panel.visible {{ display: flex; }}

  /* Panel title */
  .panel-title {{
    background: #0a0a0a;
    border-bottom: 1px solid #1a1a1a;
    padding: 4px 8px;
    display: flex;
    align-items: baseline;
    gap: 8px;
  }}
  .pt-ticker {{ font-size: 15px; font-weight: 700; }}
  .pt-equity {{ font-size: 10px; color: #444; }}
  .pt-name   {{ font-size: 11px; color: #777; flex: 1; }}
  .pt-src    {{ font-size: 9px; color: #FF6200; margin-left: auto; }}

  /* Key stats */
  .kv-table {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    border-bottom: 1px solid #1a1a1a;
  }}
  .kv-cell {{
    padding: 5px 8px;
    border-right: 1px solid #1a1a1a;
    border-bottom: 1px solid #1a1a1a;
  }}
  .kv-cell:nth-child(3n) {{ border-right: none; }}
  .kv-lbl {{ font-size: 9px; color: #444; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 2px; }}
  .kv-val {{ font-size: 13px; font-weight: 700; color: #fff; }}
  .kv-val.pos {{ color: #00cc44; }}
  .kv-val.neg {{ color: #ff3333; }}
  .kv-unit {{ font-size: 8px; color: #333; margin-left: 2px; }}

  /* Charts section */
  .charts-wrap {{ padding: 8px; flex: 1; }}

  /* Each chart view is a div we show/hide */
  .view-section {{ display: none; }}
  .view-section.active {{ display: block; }}

  /* Overview = 2x2 grid */
  .chart-grid-2x2 {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 6px;
  }}
  /* Single metric view = 1 full-width chart */
  .chart-grid-1x1 {{ display: grid; grid-template-columns: 1fr; gap: 6px; }}

  .chart-box {{
    background: #080808;
    border: 1px solid #1a1a1a;
    padding: 6px 8px 8px;
  }}
  .chart-header {{
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 4px;
  }}
  .chart-title {{ font-size: 8px; color: #444; text-transform: uppercase; letter-spacing: 0.1em; }}
  .chart-last  {{ font-size: 11px; font-weight: 700; color: #ccc; }}
  .chart-last.pos {{ color: #00cc44; }}
  .chart-last.neg {{ color: #ff3333; }}
  canvas {{ max-height: 120px; width: 100% !important; display: block; }}
  .chart-grid-1x1 canvas {{ max-height: 240px; }}

  .solo .kv-table {{ grid-template-columns: repeat(6, 1fr); }}

  /* Status bar */
  .bbg-status {{
    background: #0a0a0a;
    border-top: 1px solid #1a1a1a;
    padding: 3px 10px;
    display: flex;
    justify-content: space-between;
    font-size: 9px;
    color: #333;
    letter-spacing: 0.05em;
  }}
  .bbg-status .ok {{ color: #00cc44; }}
</style>
</head>
<body>

<div class="bbg-top">
  <span>BLOOMBERG PROFESSIONAL &nbsp;&#9679;&nbsp; SEC EDGAR XBRL FEED &nbsp;&#9679;&nbsp; ANNUAL 10-K</span>
  <span id="topDate"></span>
</div>

<div class="bbg-header">
  <div class="bbg-logo">B</div>
  <div class="bbg-cmd">FA &lt;GO&gt;</div>
  <div class="bbg-desc">FUNDAMENTAL ANALYSIS &mdash; SEC EDGAR 10-K &mdash; USD $MM</div>
  <div class="bbg-clock">
    <div class="city">NEW YORK</div>
    <div class="time" id="clock">--:--:--</div>
  </div>
</div>

<div class="bbg-menu">
  <div class="menu-key active" data-view="overview" onclick="setView(this)"><span class="fnum">1</span> OVERVIEW</div>
  <div class="menu-key" data-view="revenue"  onclick="setView(this)"><span class="fnum">2</span> REVENUE</div>
  <div class="menu-key" data-view="income"   onclick="setView(this)"><span class="fnum">3</span> NET INCOME</div>
  <div class="menu-key" data-view="cashflow" onclick="setView(this)"><span class="fnum">4</span> CASH FLOW</div>
  <div class="menu-key" data-view="leverage" onclick="setView(this)"><span class="fnum">5</span> LEVERAGE</div>
  <div class="menu-right">
    <div class="menu-key" style="color:#FF6200" onclick="exportNote()">EXPORT</div>
    <div class="menu-key" style="color:#FF6200" onclick="window.print()">PRINT</div>
  </div>
</div>

<div class="sec-strip">
  <span class="sec-label">SECURITIES LOADED:</span>
  <div id="secStrip"></div>
</div>

<div class="panel-grid" id="panelGrid"></div>

<div class="bbg-status">
  <div>SOURCE: SEC.GOV/EDGAR &nbsp;|&nbsp; FILTER: 10-K + 10-K/A &nbsp;|&nbsp; PERIOD: 330-390 DAYS &nbsp;|&nbsp; AMENDMENTS PRIORITIZED &nbsp;|&nbsp; LATEST RESTATEMENT WINS</div>
  <div><span class="ok">&#9679; CONNECTED</span> &nbsp; SEC.GOV</div>
</div>

<script>
const RAW     = {data_json};
const TICKERS = {tickers_json};

const COLORS = ['#00b8ff','#ffcc00','#cc88ff','#00ffcc','#ff8844','#88ccff'];
let charts = {{}};
let currentView = 'overview';

function fmt(v) {{
  if (v === null || v === undefined) return 'N/A';
  const a    = Math.abs(v);
  const sign = v < 0 ? '-' : '';
  if (a >= 1000) return sign + '$' + (a / 1000).toFixed(2) + 'B';
  return sign + '$' + a.toFixed(0) + 'M';
}}

function cls(v) {{
  if (v === null || v === undefined) return '';
  return v < 0 ? 'neg' : 'pos';
}}

function latestOf(series) {{
  if (!series || !series.values || !series.values.length) return null;
  for (let i = series.values.length - 1; i >= 0; i--) {{
    if (series.values[i] !== null) return series.values[i];
  }}
  return null;
}}

function chartOpts(color) {{
  return {{
    responsive: true,
    maintainAspectRatio: false,
    animation: {{ duration: 250 }},
    plugins: {{
      legend: {{ display: false }},
      tooltip: {{
        backgroundColor: '#111',
        borderColor: color,
        borderWidth: 1,
        titleColor: '#555',
        bodyColor: '#fff',
        titleFont: {{ family: 'Courier New', size: 9 }},
        bodyFont:  {{ family: 'Courier New', size: 11, weight: '700' }},
        callbacks: {{ label: ctx => ' ' + fmt(ctx.parsed.y) }}
      }}
    }},
    scales: {{
      x: {{
        ticks: {{ color: '#444', font: {{ family: 'Courier New', size: 8 }}, maxRotation: 0 }},
        grid:  {{ color: '#111' }},
        border:{{ color: '#222' }}
      }},
      y: {{
        ticks: {{ color: '#444', font: {{ family: 'Courier New', size: 8 }}, callback: v => fmt(v) }},
        grid:  {{ color: '#111' }},
        border:{{ color: '#222' }}
      }}
    }}
  }};
}}

function makeChart(id, years, values, color, type) {{
  const ctx = document.getElementById(id);
  if (!ctx) return;
  if (charts[id]) charts[id].destroy();
  const isLine = type === 'line';
  const bg = values ? values.map(v => v !== null && v < 0 ? '#ff333344' : color + '44') : color + '44';
  const bd = values ? values.map(v => v !== null && v < 0 ? '#ff3333'   : color)         : color;
  charts[id] = new Chart(ctx, {{
    type: isLine ? 'line' : 'bar',
    data: {{
      labels: years,
      datasets: [{{
        data: values,
        backgroundColor: isLine ? 'transparent' : bg,
        borderColor: isLine ? color : bd,
        borderWidth: isLine ? 1.5 : 1,
        pointBackgroundColor: color,
        pointBorderColor: '#000',
        pointRadius: isLine ? 3 : 0,
        fill: false,
        tension: 0.1,
      }}]
    }},
    options: chartOpts(color)
  }});
}}

function chartBox(id, title, lastVal, color) {{
  const lv = latestOf(lastVal);
  return `<div class="chart-box">
    <div class="chart-header">
      <span class="chart-title">${{title}}</span>
      <span class="chart-last ${{cls(lv)}}">${{fmt(lv)}}</span>
    </div>
    <canvas id="${{id}}"></canvas>
  </div>`;
}}

function renderPanels() {{
  const grid = document.getElementById('panelGrid');
  grid.innerHTML = '';

  TICKERS.forEach((ticker, i) => {{
    const d    = RAW[ticker];
    const col  = COLORS[i % COLORS.length];
    const solo = TICKERS.length === 1;

    const kvHTML = Object.entries(d.snap).map(([k, v]) =>
      `<div class="kv-cell">
        <div class="kv-lbl">${{k}}</div>
        <div class="kv-val ${{cls(v)}}">${{fmt(v)}}<span class="kv-unit">MM</span></div>
      </div>`
    ).join('');

    const panel = document.createElement('div');
    panel.className = 'co-panel visible' + (solo ? ' solo' : '');
    panel.id = 'panel-' + ticker;

    // Each view section has its own chart(s)
    // Overview: 2x2 grid of all 4 charts
    // Revenue / Income / CashFlow / Leverage: single full-width chart
    panel.innerHTML = `
      <div class="panel-title">
        <span class="pt-ticker" style="color:${{col}}">${{ticker}}</span>
        <span class="pt-equity">US EQUITY</span>
        <span class="pt-name">${{d.name.toUpperCase()}}</span>
        <span class="pt-src">SEC EDGAR</span>
      </div>
      <div class="kv-table">${{kvHTML}}</div>
      <div class="charts-wrap">

        <div class="view-section active" data-section="overview">
          <div class="chart-grid-2x2">
            ${{chartBox('ov-rev-'+ticker,  'REVENUE',        d.revenue,   col)}}
            ${{chartBox('ov-ni-'+ticker,   'NET INCOME',     d.netIncome, col)}}
            ${{chartBox('ov-cfo-'+ticker,  'OPER CASH FLOW', d.cfo,       col)}}
            ${{chartBox('ov-debt-'+ticker, 'LONG-TERM DEBT', d.debt,      col)}}
          </div>
        </div>

        <div class="view-section" data-section="revenue">
          <div class="chart-grid-1x1">
            ${{chartBox('rv-'+ticker, 'REVENUE ($MM)', d.revenue, col)}}
          </div>
        </div>

        <div class="view-section" data-section="income">
          <div class="chart-grid-1x1">
            ${{chartBox('ni-'+ticker, 'NET INCOME ($MM)', d.netIncome, col)}}
          </div>
        </div>

        <div class="view-section" data-section="cashflow">
          <div class="chart-grid-1x1">
            ${{chartBox('cf-'+ticker, 'OPERATING CASH FLOW ($MM)', d.cfo, col)}}
          </div>
        </div>

        <div class="view-section" data-section="leverage">
          <div class="chart-grid-1x1">
            ${{chartBox('lv-'+ticker, 'LONG-TERM DEBT ($MM)', d.debt, col)}}
          </div>
        </div>

      </div>`;

    grid.appendChild(panel);
  }});

  setTimeout(() => drawChartsForView('overview'), 100);
}}

function drawChartsForView(view) {{
  TICKERS.forEach((ticker, i) => {{
    const d   = RAW[ticker];
    const col = COLORS[i % COLORS.length];

    if (view === 'overview') {{
      makeChart('ov-rev-'+ticker,  d.revenue.years,  d.revenue.values,  col, 'bar');
      makeChart('ov-ni-'+ticker,   d.netIncome.years, d.netIncome.values, col, 'bar');
      makeChart('ov-cfo-'+ticker,  d.cfo.years,       d.cfo.values,       col, 'line');
      makeChart('ov-debt-'+ticker, d.debt.years,      d.debt.values,      col, 'line');
    }} else if (view === 'revenue') {{
      makeChart('rv-'+ticker, d.revenue.years, d.revenue.values, col, 'bar');
    }} else if (view === 'income') {{
      makeChart('ni-'+ticker, d.netIncome.years, d.netIncome.values, col, 'bar');
    }} else if (view === 'cashflow') {{
      makeChart('cf-'+ticker, d.cfo.years, d.cfo.values, col, 'line');
    }} else if (view === 'leverage') {{
      makeChart('lv-'+ticker, d.debt.years, d.debt.values, col, 'line');
    }}
  }});
}}

function setView(el) {{
  const view = el.dataset.view;
  if (view === currentView) return;
  currentView = view;

  // Update menu active state
  document.querySelectorAll('.menu-key[data-view]').forEach(k => k.classList.remove('active'));
  el.classList.add('active');

  // Show/hide the right view-section inside every panel
  document.querySelectorAll('.view-section').forEach(s => {{
    s.classList.toggle('active', s.dataset.section === view);
  }});

  // Draw charts for the newly visible canvases
  setTimeout(() => drawChartsForView(view), 30);
}}

function buildSecStrip() {{
  const strip = document.getElementById('secStrip');
  TICKERS.forEach((t, i) => {{
    const col = COLORS[i % COLORS.length];
    const el  = document.createElement('div');
    el.className = 'sec-item';
    el.innerHTML = `<span class="sec-sym" style="color:${{col}}">${{t}}</span><span class="sec-exch">US</span>`;
    strip.appendChild(el);
  }});
}}

function exportNote() {{
  alert('CSVs saved: edgar_10k_tidy.csv and edgar_10k_data.csv');
}}

function tick() {{
  const ny = new Date(new Date().toLocaleString('en-US', {{ timeZone: 'America/New_York' }}));
  document.getElementById('clock').textContent = ny.toTimeString().slice(0, 8);
  document.getElementById('topDate').textContent =
    ny.toLocaleDateString('en-US', {{ weekday:'short', month:'short', day:'2-digit', year:'numeric' }}).toUpperCase();
}}
tick();
setInterval(tick, 1000);

buildSecStrip();
renderPanels();
</script>
</body>
</html>"""

    with open("edgar_dashboard.html", "w", encoding="utf-8") as f:
        f.write(html)



if __name__ == "__main__":
    main()
