# SEC EDGAR 10-K Scraper
### Pull annual financial data for any public company — free, no API key

---

## What this project does

Two scripts work together:

| Script | Purpose |
|---|---|
| `edgar_scraper.py` | Hits the SEC EDGAR API, pulls 10-K financial facts, saves to CSV |
| `analyze_10k.py` | Reads that CSV and builds formatted tables + an Excel comps sheet |

Out of the box it pulls **RIG, VAL, SLB, HP, NE** — the core offshore drilling / oilfield services universe for energy IB — but you can swap in any US-listed ticker.

---

## Setup (5 minutes)

```bash
# 1. Install dependencies
pip install requests pandas openpyxl

# 2. Set your real name + email in the environment
export SEC_USER_AGENT="Your Name your@email.com"
# SEC policy requires this — fake values will get you 403 blocked

# 3. Run the scraper
python edgar_scraper.py

# 4. Run the analysis
python analyze_10k.py
```

---

## Output files

```
edgar_10k_tidy.csv      One row per (ticker, year, metric) — good for pandas work
edgar_10k_data.csv      Wide format: one row per (ticker, year) — good for Excel
comps_analysis.xlsx     Formatted workbook with Revenue, NI, FCF, Leverage sheets
```

---

## Metrics pulled from EDGAR

| Label | XBRL Concept | Notes |
|---|---|---|
| Revenue | `us-gaap/Revenues` | Falls back to ContractRevenue |
| Net Income | `us-gaap/NetIncomeLoss` | Includes losses (negative) |
| EBIT | `us-gaap/OperatingIncomeLoss` | Pre-interest, pre-tax |
| Operating Cash Flow | `us-gaap/NetCashProvidedByUsedInOperatingActivities` | |
| CapEx | `us-gaap/PaymentsToAcquirePropertyPlantAndEquipment` | |
| Total Assets | `us-gaap/Assets` | Balance sheet |
| Long-Term Debt | `us-gaap/LongTermDebt` | Excludes current portion |
| Cash | `us-gaap/CashAndCashEquivalentsAtCarryingValue` | |
| Shareholders' Equity | `us-gaap/StockholdersEquity` | |
| Interest Expense | `us-gaap/InterestExpense` | |
| D&A | `us-gaap/DepreciationDepletionAndAmortization` | |

**Derived metrics** (computed in analyze_10k.py):
- **FCF** = Operating CF − CapEx
- **Net Debt** = LT Debt − Cash
- **Net Margin** = Net Income / Revenue
- **EBITDA Margin** = EBITDA / Revenue (when available)

---

## Customizing tickers

In `edgar_scraper.py`, find line ~38:

```python
TICKERS = ["RIG", "VAL", "SLB", "HP", "NE"]
```

Replace with any US-listed tickers. Examples for other energy verticals:

```python
# E&P
TICKERS = ["OXY", "DVN", "MTDR", "FANG", "CTRA"]

# Midstream
TICKERS = ["ET", "EPD", "KMI", "WMB", "MPLX"]

# Integrated
TICKERS = ["XOM", "CVX", "COP", "BP", "SHEL"]
```

---

## Adding more metrics

Find the `CONCEPTS` dict in `edgar_scraper.py` (~line 43) and add any XBRL tag:

```python
"Backlog": "us-gaap/RevenueRemainingPerformanceObligation",
"Goodwill": "us-gaap/Goodwill",
"CommonSharesOutstanding": "us-gaap/CommonStockSharesOutstanding",
```

Full tag browser: https://data.sec.gov/api/xbrl/frames/

---

## How the SEC EDGAR API works

```
Ticker → CIK lookup
  GET https://www.sec.gov/files/company_tickers.json
  Returns every US public company's ticker → CIK mapping

Filing history
  GET https://data.sec.gov/submissions/CIK{cik}.json
  Returns all filings: 10-K, 10-Q, 8-K, etc.

Financial facts (XBRL structured data)
  GET https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json
  Returns every reported financial figure, tagged by accounting concept
  This is what we parse in extract_annual_values()
```

The key insight: SEC requires companies to tag their financials using **XBRL** 
(eXtensible Business Reporting Language). This is what makes machine-readable 
extraction possible without scraping raw HTML.

---

## Concepts to understand as you build this

**Why XBRL?** SEC mandated structured financial tagging starting ~2009. Before 
that, every financial statement was a PDF/HTML page you'd have to parse by hand. 
XBRL means you can pull Apple's revenue and Exxon's revenue with the exact same 
code — same tag, same structure.

**Why some companies are missing data?** Smaller filers, foreign private issuers 
(like some RIG entities), and very old filings may not have structured XBRL data. 
The scraper handles this gracefully and skips rather than crashing.

**Rate limits:** SEC asks for no more than 10 requests/second. The `time.sleep(0.15)` 
calls in the scraper keep you well under that. If you run this in a loop over 
hundreds of tickers, add a longer sleep between companies.

---

## Next extensions to build

1. **Add price data** — pull current price from yfinance, compute EV = Market Cap + Net Debt, then calculate EV/EBITDA and P/E multiples automatically

2. **Streamlit dashboard** — wrap the output in a Streamlit app so you have a live, visual comps table you can demo in a coffee chat

3. **Historical 10-K HTML parsing** — use BeautifulSoup to pull data from filings that predate XBRL structured tagging (pre-2009)

4. **Alert system** — schedule the scraper to run quarterly and email you when a new 10-K drops for a watched company

---

## Recruiting talking points

When a banker asks about your technical projects, you can say:

> "I built a Python tool that pulls structured financial data directly from 
> the SEC EDGAR API for any public company — no manual data entry. It extracts 
> ten years of revenue, EBITDA, cash flow, and leverage metrics and outputs a 
> formatted comparable company analysis. I built it to support my offshore 
> drilling pitch work on Transocean and Valaris."

That hits three things bankers care about: **initiative**, **technical ability**, 
and **domain relevance**.
