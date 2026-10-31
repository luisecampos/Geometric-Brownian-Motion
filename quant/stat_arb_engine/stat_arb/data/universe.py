"""
data/universe.py
----------------
The wide universe: the 11 SPDR sector ETFs plus ~25 large caps from each
sector, with two maps:

  SECTORS     ticker -> GICS sector. Walk-forward only tests pairs inside a
              sector (and each ETF only against its own sector), which keeps
              the candidate set economically sensible and the pair count
              ~11 x C(27, 2) ~ 3,900 per window instead of C(300, 2) ~ 45,000.
  INDUSTRIES  ticker -> finer industry group inside the sector. Feeds the
              meta-labeler's `same_sector` feature. With pairs restricted to
              one GICS sector, a sector-level flag would be 1 for every pair
              and carry no information, so the feature is "same industry
              group" here (banks vs insurers, E&P vs refiners, ...). Sector
              ETFs get their own group, so ETF/stock pairs score 0.

Names were picked as large caps that traded through the whole 2018-2024
window, from memory of the 2018 sector-ETF holdings. That is a survivorship-
biased universe: companies that were large in 2018 and later went bankrupt,
were acquired, or were delisted are missing (yfinance has no data for most
of them), and so are names that only became large later. For pairs trading
the bias is milder than for a long-only backtest, since both legs come from
the same pool, but it is not zero: a pair whose leg went to zero would have
been a large loss the backtest can't see.

The bias shows up directly: as of late 2026 Yahoo returns no history at all
for BK, AVB, EQR, EA, MMC, CTRA, SEE and IPG (renamed, acquired or taken
private since), so fetch_prices drops them. They are left in the list so
the drop stays visible. XLC only launched in June 2018, so it is dropped
for the 2018 start too; its stocks still pair with each other. That leaves
291 names.
"""

from __future__ import annotations


# sector -> {industry group: [tickers]}. The sector ETF is listed first.
_UNIVERSE: dict[str, dict[str, list[str]]] = {
    "tech": {
        "etf"           : ["XLK"],
        "hardware"      : ["AAPL", "CSCO", "HPQ", "NTAP", "GLW", "APH", "TEL", "MSI"],
        "semis"         : ["NVDA", "AVGO", "INTC", "TXN", "QCOM", "AMD", "AMAT", "MU",
                           "ADI", "LRCX", "KLAC"],
        "software"      : ["MSFT", "ORCL", "ADBE", "CRM", "INTU", "SNPS", "CDNS"],
        "it_services"   : ["ACN", "IBM"],
    },
    "financials": {
        "etf"           : ["XLF"],
        "banks"         : ["JPM", "BAC", "WFC", "C", "USB", "PNC", "TFC", "COF"],
        "capital_mkts"  : ["GS", "MS", "BLK", "SCHW", "BK", "STT", "SPGI", "MCO", "ICE", "CME"],
        "insurance"     : ["MET", "PRU", "AIG", "TRV", "ALL", "CB", "AFL", "PGR", "MMC", "AON"],
        "diversified"   : ["BRK-B", "AXP"],
    },
    "energy": {
        "etf"           : ["XLE"],
        "integrated"    : ["XOM", "CVX"],
        "e_and_p"       : ["COP", "EOG", "OXY", "DVN", "APA", "FANG", "CTRA", "EQT", "OVV"],
        "refiners"      : ["PSX", "MPC", "VLO"],
        "services"      : ["SLB", "HAL", "BKR", "NOV", "HP"],
        "midstream"     : ["KMI", "WMB", "OKE", "TRGP"],
    },
    "health": {
        "etf"           : ["XLV"],
        "pharma"        : ["JNJ", "PFE", "MRK", "ABBV", "LLY", "BMY", "ZTS"],
        "biotech"       : ["AMGN", "GILD", "VRTX", "REGN", "BIIB"],
        "devices"       : ["ABT", "MDT", "ISRG", "SYK", "BDX", "BSX"],
        "tools"         : ["TMO", "DHR", "IQV"],
        "managed_care"  : ["UNH", "CVS", "CI", "ELV", "HUM", "HCA", "MCK", "CAH"],
    },
    "industrials": {
        "etf"           : ["XLI"],
        "aerospace"     : ["BA", "LMT", "RTX", "GD", "NOC"],
        "machinery"     : ["CAT", "DE", "CMI", "PCAR", "ITW", "PH"],
        "conglomerates" : ["HON", "GE", "MMM", "EMR", "ETN", "ROK", "JCI"],
        "transport"     : ["UNP", "CSX", "NSC", "UPS", "FDX"],
        "airlines"      : ["DAL", "UAL", "AAL", "LUV"],
        "services"      : ["WM", "ADP"],
    },
    "staples": {
        "etf"           : ["XLP"],
        "beverages"     : ["KO", "PEP", "STZ", "KDP"],
        "food"          : ["MDLZ", "GIS", "KHC", "HSY", "MKC", "TSN", "HRL", "CAG", "CPB", "SJM", "ADM"],
        "household"     : ["PG", "CL", "KMB", "CLX", "CHD", "EL"],
        "tobacco"       : ["MO", "PM"],
        "retail"        : ["WMT", "COST", "TGT", "KR", "SYY", "DG", "DLTR"],
    },
    "discretionary": {
        "etf"           : ["XLY"],
        "internet_retail": ["AMZN", "EBAY", "BKNG"],
        "autos"         : ["TSLA", "GM", "F", "GPC"],
        "home_improve"  : ["HD", "LOW"],
        "restaurants"   : ["MCD", "SBUX", "YUM", "CMG"],
        "apparel_retail": ["NKE", "TJX", "ROST", "BBY", "ULTA", "TSCO", "ORLY", "AZO"],
        "hotels_leisure": ["MAR", "HLT", "LVS", "MGM", "RCL", "CCL"],
        "homebuilders"  : ["DHI", "LEN"],
    },
    "utilities": {
        "etf"           : ["XLU"],
        "electric"      : ["NEE", "DUK", "SO", "AEP", "EXC", "XEL", "ED", "EIX", "ETR", "PPL",
                           "FE", "EVRG", "PNW", "LNT", "WEC", "ES", "DTE", "AEE", "CMS"],
        "multi_gas"     : ["D", "SRE", "PEG", "CNP", "NI", "ATO"],
        "ipp"           : ["AES", "NRG"],
    },
    "materials": {
        "etf"           : ["XLB"],
        "chemicals"     : ["LIN", "APD", "SHW", "ECL", "DD", "PPG", "IFF", "ALB", "EMN",
                           "CE", "LYB", "CF", "MOS"],
        "metals_mining" : ["NEM", "FCX", "NUE", "STLD"],
        "construction"  : ["VMC", "MLM"],
        "packaging"     : ["BALL", "IP", "PKG", "AVY", "SEE"],
    },
    "real_estate": {
        "etf"           : ["XLRE"],
        "towers_data"   : ["AMT", "CCI", "SBAC", "EQIX", "DLR", "IRM"],
        "residential"   : ["AVB", "EQR", "ESS", "MAA", "UDR", "CPT"],
        "retail_reit"   : ["SPG", "O", "KIM", "REG", "FRT"],
        "other_reit"    : ["PLD", "PSA", "EXR", "WELL", "VTR", "BXP", "ARE", "HST"],
    },
    "communication": {
        "etf"           : ["XLC"],
        "media"         : ["GOOGL", "META", "NFLX", "DIS", "CMCSA", "CHTR", "EA", "TTWO",
                           "OMC", "IPG", "LYV", "NWSA"],
        "telecom"       : ["T", "VZ", "TMUS"],
    },
}

SECTORS    = {t: s for s, groups in _UNIVERSE.items() for ts in groups.values() for t in ts}
INDUSTRIES = {t: f"{s}/{g}" for s, groups in _UNIVERSE.items() for g, ts in groups.items() for t in ts}
UNIVERSE   = list(SECTORS)
SECTOR_ETFS = {s: groups["etf"][0] for s, groups in _UNIVERSE.items()}   # sector -> its ETF
