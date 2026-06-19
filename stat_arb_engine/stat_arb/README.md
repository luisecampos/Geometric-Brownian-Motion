# Stat-Arb Pairs Trading Engine

A realistic pairs trading system built the way quant research desks do it.

## What it does

1. Downloads price data from Yahoo Finance
2. Tests every ticker pair for cointegration (Engle-Granger)
3. Estimates a dynamic hedge ratio using a Kalman filter
4. Generates z-score entry/exit signals
5. Backtests the strategy and reports Sharpe, drawdown, win rate, etc.

## Setup

```bash
# 1. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
python main.py
```

## Project structure

```
stat_arb/
├── main.py                  # Entry point + config
├── requirements.txt
├── data/
│   └── fetch.py             # Price data download
├── signals/
│   ├── cointegration.py     # Engle-Granger pair finder
│   ├── kalman.py            # Dynamic hedge ratio (Kalman filter)
│   └── zscore.py            # Rolling z-score signal
└── backtest/
    ├── engine.py            # Day-by-day simulation
    └── report.py            # Performance metrics
```

## Key parameters (edit in main.py)

| Parameter         | Default | Effect                                      |
|-------------------|---------|---------------------------------------------|
| `TICKERS`         | 10 stocks | Universe to search for pairs              |
| `LOOKBACK_WINDOW` | 60 days | Z-score rolling window                      |
| `ENTRY_THRESHOLD` | 2.0σ    | Higher = fewer, higher-conviction trades    |
| `EXIT_THRESHOLD`  | 0.5σ    | Lower = hold longer, more mean reversion    |
| `PVALUE_CUTOFF`   | 0.05    | Cointegration significance level            |

## What to learn / extend next

- **Add more tickers** — bigger universe = more pairs found
- **Walk-forward validation** — re-fit cointegration on rolling windows
- **Position sizing** — Kelly criterion or vol-targeting instead of $1 notional
- **Stop losses** — exit if z-score blows past 3.5σ (pair broke down)
- **Portfolio of pairs** — run multiple pairs simultaneously, track correlation of PnL
- **Execution layer** — port the signal generation to C++ for live trading
