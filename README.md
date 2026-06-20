# Geometric Brownian Motion Simulator

A Monte Carlo simulation engine for modeling equity price paths using stochastic differential equations. Built for financial forecasting and risk analysis.

## Overview

This simulator implements the Geometric Brownian Motion (GBM) model, the foundational framework for derivatives pricing and portfolio risk management. Uses Monte Carlo methods to generate realistic price trajectories for equities.

## Features

- **GBM Price Path Generation**: Simulates correlated asset price trajectories using Itô calculus
- **Risk Metrics**: Calculates Value-at-Risk (VaR) and Conditional Value-at-Risk (CVaR) at 95% confidence
- **Visualization**: Outputs price paths and distribution plots for analysis
- **Backtesting**: Historical validation on major equity indices and individual stocks

## Usage

```python
python GBM.py
```

## Backtests

- S&P 500 historical price modeling
- Individual equity price simulations
- 95% confidence interval analysis

## Technical Stack

- Python 3.12+
- NumPy (numerical computation)
- Pandas (data handling)
- Matplotlib (visualization)

## Applications

- Portfolio Value-at-Risk analysis
- Equity price forecasting
- Monte Carlo simulation for financial modeling
