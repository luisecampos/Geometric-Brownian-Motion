# Geometric Brownian Motion Simulator

A Monte Carlo simulation engine for modeling equity and commodity price paths using stochastic differential equations. Built for financial forecasting and risk analysis.

## Overview

This simulator implements the Geometric Brownian Motion (GBM) model, the foundational framework for derivatives pricing and portfolio risk management. Uses Monte Carlo methods to generate realistic price trajectories for equities and commodities.

## Features

- **GBM Price Path Generation**: Simulates correlated asset price trajectories using Itô calculus
- **Multi-Asset Support**: Backtested on equities (S&P 500, individual stocks) and commodities (WTI crude oil)
- **Risk Metrics**: Calculates Value-at-Risk (VaR) and Conditional Value-at-Risk (CVaR) at 95% confidence
- **Visualization**: Outputs price paths and distribution plots for analysis

## Usage

```python
python GBM.py
```

## Backtests

- S&P 500 historical price modeling
- WTI crude oil commodity forecasting
- 95% confidence interval analysis

## Technical Stack

- Python 3.12+
- NumPy (numerical computation)
- Pandas (data handling)
- Matplotlib (visualization)

## Applications

- Portfolio Value-at-Risk analysis
- Energy commodity forecasting
- Price path simulation for financial modeling
