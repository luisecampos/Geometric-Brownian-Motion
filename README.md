# Geometric Brownian Motion Simulator

A Monte Carlo simulation engine for modeling equity and commodity price paths using stochastic differential equations. Built for financial forecasting and risk analysis.

## Overview

This simulator implements the Geometric Brownian Motion (GBM) model, the foundational framework for derivatives pricing and portfolio risk management. Widely used in investment banking, trading, and quantitative finance.

## Features

- **GBM Price Path Generation**: Simulates correlated asset price trajectories
- **Black-Scholes Validation**: Compares Monte Carlo results against closed-form option pricing
- **Risk Metrics**: Calculates Value-at-Risk (VaR) and Conditional Value-at-Risk (CVaR) at 95% confidence
- **Multi-Asset Support**: Backtested on equities (S&P 500, individual stocks) 

## Usage

```python
python GBM.py
```

## Results

- **Accuracy**: Monte Carlo option prices align within 2-3% of Black-Scholes analytical solutions
- **Backtests**: Validated on historical S&P 500 and; 95% confidence intervals capture actual price movements
- **Output**: Price simulations, visualizations, and risk metrics

## Technical Stack

- Python 3.12+
- NumPy (numerical computation)
- Pandas (data handling)
- Matplotlib (visualization)

## Applications

- Options pricing and Greeks calculation
- Portfolio Value-at-Risk analysis
- Energy commodity forecasting
- Investment decision support

