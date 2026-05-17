# Derivatives Pricing Engine

A Python-based derivatives pricing library for European vanilla options, Greeks, implied volatility, binomial tree pricing, and Monte Carlo simulation.

This project is designed as an educational and portfolio project to demonstrate quantitative finance, numerical methods, and software engineering practices in Python.

## Features

- Black-Scholes pricing for European call and put options
- Continuous dividend yield support
- Greeks: Delta, Gamma, Vega, Theta, Rho
- Implied volatility solver using Brent's method
- Arbitrage bounds validation for implied volatility
- CRR binomial tree pricing for European and American options
- Early exercise handling for American options
- Monte Carlo simulation under GBM dynamics
- Arithmetic Asian option pricing
- Monte Carlo standard error and 95% confidence interval
- Pytest test suite
- Jupyter sanity-check notebook
- Streamlit demo app

## Project Structure

```text
derivatives-pricing-engine/
- app/
  - app.py
- notebooks/
  - 01_sanity_checks.ipynb
- src/
  - pricing_engine/
    - __init__.py
    - bs.py
    - curves.py
    - greeks.py
    - implied_vol.py
    - monte_carlo.py
    - trees.py
- tests/
  - test_bs.py
  - test_curves.py
  - test_greeks.py
  - test_implied_vol.py
  - test_monte_carlo.py
  - test_trees.py
- README.md
- requirements.txt
- pyproject.toml
```


## Installation

Clone the repository and install the package in editable mode:

```bash
pip install -e ".[dev]"
```

If you do not need the development dependencies:

```bash
pip install -e .
```

## Quickstart

### Black-Scholes pricing

```python
from pricing_engine.bs import bs_call_price, bs_put_price

call = bs_call_price(S=100, K=100, r=0.05, sigma=0.2, T=1.0, q=0.0)
put = bs_put_price(S=100, K=100, r=0.05, sigma=0.2, T=1.0, q=0.0)

print(call)
print(put)
```

### Greeks

```python
from pricing_engine.greeks import (
    delta_call,
    delta_put,
    calculate_gamma,
    calculate_vega,
    calculate_theta_call,
    calculate_theta_put,
    calculate_rho_call,
    calculate_rho_put,
)

delta = delta_call(100, 100, 0.05, 0.2, 1.0)
gamma = calculate_gamma(100, 100, 0.05, 0.2, 1.0)
vega = calculate_vega(100, 100, 0.05, 0.2, 1.0)
theta = calculate_theta_call(100, 100, 0.05, 0.2, 1.0)
rho = calculate_rho_call(100, 100, 0.05, 0.2, 1.0)
```

### Implied volatility

```python
from pricing_engine.bs import bs_call_price
from pricing_engine.implied_vol import implied_volatility

market_price = bs_call_price(100, 100, 0.05, 0.2, 1.0)
iv = implied_volatility(market_price, "C", 100, 100, 0.05, 1.0)

print(iv)
```

### Binomial tree

```python
from pricing_engine.trees import binomial_prices

european_call = binomial_prices(
    S=100,
    K=100,
    r=0.05,
    sigma=0.2,
    T=1.0,
    option_type="C",
    exercise="european",
    q=0.0,
    steps=500,
)

american_put = binomial_prices(
    S=80,
    K=100,
    r=0.05,
    sigma=0.2,
    T=1.0,
    option_type="P",
    exercise="american",
    q=0.0,
    steps=500,
)
```

### Monte Carlo Asian option

```python
from pricing_engine.monte_carlo import asian_option_mc_price

result = asian_option_mc_price(
    S=100,
    K=100,
    r=0.05,
    sigma=0.2,
    T=1.0,
    option_type="C",
    q=0.0,
    n_paths=100000,
    n_steps=252,
    seed=42,
    return_detail=True,
)

print(result)
```

## Models Implemented

### Black-Scholes

The Black-Scholes module prices European call and put options under a geometric Brownian motion assumption with constant interest rate, constant volatility, and optional continuous dividend yield.

### Greeks

The Greeks module computes:

- Delta
- Gamma
- Vega
- Theta
- Rho

Finite difference checks are included in the test suite.

Theta is implemented using the market convention for time decay.

### Implied Volatility

The implied volatility module solves for the volatility that matches a given market option price.

It uses Brent's root-finding method and validates arbitrage bounds before solving.

### Binomial Tree

The tree module implements a Cox-Ross-Rubinstein binomial model for:

- European call and put options
- American call and put options
- Early exercise decisions through backward induction

### Monte Carlo

The Monte Carlo module simulates GBM paths and prices arithmetic Asian options.

It also returns, optionally:

- price
- standard error
- 95% confidence interval
- number of paths
- number of time steps

## Tests

Run the test suite with:

```bash
pytest
```

The tests cover:

- Black-Scholes known values
- Put-call parity
- Arbitrage bounds
- Greeks finite difference validation
- Implied volatility round-trip checks
- Binomial tree convergence to Black-Scholes
- American vs European option properties
- Monte Carlo reproducibility
- Monte Carlo confidence intervals

## Notebook

A sanity-check notebook is available at:

notebooks/01_sanity_checks.ipynb

It includes:

- Black-Scholes pricing examples
- Put-call parity check
- Greeks table
- Implied volatility round-trip
- Binomial convergence plot
- American vs European comparison
- Monte Carlo convergence
- Asian vs European comparison

## Streamlit App

A Streamlit demo app is planned at app/app.py.

Run it with:

```bash
streamlit run app/app.py
```

## Assumptions

This project assumes:

- constant risk-free rate
- constant volatility
- continuous dividend yield
- geometric Brownian motion dynamics
- no transaction costs
- no liquidity constraints
- no discrete dividends
- no stochastic rates
- no stochastic volatility


## Disclaimer

This project is for educational purposes only and does not constitute investment advice. The author is not responsible for any financial decisions made based on this project.
