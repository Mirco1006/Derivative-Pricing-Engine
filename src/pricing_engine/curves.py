import numpy as np


def discount_factor(r, T):
    """
    Continuous compounding discount factor.

    DF = exp(-r * T)

    Parameters
    ----------
    r : float
        Continuously compounded risk-free rate.
    T : float
        Time to maturity in years. Must be non-negative.

    Returns
    -------
    float
        Discount factor in [0, 1] for r >= 0, > 1 for r < 0.
    """
    if T < 0:
        raise ValueError("Time to maturity cannot be negative.")
    return np.exp(-r * T)


def dividend_discount_factor(q, T):
    """
    Continuous dividend discount factor.

    Used to discount the spot price by the continuous dividend yield.

    DDF = exp(-q * T)

    Parameters
    ----------
    q : float
        Continuous dividend yield.
    T : float
        Time to maturity in years. Must be non-negative.

    Returns
    -------
    float
        Dividend discount factor.
    """
    if T < 0:
        raise ValueError("Time to maturity cannot be negative.")
    return np.exp(-q * T)


def forward_price(S, r, q, T):
    """
    Forward price of an asset under continuous compounding.

    F = S * exp((r - q) * T)

    Parameters
    ----------
    S : float
        Spot price of the underlying. Must be strictly positive.
    r : float
        Continuously compounded risk-free rate.
    q : float
        Continuous dividend yield.
    T : float
        Time to maturity in years. Must be non-negative.

    Returns
    -------
    float
        No-arbitrage forward price.
    """
    if T < 0:
        raise ValueError("Time to maturity cannot be negative.")
    if S <= 0:
        raise ValueError("Spot price must be strictly positive.")
    return S * np.exp((r - q) * T)
