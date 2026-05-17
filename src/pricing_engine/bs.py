import math
from scipy.stats import norm
from .curves import discount_factor, dividend_discount_factor


def _validate_inputs(S, K, T, sigma):
    """Raise ValueError if any of the Black–Scholes inputs is invalid."""
    if S <= 0:
        raise ValueError("Spot price S must be strictly positive.")
    if K <= 0:
        raise ValueError("Strike K must be strictly positive.")
    if T < 0:
        raise ValueError("Time to maturity T cannot be negative.")
    if sigma < 0:
        raise ValueError("Volatility sigma cannot be negative.")


def bs_call_price(S, K, r, sigma, T, q=0):
    """
    Black–Scholes price of a European call option.

    C = S * exp(-q*T) * N(d1) - K * exp(-r*T) * N(d2)

    Parameters
    ----------
    S : float
        Spot price of the underlying. Must be strictly positive.
    K : float
        Strike price. Must be strictly positive.
    r : float
        Continuously compounded risk-free rate.
    sigma : float
        Volatility of the underlying (annualised). Must be non-negative.
    T : float
        Time to maturity in years. Must be non-negative.
    q : float, optional
        Continuous dividend yield. Default is 0.0.

    Returns
    -------
    float
        European call price. Returns intrinsic value at T = 0 and the
        discounted forward payoff when sigma = 0.
    """
    _validate_inputs(S, K, T, sigma)

    if T <= 1e-12:
        return max(S - K, 0.0)

    if sigma <= 1e-12:
        return discount_factor(r, T) * max(S * math.exp((r - q) * T) - K, 0.0)

    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * (math.sqrt(T)))
    d2 = d1 - sigma * (math.sqrt(T))

    return S * dividend_discount_factor(q, T) * norm.cdf(d1) - K * discount_factor(
        r, T
    ) * norm.cdf(d2)


def bs_put_price(S, K, r, sigma, T, q=0):
    """
    Black–Scholes price of a European put option.

    P = K * exp(-r*T) * N(-d2) - S * exp(-q*T) * N(-d1)

    Parameters
    ----------
    S : float
        Spot price of the underlying. Must be strictly positive.
    K : float
        Strike price. Must be strictly positive.
    r : float
        Continuously compounded risk-free rate.
    sigma : float
        Volatility of the underlying (annualised). Must be non-negative.
    T : float
        Time to maturity in years. Must be non-negative.
    q : float, optional
        Continuous dividend yield. Default is 0.0.

    Returns
    -------
    float
        European put price. Returns intrinsic value at T = 0 and the
        discounted forward payoff when sigma = 0.
    """
    _validate_inputs(S, K, T, sigma)

    if T <= 1e-12:
        return max(K - S, 0.0)

    if sigma <= 1e-12:
        return discount_factor(r, T) * max(K - (S * math.exp((r - q) * T)), 0.0)

    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * (math.sqrt(T)))
    d2 = d1 - sigma * (math.sqrt(T))

    return K * discount_factor(r, T) * norm.cdf(-d2) - S * dividend_discount_factor(
        q, T
    ) * norm.cdf(-d1)
