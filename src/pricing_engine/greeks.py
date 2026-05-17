import math
from scipy.stats import norm
from .curves import discount_factor, dividend_discount_factor


def calculate_d1(S, K, r, sigma, T, q=0):
    """
    Black–Scholes d1 term.

    d1 = [ln(S/K) + (r - q + 0.5 * sigma**2) * T] / (sigma * sqrt(T))

    Parameters
    ----------
    S : float
        Spot price. Must be strictly positive.
    K : float
        Strike price. Must be strictly positive.
    r : float
        Continuously compounded risk-free rate.
    sigma : float
        Volatility (annualised). Must be strictly positive.
    T : float
        Time to maturity in years. Must be strictly positive.
    q : float, optional
        Continuous dividend yield. Default is 0.0.

    Returns
    -------
    float
        Value of d1.
    """
    if T <= 0 or sigma <= 0:
        raise ValueError("Greeks undefined for T <= 0 or sigma <= 0")

    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * (math.sqrt(T)))
    return d1


def calculate_d2(d1, sigma, T):
    """
    Black–Scholes d2 term: d2 = d1 - sigma * sqrt(T).

    Parameters
    ----------
    d1 : float
        Pre-computed d1 value.
    sigma : float
        Volatility (annualised).
    T : float
        Time to maturity in years.

    Returns
    -------
    float
        Value of d2.
    """
    d2 = d1 - sigma * (math.sqrt(T))
    return d2


def delta_call(S, K, r, sigma, T, q=0):
    """
    Delta of a European call: exp(-q*T) * N(d1).

    Sensitivity of the call price to the spot. Lies in [0, exp(-q*T)].
    """
    d1 = calculate_d1(S, K, r, sigma, T, q)
    delta = dividend_discount_factor(q, T) * norm.cdf(d1)
    return delta


def delta_put(S, K, r, sigma, T, q=0):
    """
    Delta of a European put: exp(-q*T) * (N(d1) - 1).

    Sensitivity of the put price to the spot. Lies in [-exp(-q*T), 0].
    """
    d1 = calculate_d1(S, K, r, sigma, T, q)
    delta = dividend_discount_factor(q, T) * (norm.cdf(d1) - 1)
    return delta


def calculate_gamma(S, K, r, sigma, T, q=0):
    """
    Gamma of a European option: exp(-q*T) * phi(d1) / (S * sigma * sqrt(T)).

    Second derivative of price w.r.t. spot. Identical for call and put.
    Always strictly positive for vanilla options.
    """
    d1 = calculate_d1(S, K, r, sigma, T, q)
    gamma = (dividend_discount_factor(q, T) * norm.pdf(d1)) / (S * sigma * math.sqrt(T))
    return gamma


def calculate_vega(S, K, r, sigma, T, q=0):
    """
    Vega of a European option: S * exp(-q*T) * phi(d1) * sqrt(T).

    Sensitivity of price to a one-unit change in volatility.
    Identical for call and put. Always strictly positive.

    Note: not scaled by 1/100. Multiply by 0.01 for sensitivity to a 1% vol move.
    """
    d1 = calculate_d1(S, K, r, sigma, T, q)
    vega = S * dividend_discount_factor(q, T) * norm.pdf(d1) * math.sqrt(T)
    return vega


def calculate_rho_call(S, K, r, sigma, T, q=0):
    """
    Rho of a European call: K * T * exp(-r*T) * N(d2).

    Sensitivity of the call price to the risk-free rate. Always positive.
    """
    d1 = calculate_d1(S, K, r, sigma, T, q)
    d2 = calculate_d2(d1, sigma, T)
    rho = K * T * discount_factor(r, T) * norm.cdf(d2)
    return rho


def calculate_rho_put(S, K, r, sigma, T, q=0):
    """
    Rho of a European put: -K * T * exp(-r*T) * N(-d2).

    Sensitivity of the put price to the risk-free rate. Always negative.
    """
    d1 = calculate_d1(S, K, r, sigma, T, q)
    d2 = calculate_d2(d1, sigma, T)
    rho = -K * T * discount_factor(r, T) * norm.cdf(-d2)
    return rho


def calculate_theta_call(S, K, r, sigma, T, q=0):
    """
    Theta of a European call (market convention: -dV/dT).

    Theta_call = -(S * exp(-q*T) * phi(d1) * sigma) / (2 * sqrt(T))
                 - r * K * exp(-r*T) * N(d2)
                 + q * S * exp(-q*T) * N(d1)

    Represents the time decay of the option per unit of calendar time
    (per year, not per day). Generally negative for standard parameters.
    """
    d1 = calculate_d1(S, K, r, sigma, T, q)
    d2 = calculate_d2(d1, sigma, T)
    theta = (
        -(
            (S * dividend_discount_factor(q, T) * norm.pdf(d1) * sigma)
            / (2 * math.sqrt(T))
        )
        - (r * K * discount_factor(r, T) * norm.cdf(d2))
        + (q * S * dividend_discount_factor(q, T) * norm.cdf(d1))
    )
    return theta


def calculate_theta_put(S, K, r, sigma, T, q=0):
    """
    Theta of a European put (market convention: -dV/dT).

    Theta_put = -(S * exp(-q*T) * phi(d1) * sigma) / (2 * sqrt(T))
                + r * K * exp(-r*T) * N(-d2)
                - q * S * exp(-q*T) * N(-d1)

    Represents the time decay of the option per unit of calendar time.
    Sign depends on whether dividends/rates dominate the decay term.
    """
    d1 = calculate_d1(S, K, r, sigma, T, q)
    d2 = calculate_d2(d1, sigma, T)
    theta = (
        -(
            (S * dividend_discount_factor(q, T) * norm.pdf(d1) * sigma)
            / (2 * math.sqrt(T))
        )
        + (r * K * discount_factor(r, T) * norm.cdf(-d2))
        - (q * S * dividend_discount_factor(q, T) * norm.cdf(-d1))
    )
    return theta
