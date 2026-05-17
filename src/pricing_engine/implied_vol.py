from scipy.optimize import brentq

from .bs import bs_call_price, bs_put_price
from .curves import dividend_discount_factor, discount_factor


def _validate_input(S, K, T, option):
    """Raise ValueError if core implied-volatility inputs are invalid."""

    if S <= 0:
        raise ValueError("Spot price cannot be smaller than 0")
    if K <= 0:
        raise ValueError("Strike price cannot be smaller or equal to 0")
    if T <= 1e-12:
        raise ValueError("Time to maturity should be bigger than 0")
    if option not in ("C", "P"):
        raise ValueError("Option type invalid. Please use C for call and P for put")


def calculate_lower_bound_call(S, K, T, r, q):
    """
    No-arbitrage lower bound for a European call price.

    Lower = max(0, S * exp(-q*T) - K * exp(-r*T))

    Parameters
    ----------
    S : float
        Spot price. Must be strictly positive.
    K : float
        Strike price. Must be strictly positive.
    T : float
        Time to maturity in years. Must be non-negative.
    r : float
        Continuously compounded risk-free rate.
    q : float
        Continuous dividend yield.

    Returns
    -------
    float
        Lower arbitrage bound for the call price.
    """
    return max(0, S * dividend_discount_factor(q, T) - K * discount_factor(r, T))


def calculate_upper_bound_call(S, T, q):
    """
    No-arbitrage upper bound for a European call price.

    Upper = S * exp(-q*T)

    Parameters
    ----------
    S : float
        Spot price. Must be strictly positive.
    T : float
        Time to maturity in years. Must be non-negative.
    q : float
        Continuous dividend yield.

    Returns
    -------
    float
        Upper arbitrage bound for the call price.
    """
    return S * dividend_discount_factor(q, T)


def calculate_lower_bound_put(S, K, T, r, q):
    """
    No-arbitrage lower bound for a European put price.

    Lower = max(0, K * exp(-r*T) - S * exp(-q*T))

    Parameters
    ----------
    S : float
        Spot price. Must be strictly positive.
    K : float
        Strike price. Must be strictly positive.
    T : float
        Time to maturity in years. Must be non-negative.
    r : float
        Continuously compounded risk-free rate.
    q : float
        Continuous dividend yield.

    Returns
    -------
    float
        Lower arbitrage bound for the put price.
    """
    return max(0, K * discount_factor(r, T) - S * dividend_discount_factor(q, T))


def calculate_upper_bound_put(K, T, r):
    """
    No-arbitrage upper bound for a European put price.

    Upper = K * exp(-r*T)

    Parameters
    ----------
    K : float
        Strike price. Must be strictly positive.
    T : float
        Time to maturity in years. Must be non-negative.
    r : float
        Continuously compounded risk-free rate.

    Returns
    -------
    float
        Upper arbitrage bound for the put price.
    """
    return K * discount_factor(r, T)


def _validate_bounds_call(market_price, S, K, T, r, q):
    """Raise ValueError if market_price violates call no-arbitrage bounds."""
    lower = calculate_lower_bound_call(S, K, T, r, q)
    upper = calculate_upper_bound_call(S, T, q)
    if not (lower <= market_price <= upper):
        raise ValueError("Invalid call price (arbitrage violation)")


def _validate_bounds_put(market_price, S, K, T, r, q):
    """Raise ValueError if market_price violates put no-arbitrage bounds."""
    lower = calculate_lower_bound_put(S, K, T, r, q)
    upper = calculate_upper_bound_put(K, T, r)
    if not (lower <= market_price <= upper):
        raise ValueError("Invalid put price (arbitrage violation)")


def implied_volatility(market_price, option_type, S, K, r, T, q=0):
    """
    Implied volatility of a European option via Brent's root-finding method.

    Solves for sigma such that BS(sigma) = market_price, where BS is the
    Black–Scholes call or put pricing formula. Validates arbitrage bounds
    before solving.

    Parameters
    ----------
    market_price : float
        Observed market price of the option.
    option_type : str
        ``"C"`` for call, ``"P"`` for put.
    S : float
        Spot price of the underlying. Must be strictly positive.
    K : float
        Strike price. Must be strictly positive.
    r : float
        Continuously compounded risk-free rate.
    T : float
        Time to maturity in years. Must be strictly positive.
    q : float, optional
        Continuous dividend yield. Default is 0.0.

    Returns
    -------
    float
        Implied volatility in the range (0, 5].

    Raises
    ------
    ValueError
        If inputs are invalid, if ``market_price`` violates no-arbitrage
        bounds, or if ``option_type`` is not ``"C"`` or ``"P"``.
    """
    _validate_input(S, K, T, option_type)

    if option_type == "C":
        _validate_bounds_call(market_price, S, K, T, r, q)

        def error(sigma):
            return bs_call_price(S, K, r, sigma, T, q) - market_price
    elif option_type == "P":
        _validate_bounds_put(market_price, S, K, T, r, q)

        def error(sigma):
            return bs_put_price(S, K, r, sigma, T, q) - market_price
    else:
        raise ValueError("Invalid option type. Please write C for call and P for put")

    sigma = brentq(error, 1e-12, 5)

    return sigma
