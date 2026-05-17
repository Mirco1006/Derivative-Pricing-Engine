import numpy as np
import math
from .curves import discount_factor


def asian_option_mc_price(
    S,
    K,
    r,
    sigma,
    T,
    option_type,
    q=0,
    n_paths=10000,
    n_steps=252,
    seed=None,
    return_detail=False,
):
    """
    Monte Carlo price of an arithmetic Asian option under GBM dynamics.

    Simulates ``n_paths`` price paths with ``n_steps`` time steps each.
    The Asian payoff is based on the arithmetic average of simulated prices,
    excluding the initial spot (i.e. the average is taken over steps 1 to T).

    Parameters
    ----------
    S : float
        Spot price of the underlying. Must be strictly positive.
    K : float
        Strike price. Must be strictly positive.
    r : float
        Continuously compounded risk-free rate.
    sigma : float
        Volatility of the underlying (annualised). Must be strictly positive.
    T : float
        Time to maturity in years. Must be strictly positive.
    option_type : str
        ``"C"`` for call, ``"P"`` for put.
    q : float, optional
        Continuous dividend yield. Default is 0.0.
    n_paths : int, optional
        Number of simulated paths. Default is 10 000.
    n_steps : int, optional
        Number of time steps per path. Default is 252.
    seed : int or None, optional
        Random seed for reproducibility. Default is None.
    return_detail : bool, optional
        If ``False`` (default), returns the price as a float.
        If ``True``, returns a dict with keys ``price``, ``standard_error``,
        ``ci_lower``, ``ci_upper``, ``n_paths``, ``n_steps``.

    Returns
    -------
    float or dict
        Option price, or a detail dict if ``return_detail=True``.

    Raises
    ------
    ValueError
        If any input is invalid.
    """
    if S <= 0:
        raise ValueError("Spot price S must be strictly positive")
    if K <= 0:
        raise ValueError("Strike price K must be strictly positive")
    if sigma <= 0:
        raise ValueError("Volatility sigma must be strictly positive")
    if T < 1e-12:
        raise ValueError("Time to maturity must be strictly positive")
    if option_type not in ("C", "P"):
        raise ValueError("Option type should be C for call and P for put")
    if q < 0:
        raise ValueError("Dividend yield q cannot be negative")
    if n_paths <= 0:
        raise ValueError("Number of paths should be strictly positive")
    if n_steps <= 0:
        raise ValueError("Number of steps n_steps should be strictly positive")

    dt = T / n_steps

    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(size=(n_paths, n_steps))

    prices = np.zeros((n_paths, n_steps + 1))
    prices[:, 0] = S

    for t in range(1, n_steps + 1):
        prices[:, t] = prices[:, t - 1] * np.exp(
            ((r - q - 0.5 * (sigma**2)) * dt) + (sigma * math.sqrt(dt) * Z[:, t - 1])
        )

    mean_price = prices[:, 1:].mean(axis=1)

    if option_type == "C":
        asian_price = np.maximum(0, mean_price - K)
    else:
        asian_price = np.maximum(0, K - mean_price)

    actualized_price = asian_price * discount_factor(r, T)
    final_price = float(actualized_price.mean())

    if not return_detail:
        return final_price
    else:
        std_error = actualized_price.std(ddof=1) / math.sqrt(n_paths)
        lower = final_price - 1.96 * std_error
        upper = final_price + 1.96 * std_error
        return {
            "price": final_price,
            "standard_error": std_error,
            "ci_lower": lower,
            "ci_upper": upper,
            "n_paths": n_paths,
            "n_steps": n_steps,
        }
