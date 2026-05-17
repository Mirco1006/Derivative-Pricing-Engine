import math
from .curves import discount_factor


def binomial_prices(
    S, K, r, sigma, T, option_type, exercise="european", q=0, steps=100
):
    """
    Cox–Ross–Rubinstein (CRR) binomial tree price for European or American options.

    Builds a recombining binomial tree under the risk-neutral measure and
    prices the option by backward induction. American options apply early
    exercise at each node.

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
    exercise : str, optional
        ``"european"`` or ``"american"``. Default is ``"european"``.
    q : float, optional
        Continuous dividend yield. Default is 0.0.
    steps : int, optional
        Number of time steps in the tree. Default is 100.

    Returns
    -------
    float
        Option price at time 0.

    Raises
    ------
    ValueError
        If any input is invalid or if the risk-neutral probability falls
        outside [0, 1].
    """
    if S <= 0:
        raise ValueError("The spot price S must be strictly positive")
    if K <= 0:
        raise ValueError("The strike price K must be strictly positive")
    if sigma <= 0:
        raise ValueError("The volatility sigma must be strictly positive")
    if T < 1e-12:
        raise ValueError("The time to maturity T cannot be negative or equal to 0")
    if option_type not in ("C", "P"):
        raise ValueError("Option type should be C for call or P for put")
    if exercise not in ("european", "american"):
        raise ValueError("Exercise must be european or american")
    if q < 0:
        raise ValueError("The dividend rate q cannot be negative")
    if steps < 1:
        raise ValueError("The number of steps cannot be 0 or negative")

    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1 / u
    p = (math.exp((r - q) * dt) - d) / (u - d)
    if not (0 <= p <= 1):
        raise ValueError("Risk-neutral probability is outside [0, 1]")

    terminal_prices = []

    for i in range(steps + 1):
        price = S * u**i * d ** (steps - i)
        terminal_prices.append(price)

    if option_type == "C":
        payoffs = [max(ST - K, 0) for ST in terminal_prices]
    else:
        payoffs = [max(K - ST, 0) for ST in terminal_prices]

    option_value = payoffs
    discount = discount_factor(r, dt)

    for step in range(steps - 1, -1, -1):
        new_values = []

        for i in range(0, step + 1):
            continuation = discount * (
                ((1 - p) * option_value[i]) + (p * option_value[i + 1])
            )
            if exercise == "american":
                S_node = S * (u**i) * (d ** (step - i))
                if option_type == "C":
                    payoff = max(0.0, S_node - K)
                else:
                    payoff = max(0.0, K - S_node)

                value = max(payoff, continuation)
            else:
                value = continuation

            new_values.append(value)
        option_value = new_values

    return option_value[0]
