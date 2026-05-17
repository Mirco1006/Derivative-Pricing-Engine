import pytest
import math

from pricing_engine.bs import bs_call_price, bs_put_price

S = 100
K = 100
r = 0.05
sigma = 0.2
T = 1.0
q = 0.0


def test_standard_call_price():
    assert abs(bs_call_price(S, K, r, sigma, T, q) - 10.4506) < 1e-3


def test_standard_put_price():
    assert abs(bs_put_price(S, K, r, sigma, T, q) - 5.5735) < 1e-3


def test_call_put_parity():
    call = bs_call_price(S, K, r, sigma, T, q)
    put = bs_put_price(S, K, r, sigma, T, q)
    assert abs((call - put) - (S * math.exp(-q * T) - K * math.exp(-r * T))) < 1e-8


def test_non_negative_call_price():
    assert bs_call_price(1, K, r, sigma, T, q) >= 0.0


def test_non_negative_put_price():
    assert bs_put_price(1000, K, r, sigma, T, q) >= 0.0


def test_arbitrage_call_bounds():
    call = bs_call_price(S, K, r, sigma, T, q)
    lower = max(S * math.exp(-q * T) - K * math.exp(-r * T), 0)
    upper = S * math.exp(-q * T)
    assert lower <= call <= upper


def test_arbitrage_put_bounds():
    put = bs_put_price(S, K, r, sigma, T, q)
    lower = max(K * math.exp(-r * T) - S * math.exp(-q * T), 0)
    upper = K * math.exp(-r * T)
    assert lower <= put <= upper


def test_call_spot_monotonicity():
    call_80 = bs_call_price(80, K, r, sigma, T, q)
    call_100 = bs_call_price(S, K, r, sigma, T, q)
    call_120 = bs_call_price(120, K, r, sigma, T, q)
    assert call_80 < call_100 < call_120


def test_put_spot_monotonicity():
    put_80 = bs_put_price(80, K, r, sigma, T, q)
    put_100 = bs_put_price(S, K, r, sigma, T, q)
    put_120 = bs_put_price(120, K, r, sigma, T, q)
    assert put_80 > put_100 > put_120


def test_call_strike_monotonicity():
    call_80 = bs_call_price(S, 80, r, sigma, T, q)
    call_100 = bs_call_price(S, K, r, sigma, T, q)
    call_120 = bs_call_price(S, 120, r, sigma, T, q)
    assert call_80 > call_100 > call_120


def test_put_strike_monotonicity():
    put_80 = bs_put_price(S, 80, r, sigma, T, q)
    put_100 = bs_put_price(S, K, r, sigma, T, q)
    put_120 = bs_put_price(S, 120, r, sigma, T, q)
    assert put_80 < put_100 < put_120


def test_call_sigma_monotonicity():
    call_01 = bs_call_price(S, K, r, 0.1, T, q)
    call_02 = bs_call_price(S, K, r, sigma, T, q)
    call_04 = bs_call_price(S, K, r, 0.4, T, q)
    assert call_01 < call_02 < call_04


def test_put_sigma_monotonicity():
    put_01 = bs_put_price(S, K, r, 0.1, T, q)
    put_02 = bs_put_price(S, K, r, sigma, T, q)
    put_04 = bs_put_price(S, K, r, 0.4, T, q)
    assert put_01 < put_02 < put_04


def test_call_zero_maturity():
    assert bs_call_price(120, K, r, 0.2, 0) == 20.0


def test_put_zero_maturity():
    assert bs_put_price(80, K, r, 0.2, 0) == 20.0


def test_call_zero_volatility():
    assert (
        abs(
            bs_call_price(S, K, r, 0, T)
            - (math.exp(-r * T) * max(S * math.exp((r - q) * T) - K, 0))
        )
        < 1e-3
    )


def test_put_zero_volatility():
    assert (
        abs(
            bs_put_price(S, K, r, 0, T)
            - (math.exp(-r * T) * max(K - S * math.exp((r - q) * T), 0))
        )
        < 1e-3
    )


def test_call_put_parity_with_dividends():
    q_div = 0.03
    call = bs_call_price(S, K, r, sigma, T, q_div)
    put = bs_put_price(S, K, r, sigma, T, q_div)
    assert abs((call - put) - (S * math.exp(-q_div * T) - K * math.exp(-r * T))) < 1e-8


def test_call_price_decreases_with_dividends():
    call_no_div = bs_call_price(S, K, r, sigma, T, q=0.0)
    call_div = bs_call_price(S, K, r, sigma, T, q=0.05)
    assert call_div < call_no_div


def test_put_price_increases_with_dividends():
    put_no_div = bs_put_price(S, K, r, sigma, T, q=0.0)
    put_div = bs_put_price(S, K, r, sigma, T, q=0.05)
    assert put_div > put_no_div


def test_call_negative_spot_raises():
    with pytest.raises(ValueError):
        bs_call_price(-1.0, K, r, sigma, T)


def test_put_negative_spot_raises():
    with pytest.raises(ValueError):
        bs_put_price(-1.0, K, r, sigma, T)


def test_call_negative_strike_raises():
    with pytest.raises(ValueError):
        bs_call_price(S, -1.0, r, sigma, T)


def test_put_negative_strike_raises():
    with pytest.raises(ValueError):
        bs_put_price(S, -1.0, r, sigma, T)


def test_call_negative_sigma_raises():
    with pytest.raises(ValueError):
        bs_call_price(S, K, r, -0.1, T)


def test_put_negative_sigma_raises():
    with pytest.raises(ValueError):
        bs_put_price(S, K, r, -0.1, T)


def test_call_negative_maturity_raises():
    with pytest.raises(ValueError):
        bs_call_price(S, K, r, sigma, -1.0)


def test_put_negative_maturity_raises():
    with pytest.raises(ValueError):
        bs_put_price(S, K, r, sigma, -1.0)
