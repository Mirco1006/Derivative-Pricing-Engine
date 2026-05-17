import math
import pytest

from pricing_engine.bs import bs_call_price, bs_put_price
from pricing_engine.trees import binomial_prices

S = 100
K = 100
r = 0.05
T = 1
sigma = 0.2
q = 0


def test_european_call_price():
    assert binomial_prices(S, K, r, sigma, T, "C", "european", q) >= 0


def test_european_put_price():
    assert binomial_prices(S, K, r, sigma, T, "P", "european", q) >= 0


def test_american_call_price():
    assert binomial_prices(S, K, r, sigma, T, "C", "american", q) >= 0


def test_american_put_price():
    assert binomial_prices(S, K, r, sigma, T, "P", "american", q) >= 0


def test_bs_price_convergence_european_call():
    bs_price = bs_call_price(S, K, r, sigma, T)
    binomial_price = binomial_prices(S, K, r, sigma, T, "C", "european", q, 500)
    assert abs(bs_price - binomial_price) < 0.01


def test_bs_price_convergence_european_put():
    bs_price = bs_put_price(S, K, r, sigma, T)
    binomial_price = binomial_prices(S, K, r, sigma, T, "P", "european", q, 500)
    assert abs(bs_price - binomial_price) < 0.01


def test_price_american_european_call():
    european = binomial_prices(S, K, r, sigma, T, "C", "european", q, 500)
    american = binomial_prices(S, K, r, sigma, T, "C", "american", q, 500)
    tolerance = 1e-8
    assert american + tolerance > european


def test_price_american_european_put():
    european = binomial_prices(S, K, r, sigma, T, "P", "european", q, 500)
    american = binomial_prices(S, K, r, sigma, T, "P", "american", q, 500)
    tolerance = 1e-8
    assert american + tolerance > european


def test_american_european_call_no_dividends():
    european = binomial_prices(S, K, r, sigma, T, "C", "european", q, 500)
    american = binomial_prices(S, K, r, sigma, T, "C", "american", q, 500)
    assert abs(american - european) < 1e-6


def test_american_european_put_ITM():
    european = binomial_prices(80, K, r, sigma, T, "P", "european", q, 500)
    american = binomial_prices(80, K, r, sigma, T, "P", "american", q, 500)
    assert american > european


def test_call_put_parity():
    call = binomial_prices(S, K, r, sigma, T, "C", "european", q, 500)
    put = binomial_prices(S, K, r, sigma, T, "P", "european", q, 500)
    assert abs((call - put) - (S * math.exp(-q * T) - K * math.exp(-r * T))) < 0.01


def test_monotonicity_call_spot():
    call_80 = binomial_prices(80, K, r, sigma, T, "C", "european", q, 500)
    call_100 = binomial_prices(100, K, r, sigma, T, "C", "european", q, 500)
    call_120 = binomial_prices(120, K, r, sigma, T, "C", "european", q, 500)
    assert call_80 < call_100 < call_120


def test_monotonicity_put_spot():
    put_80 = binomial_prices(80, K, r, sigma, T, "P", "european", q, 500)
    put_100 = binomial_prices(100, K, r, sigma, T, "P", "european", q, 500)
    put_120 = binomial_prices(120, K, r, sigma, T, "P", "european", q, 500)
    assert put_120 < put_100 < put_80


def test_monotonicity_call_strike():
    call_80 = binomial_prices(S, 80, r, sigma, T, "C", "european", q, 500)
    call_100 = binomial_prices(S, 100, r, sigma, T, "C", "european", q, 500)
    call_120 = binomial_prices(S, 120, r, sigma, T, "C", "european", q, 500)
    assert call_80 > call_100 > call_120


def test_monotonicity_put_strike():
    put_80 = binomial_prices(S, 80, r, sigma, T, "P", "european", q, 500)
    put_100 = binomial_prices(S, 100, r, sigma, T, "P", "european", q, 500)
    put_120 = binomial_prices(S, 120, r, sigma, T, "P", "european", q, 500)
    assert put_120 > put_100 > put_80


def test_monotonicity_call_sigma():
    call_01 = binomial_prices(S, K, r, 0.1, T, "C", "european", q, 500)
    call_02 = binomial_prices(S, K, r, 0.2, T, "C", "european", q, 500)
    call_04 = binomial_prices(S, K, r, 0.4, T, "C", "european", q, 500)
    assert call_01 < call_02 < call_04


def test_monotonicity_put_sigma():
    put_01 = binomial_prices(S, K, r, 0.1, T, "P", "european", q, 500)
    put_02 = binomial_prices(S, K, r, 0.2, T, "P", "european", q, 500)
    put_04 = binomial_prices(S, K, r, 0.4, T, "P", "european", q, 500)
    assert put_01 < put_02 < put_04


def test_put_negative_spot_raises():
    with pytest.raises(ValueError):
        binomial_prices(-1.0, K, r, sigma, T, "P", "european", q, 100)


def test_put_negative_strike_raises():
    with pytest.raises(ValueError):
        binomial_prices(S, -1.0, r, sigma, T, "P", "european", q, 100)


def test_put_negative_sigma_raises():
    with pytest.raises(ValueError):
        binomial_prices(S, K, r, -0.1, T, "P", "european", q, 100)


def test_put_negative_maturity_raises():
    with pytest.raises(ValueError):
        binomial_prices(S, K, r, sigma, -0.1, "P", "european", q, 100)


def test_put_steps_raises():
    with pytest.raises(ValueError):
        binomial_prices(S, K, r, sigma, T, "P", "european", q, 0)


def test_invalid_option_type_raises():
    with pytest.raises(ValueError):
        binomial_prices(S, K, r, sigma, T, "X", "european", q, 100)


def test_invalid_exercise_type_raises():
    with pytest.raises(ValueError):
        binomial_prices(S, K, r, sigma, T, "P", "italian", q, 100)
