import math
import pytest

from pricing_engine.bs import bs_call_price, bs_put_price
from pricing_engine.implied_vol import implied_volatility

S = 100
K = 100
r = 0.05
sigma = 0.2
T = 1
q = 0

sigma_true = 0.2


def test_implied_vol_call():
    price = bs_call_price(S, K, r, sigma, T)
    assert abs(implied_volatility(price, "C", S, K, r, T) - sigma_true) < 1e-6


def test_implied_vol_put():
    price = bs_put_price(S, K, r, sigma, T)
    assert abs(implied_volatility(price, "P", S, K, r, T) - sigma_true) < 1e-6


def test_call_price_too_low():
    price = max(S * math.exp(-q * T) - K * math.exp(-r * T), 0) - 1
    with pytest.raises(ValueError):
        implied_volatility(price, "C", S, K, r, T)


def test_call_price_too_high():
    price = S * math.exp(-q * T) + 1
    with pytest.raises(ValueError):
        implied_volatility(price, "C", S, K, r, T)


def test_put_price_too_low():
    price = max(K * math.exp(-r * T) - S * math.exp(-q * T), 0) - 1
    with pytest.raises(ValueError):
        implied_volatility(price, "P", S, K, r, T)


def test_put_price_too_high():
    price = K * math.exp(-r * T) + 1
    with pytest.raises(ValueError):
        implied_volatility(price, "P", S, K, r, T)


def test_invalid_option_type():
    price = K * math.exp(-r * T) - 1
    with pytest.raises(ValueError):
        implied_volatility(price, "X", S, K, r, T)


def test_invalid_maturity():
    price = K * math.exp(-r * T) - 1
    with pytest.raises(ValueError):
        implied_volatility(price, "P", S, K, r, 0)


def test_implied_vol_consistency():
    price1 = bs_call_price(S, K, r, 0.15, T)
    price2 = bs_call_price(S, K, r, 0.3, T)
    implied_vol_1 = implied_volatility(price1, "C", S, K, r, T)
    implied_vol_2 = implied_volatility(price2, "C", S, K, r, T)
    assert implied_vol_1 < implied_vol_2


def test_implied_vol_call_deep_ITM():
    price = bs_call_price(150, K, r, sigma, T)
    assert abs(implied_volatility(price, "C", 150, K, r, T) - sigma_true) < 1e-5


def test_implied_vol_call_deep_OTM():
    price = bs_call_price(50, K, r, sigma, T)
    assert abs(implied_volatility(price, "C", 50, K, r, T) - sigma_true) < 1e-5


def test_implied_vol_put_deep_ITM():
    price = bs_put_price(50, K, r, sigma, T)
    assert abs(implied_volatility(price, "P", 50, K, r, T) - sigma_true) < 1e-5


def test_implied_vol_put_deep_OTM():
    price = bs_put_price(150, K, r, sigma, T)
    assert abs(implied_volatility(price, "P", 150, K, r, T) - sigma_true) < 1e-5
