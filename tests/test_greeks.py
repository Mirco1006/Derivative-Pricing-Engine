import math
import pytest

from pricing_engine.bs import bs_call_price, bs_put_price
from pricing_engine.greeks import (
    delta_call,
    delta_put,
    calculate_gamma,
    calculate_vega,
    calculate_rho_call,
    calculate_rho_put,
    calculate_theta_call,
    calculate_theta_put,
)

S = 100
K = 100
r = 0.05
sigma = 0.2
T = 1.0
q = 0.0
h_spot = 0.1
h_vol = 0.001
h_rate = 0.001
h_time = 0.0001


def test_standard_delta_call():
    assert 0 <= delta_call(S, K, r, sigma, T, q) <= 1


def test_standard_delta_put():
    assert -1 <= delta_put(S, K, r, sigma, T, q) <= 0


def test_delta_call_put():
    delta_C = delta_call(S, K, r, sigma, T, q)
    delta_P = delta_put(S, K, r, sigma, T, q)
    assert abs((delta_C - delta_P) - math.exp(-q * T)) < 1e-6


def test_finite_difference_delta_call():
    delta_C = delta_call(S, K, r, sigma, T, q)
    finite_diff = (
        bs_call_price(S + h_spot, K, r, sigma, T)
        - bs_call_price(S - h_spot, K, r, sigma, T)
    ) / (2 * h_spot)
    assert abs(delta_C - finite_diff) < 1e-3


def test_finite_difference_put_call():
    delta_P = delta_put(S, K, r, sigma, T, q)
    finite_diff = (
        bs_put_price(S + h_spot, K, r, sigma, T)
        - bs_put_price(S - h_spot, K, r, sigma, T)
    ) / (2 * h_spot)
    assert abs(delta_P - finite_diff) < 1e-3


def test_delta_call_put_with_dividends():
    q_div = 0.03
    delta_C = delta_call(S, K, r, sigma, T, q_div)
    delta_P = delta_put(S, K, r, sigma, T, q_div)
    assert abs((delta_C - delta_P) - math.exp(-q_div * T)) < 1e-6


def test_standard_gamma():
    assert calculate_gamma(S, K, r, sigma, T, q) > 0


def test_finite_diff_gamma():
    gamma = calculate_gamma(S, K, r, sigma, T, q)
    finite_diff = (
        bs_call_price(S + h_spot, K, r, sigma, T)
        - (2 * bs_call_price(S, K, r, sigma, T))
        + bs_call_price(S - h_spot, K, r, sigma, T)
    ) / (h_spot**2)
    assert abs(gamma - finite_diff) < 1e-3


def test_standard_vega():
    assert calculate_vega(S, K, r, sigma, T) > 0


def test_finite_diff_vega():
    vega = calculate_vega(S, K, r, sigma, T)
    finite_diff = (
        bs_call_price(S, K, r, sigma + h_vol, T)
        - bs_call_price(S, K, r, sigma - h_vol, T)
    ) / (2 * h_vol)
    assert abs(vega - finite_diff) < 1e-3


def test_standard_rho_call():
    assert calculate_rho_call(S, K, r, sigma, T) > 0


def test_standard_rho_put():
    assert calculate_rho_put(S, K, r, sigma, T) < 0


def test_rho_call_put_relation():
    rho_call = calculate_rho_call(S, K, r, sigma, T)
    rho_put = calculate_rho_put(S, K, r, sigma, T)
    assert abs((rho_call - rho_put) - K * T * math.exp(-r * T)) < 1e-3


def test_finite_diff_rho_call():
    rho_call = calculate_rho_call(S, K, r, sigma, T)
    finite_diff = (
        bs_call_price(S, K, r + h_rate, sigma, T)
        - bs_call_price(S, K, r - h_rate, sigma, T)
    ) / (2 * h_rate)
    assert abs(rho_call - finite_diff) < 1e-3


def test_finite_diff_rho_put():
    rho_put = calculate_rho_put(S, K, r, sigma, T)
    finite_diff = (
        bs_put_price(S, K, r + h_rate, sigma, T)
        - bs_put_price(S, K, r - h_rate, sigma, T)
    ) / (2 * h_rate)
    assert abs(rho_put - finite_diff) < 1e-3


def test_standard_theta_call():
    assert calculate_theta_call(S, K, r, sigma, T) < 0


def test_finite_diff_theta_call():
    theta = calculate_theta_call(S, K, r, sigma, T)
    finite_diff = -(
        bs_call_price(S, K, r, sigma, T + h_time)
        - bs_call_price(S, K, r, sigma, T - h_time)
    ) / (2 * h_time)
    assert abs(theta - finite_diff) < 1e-3


def test_finite_diff_theta_put():
    theta = calculate_theta_put(S, K, r, sigma, T)
    finite_diff = -(
        bs_put_price(S, K, r, sigma, T + h_time)
        - bs_put_price(S, K, r, sigma, T - h_time)
    ) / (2 * h_time)
    assert abs(theta - finite_diff) < 1e-3


def test_delta_negative_maturity_raises():
    with pytest.raises(ValueError):
        delta_call(S, K, r, sigma, 0)


def test_delta_negative_sigma_raises():
    with pytest.raises(ValueError):
        delta_put(S, K, r, 0, T)


def test_gamma_negative_maturity():
    with pytest.raises(ValueError):
        calculate_gamma(S, K, r, sigma, 0)


def test_vega_negative_sigma():
    with pytest.raises(ValueError):
        calculate_vega(S, K, r, 0, T)


def test_rho_negative_maturity_raises():
    with pytest.raises(ValueError):
        calculate_rho_call(S, K, r, sigma, 0)


def test_theta_negative_sigma_raises():
    with pytest.raises(ValueError):
        calculate_theta_call(S, K, r, 0, T)
