import pytest

from pricing_engine.bs import bs_call_price, bs_put_price
from pricing_engine.monte_carlo import asian_option_mc_price

S = 100
K = 100
r = 0.05
sigma = 0.2
T = 1.0
q = 0.0
seed = 42
n_paths = 20000
n_steps = 50


def test_positive_call_price():
    assert asian_option_mc_price(S, K, r, sigma, T, "C", q, n_paths, n_steps) >= 0


def test_positive_put_price():
    assert asian_option_mc_price(S, K, r, sigma, T, "P", q, n_paths, n_steps) >= 0


def test_same_price_seed():
    price1 = asian_option_mc_price(S, K, r, sigma, T, "C", q, n_paths, n_steps, seed)
    price2 = asian_option_mc_price(S, K, r, sigma, T, "C", q, n_paths, n_steps, seed)
    assert price1 == price2


def test_return_float():
    price = asian_option_mc_price(S, K, r, sigma, T, "C", q, n_paths, n_steps, seed)
    assert isinstance(price, float)


def test_return_dictionary():
    price_info = asian_option_mc_price(
        S, K, r, sigma, T, "C", q, n_paths, n_steps, seed, True
    )
    expected_keys = {
        "price",
        "standard_error",
        "ci_lower",
        "ci_upper",
        "n_paths",
        "n_steps",
    }
    assert expected_keys.issubset(price_info.keys())


def test_price_in_interval_and_standard_error():
    price_info = asian_option_mc_price(
        S, K, r, sigma, T, "C", q, n_paths, n_steps, seed, True
    )
    assert price_info["ci_lower"] <= price_info["price"] <= price_info["ci_upper"]
    assert price_info["standard_error"] >= 0


def test_standard_error_monotonicity():
    price_1 = asian_option_mc_price(
        S, K, r, sigma, T, "C", q, 2000, n_steps, seed, True
    )
    price_2 = asian_option_mc_price(
        S, K, r, sigma, T, "C", q, 20000, n_steps, seed, True
    )
    assert price_1["standard_error"] > price_2["standard_error"]


def test_monotonicity_call_spot():
    price_80 = asian_option_mc_price(80, K, r, sigma, T, "C", q, n_paths, n_steps, seed)
    price_100 = asian_option_mc_price(
        100, K, r, sigma, T, "C", q, n_paths, n_steps, seed
    )
    price_120 = asian_option_mc_price(
        120, K, r, sigma, T, "C", q, n_paths, n_steps, seed
    )
    assert price_80 < price_100 < price_120


def test_monotonicity_put_spot():
    price_80 = asian_option_mc_price(80, K, r, sigma, T, "P", q, n_paths, n_steps, seed)
    price_100 = asian_option_mc_price(
        100, K, r, sigma, T, "P", q, n_paths, n_steps, seed
    )
    price_120 = asian_option_mc_price(
        120, K, r, sigma, T, "P", q, n_paths, n_steps, seed
    )
    assert price_80 > price_100 > price_120


def test_monotonicity_call_strike():
    price_80 = asian_option_mc_price(S, 80, r, sigma, T, "C", q, n_paths, n_steps, seed)
    price_100 = asian_option_mc_price(
        S, 100, r, sigma, T, "C", q, n_paths, n_steps, seed
    )
    price_120 = asian_option_mc_price(
        S, 120, r, sigma, T, "C", q, n_paths, n_steps, seed
    )
    assert price_80 > price_100 > price_120


def test_monotonicity_put_strike():
    price_80 = asian_option_mc_price(S, 80, r, sigma, T, "P", q, n_paths, n_steps, seed)
    price_100 = asian_option_mc_price(
        S, 100, r, sigma, T, "P", q, n_paths, n_steps, seed
    )
    price_120 = asian_option_mc_price(
        S, 120, r, sigma, T, "P", q, n_paths, n_steps, seed
    )
    assert price_80 < price_100 < price_120


def test_monotonicity_call_sigma():
    price_01 = asian_option_mc_price(S, K, r, 0.1, T, "C", q, n_paths, n_steps, seed)
    price_02 = asian_option_mc_price(S, K, r, 0.2, T, "C", q, n_paths, n_steps, seed)
    price_04 = asian_option_mc_price(S, K, r, 0.4, T, "C", q, n_paths, n_steps, seed)
    assert price_01 < price_02 < price_04


def test_monotonicity_put_sigma():
    price_01 = asian_option_mc_price(S, K, r, 0.1, T, "P", q, n_paths, n_steps, seed)
    price_02 = asian_option_mc_price(S, K, r, 0.2, T, "P", q, n_paths, n_steps, seed)
    price_04 = asian_option_mc_price(S, K, r, 0.4, T, "P", q, n_paths, n_steps, seed)
    assert price_01 < price_02 < price_04


def test_1_step_call():
    mc_price = asian_option_mc_price(S, K, r, sigma, T, "C", q, 100000, 1, seed)
    bs_price = bs_call_price(S, K, r, sigma, T)
    assert abs(mc_price - bs_price) < 0.1


def test_1_step_put():
    mc_price = asian_option_mc_price(S, K, r, sigma, T, "P", q, 100000, 1, seed)
    bs_price = bs_put_price(S, K, r, sigma, T)
    assert abs(mc_price - bs_price) < 0.1


def test_asian_call_less_than_european_call():
    asian_price = asian_option_mc_price(
        S, K, r, sigma, T, "C", q, 100000, n_steps, seed
    )
    bs_price = bs_call_price(S, K, r, sigma, T)
    assert asian_price < bs_price


def test_negative_spot_raises():
    with pytest.raises(ValueError):
        asian_option_mc_price(-1.0, K, r, sigma, T, "C", q, n_paths, n_steps)


def test_negative_strike_raises():
    with pytest.raises(ValueError):
        asian_option_mc_price(S, -1.0, r, sigma, T, "C", q, n_paths, n_steps)


def test_negative_sigma_raises():
    with pytest.raises(ValueError):
        asian_option_mc_price(S, K, r, -0.1, T, "C", q, n_paths, n_steps)


def test_negative_maturity_raises():
    with pytest.raises(ValueError):
        asian_option_mc_price(S, K, r, sigma, -1.0, "C", q, n_paths, n_steps)


def test_option_type_invalid_raises():
    with pytest.raises(ValueError):
        asian_option_mc_price(S, K, r, sigma, T, "X", q, n_paths, n_steps)


def test_negative_n_paths_raises():
    with pytest.raises(ValueError):
        asian_option_mc_price(S, K, r, sigma, T, "C", q, -1.0, n_steps)


def test_negative_n_steps_raises():
    with pytest.raises(ValueError):
        asian_option_mc_price(S, K, r, sigma, T, "C", q, n_paths, 0.0)
