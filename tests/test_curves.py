import pytest
from pricing_engine.curves import (
    discount_factor,
    forward_price,
    dividend_discount_factor,
)


def test_discount_factor_zero_maturity():
    assert discount_factor(0.05, 0.0) == 1.0


def test_discount_factor_zero_interest_rate():
    assert discount_factor(0.0, 1.0) == 1.0


def test_discount_factor_positive_interest_rate():
    assert discount_factor(0.05, 1.0) < 1.0


def test_discount_factor_negative_interest_rate():
    assert discount_factor(-0.05, 1.0) > 1.0


def test_forward_price_zero_maturity():
    assert forward_price(100, 0.05, 0.0, 0.0) == 100


def test_forward_price_interest_rate_equal_dividends():
    assert forward_price(100, 0.05, 0.05, 1.0) == 100


def test_forward_price_interest_rate_bigger_dividends():
    assert forward_price(100, 0.05, 0.02, 1.0) > 100


def test_forward_price_interest_rate_lower_dividends():
    assert forward_price(100, 0.02, 0.05, 1.0) < 100


def test_dividend_discount_factor_zero_maturity():
    assert dividend_discount_factor(0.05, 0.0) == 1.0


def test_dividend_discount_factor_zero_yield():
    assert dividend_discount_factor(0.0, 1.0) == 1.0


def test_dividend_discount_factor_positive_yield():
    assert dividend_discount_factor(0.05, 1.0) < 1.0


def test_dividend_discount_factor_negative_maturity_raises():
    with pytest.raises(ValueError):
        dividend_discount_factor(0.05, -1.0)


def test_discount_factor_negative_maturity_raises():
    with pytest.raises(ValueError):
        discount_factor(0.05, -1.0)


def test_forward_price_negative_maturity_raises():
    with pytest.raises(ValueError):
        forward_price(100, 0.05, 0.02, -1.0)


def test_forward_negative_spot():
    with pytest.raises(ValueError):
        forward_price(-1.0, 0.05, 0.02, 1.0)
