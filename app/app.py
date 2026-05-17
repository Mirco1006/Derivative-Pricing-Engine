import streamlit as st

from pricing_engine.bs import bs_call_price, bs_put_price
import pricing_engine.greeks as greeks
from pricing_engine.implied_vol import implied_volatility
from pricing_engine.monte_carlo import asian_option_mc_price
from pricing_engine.trees import binomial_prices

st.set_page_config(page_title="Derivatives Pricing Engine", layout="wide")
st.title("Derivatives Pricing Engine")
st.caption("Educational project only. Not investment advice.")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Black-Scholes",
    "Greeks",
    "Implied Volatility",
    "Binomial Tree",
    "Monte Carlo Asian",
])

st.sidebar.header("Parameters")
S = st.sidebar.number_input("Spot price (S)", value=100.0, min_value=0.01)
K = st.sidebar.number_input("Strike price (K)", value=100.0, min_value=0.01)
r = st.sidebar.number_input("Risk-free rate (r)", value=0.05, step=0.01)
sigma = st.sidebar.number_input("Volatility (σ)", value=0.2, min_value=0.01, step=0.01)
T = st.sidebar.number_input("Time to maturity (T, years)", value=1.0, min_value=0.01)
q = st.sidebar.number_input("Dividend yield (q)", value=0.0, min_value=0.0, step=0.01)
option_type = st.sidebar.radio("Option type", ["Call", "Put"])

with tab1:
    st.header("Black-Scholes Pricing")

    if option_type == "Call":
        price = bs_call_price(S, K, r, sigma, T, q)
    else:
        price = bs_put_price(S, K, r, sigma, T, q)

    st.metric(label=f"{option_type} Price", value=f"{price:.4f}")

with tab2:
    st.header("Greeks")

    if option_type == "Call":
        delta = greeks.delta_call(S, K, r, sigma, T, q)
        rho = greeks.calculate_rho_call(S, K, r, sigma, T, q)
        theta = greeks.calculate_theta_call(S, K, r, sigma, T, q)
    else:
        delta = greeks.delta_put(S, K, r, sigma, T, q)
        rho = greeks.calculate_rho_put(S, K, r, sigma, T, q)
        theta = greeks.calculate_theta_put(S, K, r, sigma, T, q)

    gamma = greeks.calculate_gamma(S, K, r, sigma, T, q)
    vega = greeks.calculate_vega(S, K, r, sigma, T, q)

    col1, col2 = st.columns(2)
    with col1:
        st.metric(label=f"{option_type} Delta", value=f"{delta:.4f}")
        st.metric(label=f"{option_type} Gamma", value=f"{gamma:.4f}")
        st.metric(label=f"{option_type} Vega", value=f"{vega:.4f}")
    with col2:
        st.metric(label=f"{option_type} Rho", value=f"{rho:.4f}")
        st.metric(label=f"{option_type} Theta", value=f"{theta:.4f}")

with tab3:
    st.header("Implied Volatility")
    market_price = st.number_input("Option Market Price", value=10.0, min_value=0.01)

    try:
        imp_vol = implied_volatility(market_price, option_type[0], S, K, r, T, q=q)
        st.metric(label=f"{option_type} Implied Volatility", value=f"{imp_vol:.4f}")
    except ValueError as e:
        st.error(f"Cannot compute implied volatility: {e}")

with tab4:
    st.header("Binomial Tree")
    exercise = st.radio("Exercise type ?", ["European", "American"])
    steps = st.number_input("Number of steps", value=100, min_value=1)

    binomial_price = binomial_prices(S, K, r, sigma, T, option_type[0], exercise.lower(), q, steps)

    st.metric(label=f"{option_type} Binomial Price", value=f"{binomial_price:.4f}")


with tab5:
    st.header("Monte Carlo Asian")
    col3, col4 = st.columns(2)
    with col3:
        n_paths = st.number_input("Number of paths", value=10000, min_value=1)
        n_steps = st.number_input("Number of steps", value=252, min_value=1)
    with col4:
        seed_presence = st.radio("Add seed number ?", ["Yes", "No"])
        if seed_presence == "Yes":
            seed = st.number_input("Seed number", value=42, min_value=1)
        else:
            seed = None
        return_detail_info = st.radio("Return detailed information ?", ["Yes", "No"])
        if return_detail_info == "Yes":
            return_detail = True
        else:
            return_detail = False

    asian_option_price = asian_option_mc_price(S, K, r, sigma, T, option_type[0], q, n_paths, n_steps, seed, return_detail)

    if return_detail_info == "No":
        st.metric(label=f"{option_type} Asian Price", value=f"{asian_option_price:.4f}")
    else:
        col5, col6 = st.columns(2)
        with col5:
            st.metric(label=f"{option_type} Asian Price", value=f"{asian_option_price['price']:.4f}")
            st.metric(label=f"{option_type} Asian Standard Error", value=f"{asian_option_price['standard_error']:.4f}")
        with col6:
            st.metric(label=f"{option_type} Asian Lower Bound", value=f"{asian_option_price['ci_lower']:.4f}")
            st.metric(label=f"{option_type} Asian Upper Bound", value=f"{asian_option_price['ci_upper']:.4f}")
