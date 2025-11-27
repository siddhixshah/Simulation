import streamlit as st
import numpy as np
import pandas as pd
from math import log, sqrt, exp, pi

st.set_page_config(layout="wide", page_title="Option Chain Simulator")

# ----------------------------
# NORMAL PDF + CDF (no SciPy)
# ----------------------------
def norm_pdf(x):
    return (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x * x)

def norm_cdf(x):
    return (1.0 + np.math.erf(x / np.sqrt(2.0))) / 2.0

# ----------------------------
# Black–Scholes + Greeks
# ----------------------------
def bs_price_greeks(S, K, r, q, sigma, t, option_type="call"):

    if t <= 0:
        # expiry behavior
        if option_type == "call":
            return {
                "price": max(0, S - K),
                "delta": 1.0 if S > K else 0.0,
                "vega": 0,
                "theta": 0
            }
        else:
            return {
                "price": max(0, K - S),
                "delta": -1.0 if S < K else 0.0,
                "vega": 0,
                "theta": 0
            }

    d1 = (log(S/K) + (r - q + 0.5 * sigma*sigma)*t) / (sigma*sqrt(t))
    d2 = d1 - sigma*sqrt(t)

    if option_type == "call":
        price = S*exp(-q*t)*norm_cdf(d1) - K*exp(-r*t)*norm_cdf(d2)
        delta = exp(-q*t)*norm_cdf(d1)
    else:
        price = K*exp(-r*t)*norm_cdf(-d2) - S*exp(-q*t)*norm_cdf(-d1)
        delta = -exp(-q*t)*norm_cdf(-d1)

    vega = S * exp(-q*t) * norm_pdf(d1) * sqrt(t)

    # Approx theta
    theta = - (S * sigma * exp(-q*t) * norm_pdf(d1)) / (2 * sqrt(t))

    return {
        "price": float(price),
        "delta": float(delta),
        "vega": float(vega/100),
        "theta": float(theta/365)
    }

# ----------------------------
# Build Option Chain
# ----------------------------
def build_chain(S, strikes, r, q, sigma, t):

    rows = []

    for K in strikes:
        call = bs_price_greeks(S, K, r, q, sigma, t, "call")
        put  = bs_price_greeks(S, K, r, q, sigma, t, "put")

        rows.append({
            "Strike": K,
            "Call FV": round(call["price"],2),
            "Call Δ": round(call["delta"],3),
            "Put FV": round(put["price"],2),
            "Put Δ": round(put["delta"],3)
        })

    df = pd.DataFrame(rows)
    return df

# ----------------------------
# STREAMLIT UI
# ----------------------------
st.title("📈 Option Chain Simulator (SciPy-Free Version)")

col1, col2, col3 = st.columns(3)

S     = col1.number_input("Underlying Price", 100, 100000, 20000)
iv    = col1.slider("Implied Volatility (σ)", 0.01, 1.0, 0.20)
r     = col2.number_input("Risk-Free Rate (r)", 0.00, 0.20, 0.06)
q     = col2.number_input("Dividend Yield (q)", 0.00, 0.20, 0.00)
days  = col3.number_input("Days to Expiry", 1, 365, 30)

t = days / 365

# Define strikes
center = round(S / 100) * 100
strikes = [center + i*100 for i in range(-10, 11)]

df_chain = build_chain(S, strikes, r, q, iv, t)

st.subheader("📊 Option Chain (Calls & Puts)")
st.dataframe(df_chain, use_container_width=True)

# Plot call/put FV curves
fig = px.line(df_chain, x="Strike", y=["Call FV", "Put FV"], title="Fair Value Curve")
st.plotly_chart(fig, use_container_width=True)



