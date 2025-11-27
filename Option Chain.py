import streamlit as st
import numpy as np
import pandas as pd
import math

# ------------------------
#   BLACK SCHOLES (NO SCIPY)
# ------------------------
def N(x):
    """Cumulative normal distribution using approximation (no scipy)."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def black_scholes_call(S, K, t, r, sigma):
    """Manual Black–Scholes (call)."""
    if t == 0 or sigma == 0:
        return max(0, S - K)

    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)

    return S * N(d1) - K * math.exp(-r * t) * N(d2)

# ------------------------
#   ALGO LOGIC
# ------------------------
def algo_quotes(fair_price):
    """Base algo quoting logic."""
    buyer_price = 20
    seller_price = 18
    return buyer_price, seller_price

def algo_reacts_to_human(fair_price, human_bid):
    """
    If human trades above algo → algo increases bid.
    If 20% above fair value → algo sells to human.
    """
    threshold = fair_price * 1.20

    if human_bid >= threshold:
        # Algo sells at 20% above fair value
        algo_sell_price = threshold
        return "SELL_TO_HUMAN", algo_sell_price

    # Otherwise algo moves with human
    algo_bid = human_bid + 1
    return "FOLLOW", algo_bid

def generate_option_chain(S, t, r, sigma):
    """Creates a simple strikes table."""
    strikes = [S - 200, S - 100, S, S + 100, S + 200]

    data = []
    for K in strikes:
        fair = black_scholes_call(S, K, t, r, sigma)
        data.append([K, round(fair, 2)])

    return pd.DataFrame(data, columns=["Strike", "Fair Price"])

# ------------------------
#   STREAMLIT UI
# ------------------------
st.title("⚡ Option Chain Simulator with Algo Market Maker")
st.write("A simple educational simulator for option pricing + algorithmic quoting logic.")

st.sidebar.header("Underlying Inputs")

S = st.sidebar.number_input("Spot Price (S)", value=20000)
t = st.sidebar.number_input("Time to Expiry (in years)", value=0.1, step=0.01)
r = st.sidebar.number_input("Risk-free Rate (r)", value=0.05)
sigma = st.sidebar.number_input("Implied Volatility (σ)", value=0.20)

st.subheader("📘 Black–Scholes Fair Prices")
df_chain = generate_option_chain(S, t, r, sigma)
st.dataframe(df_chain)

st.divider()

# ------------------------
#   USER SELECTS STRIKE TO SIMULATE NON-LIQUID MARKET
# ------------------------
st.header("🧪 Non-Liquid Strike Simulation")

selected_strike = st.selectbox("Pick a Strike", df_chain["Strike"].tolist())
fair_price = df_chain[df_chain["Strike"] == selected_strike]["Fair Price"].iloc[0]

st.write(f"**Fair Price (Black–Scholes): ₹{fair_price}**")

base_bid, base_ask = algo_quotes(fair_price)

col1, col2 = st.columns(2)
col1.metric("Algo Buyer Quote", f"₹{base_bid}")
col2.metric("Algo Seller Quote", f"₹{base_ask}")

st.subheader("Enter Human Trader Bid")
human_bid = st.number_input("Human Bid Price", value=base_bid, step=1)

if st.button("Run Simulation"):
    mode, value = algo_reacts_to_human(fair_price, human_bid)

    if mode == "SELL_TO_HUMAN":
        st.success(f"💥 Human bid exceeded 20% above fair value.")
        st.success(f"➡ Algo sells to human at **₹{value:.2f}**")
        st.info("Algo resets back to 20 buy / 100 sell afterwards.")

    else:
        st.warning("Algo follows the human buyer.")
        st.write(f"➡ Algo new bid: **₹{value}** (₹1 above human)")

st.divider()

st.header("📊 Explanation of Algo Behavior")

st.write("""
### Rules Implemented  
1. **Algo stands at ₹20 (buyer) and ₹18 (seller)** for an illiquid strike.  
2. If a human places a higher bid → **algo follows human +1**.  
3. If human reaches **20% above fair value** →  
   - Algo immediately **sells** at *fair_value × 1.20*.  
   - Then algo resets to **₹20 bid / ₹100 ask**.  
4. Human can later sell back to the algo at a loss.

This simulates basic *market making behavior* for illiquid options.
""")
