# app.py
"""
Option Chain Unified Dashboard
Features:
- Market-maker & dynamic bid/ask quotes
- Synthetic liquidity across strikes
- Greeks & Black-Scholes fair value
- Mispricing detector (20% above fair -> flagged)
- Order book + FIFO matching engine
- Mini-brokerage (positions, MTM P/L)
- Volatility shock simulator
- Simple delta-hedging simulation
- Heatmap of activity + arbitrage finder

Run:
    pip install streamlit numpy pandas scipy plotly
    streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
from math import log, sqrt, exp
from scipy.stats import norm
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(layout="wide", page_title="Option Chain + Algo Dashboard")

# ----------------------------
# Utility: Black-Scholes (European)
# ----------------------------
def bs_price_greeks(S, K, r, q, sigma, t, option_type="call"):
    """
    Returns price, delta, vega, theta, rho (approx) using Black-Scholes.
    S: underlying price
    K: strike
    r: risk-free rate
    q: dividend yield
    sigma: volatility (annual)
    t: time to expiry in years
    """
    if t <= 0:
        if option_type == "call":
            price = max(0.0, S - K)
            delta = 1.0 if S > K else 0.0
        else:
            price = max(0.0, K - S)
            delta = -1.0 if S < K else 0.0
        return {"price": price, "delta": delta, "vega": 0.0, "theta": 0.0, "rho": 0.0}

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    nd1 = norm.cdf(d1)
    nd2 = norm.cdf(d2)
    npd1 = norm.pdf(d1)
    if option_type == "call":
        price = S * np.exp(-q * t) * nd1 - K * np.exp(-r * t) * nd2
        delta = np.exp(-q * t) * nd1
    else:
        price = K * np.exp(-r * t) * norm.cdf(-d2) - S * np.exp(-q * t) * norm.cdf(-d1)
        delta = -np.exp(-q * t) * norm.cdf(-d1)
    vega = S * np.exp(-q * t) * npd1 * np.sqrt(t)
    theta = - (S * np.exp(-q * t) * npd1 * sigma) / (2 * np.sqrt(t)) - r * K * np.exp(-r * t) * nd2 if option_type == "call" else - (S * np.exp(-q * t) * npd1 * sigma) / (2 * np.sqrt(t)) + r * K * np.exp(-r * t) * norm.cdf(-d2)
    rho = K * t * np.exp(-r * t) * nd2 if option_type == "call" else -K * t * np.exp(-r * t) * norm.cdf(-d2)
    return {"price": float(price), "delta": float(delta), "vega": float(vega/100), "theta": float(theta/365), "rho": float(rho/100)}

# ----------------------------
# Session state initialization
# ----------------------------
def init_state():
    if "underlying" not in st.session_state:
        st.session_state.underlying = 1000.0  # underlying price (e.g., NIFTY)
    if "iv" not in st.session_state:
        st.session_state.iv = 0.30  # 30% annual vol
    if "r" not in st.session_state:
        st.session_state.r = 0.06  # risk-free rate
    if "q" not in st.session_state:
        st.session_state.q = 0.0  # dividend yield
    if "expiry_days" not in st.session_state:
        st.session_state.expiry_days = 30
    if "strikes" not in st.session_state:
        base = round(st.session_state.underlying / 50) * 50
        strikes = [base + i * 50 for i in range(-6, 7)]
        st.session_state.strikes = strikes
    if "order_book" not in st.session_state:
        st.session_state.order_book = {"bids": [], "asks": []}  # each: dict with id, side, price, qty, time, user
    if "transactions" not in st.session_state:
        st.session_state.transactions = []
    if "positions" not in st.session_state:
        st.session_state.positions = []  # list of dicts: instrument, type (call/put), strike, qty, avg_price, side
    if "algo_quotes" not in st.session_state:
        st.session_state.algo_quotes = {}  # key: (type,strike) -> bid/ask
    if "activity" not in st.session_state:
        st.session_state.activity = []  # logs for heatmap
    if "next_order_id" not in st.session_state:
        st.session_state.next_order_id = 1

init_state()

# ----------------------------
# Helpers: order id/time
# ----------------------------
def now_ts():
    return datetime.utcnow().isoformat()

def new_order_id():
    oid = st.session_state.next_order_id
    st.session_state.next_order_id += 1
    return oid

# ----------------------------
# Build option chain DataFrame (calls only for demo; extendable)
# ----------------------------
def build_chain():
    S = st.session_state.underlying
    strikes = st.session_state.strikes
    expiry_days = st.session_state.expiry_days
    t = max(expiry_days, 0) / 365.0
    rows = []
    for K in strikes:
        c = bs_price_greeks(S, K, st.session_state.r, st.session_state.q, st.session_state.iv, t, "call")
        p = bs_price_greeks(S, K, st.session_state.r, st.session_state.q, st.session_state.iv, t, "put")
        # default algo quotes or initialize if not present
        keyc = ("call", K)
        if keyc not in st.session_state.algo_quotes:
            # algo posts wide spreads for non-liquid strikes, tighter near ATM
            dist = abs(K - S)
            spread = max(1, round(0.02 * c["price"] + dist/200, 2))
            bid = max(0.0, c["price"] - spread)
            ask = c["price"] + spread
            st.session_state.algo_quotes[keyc] = {"bid": round(bid,2), "ask": round(ask,2)}
        keyp = ("put", K)
        if keyp not in st.session_state.algo_quotes:
            dist = abs(K - S)
            spread = max(1, round(0.02 * p["price"] + dist/200, 2))
            bid = max(0.0, p["price"] - spread)
            ask = p["price"] + spread
            st.session_state.algo_quotes[keyp] = {"bid": round(bid,2), "ask": round(ask,2)}

        rows.append({
            "strike": K,
            "call_fv": round(c["price"],2),
            "call_delta": round(c["delta"],3),
            "call_vega": round(c["vega"],3),
            "call_bid": st.session_state.algo_quotes[("call",K)]["bid"],
            "call_ask": st.session_state.algo_quotes[("call",K)]["ask"],
            "put_fv": round(p["price"],2),
            "put_delta": round(p["delta"],3),
            "put_vega": round(p["vega"],3),
            "put_bid": st.session_state.algo_quotes[("put",K)]["bid"],
            "put_ask": st.session_state.algo_quotes[("put",K)]["ask"],
        })
    df = pd.DataFrame(rows).sort_values("strike")
    return df

# ----------------------------
# Market-maker logic (adjust quotes when humans act)
# ----------------------------
def mm_outbid(side, strike, human_price, option_type):
    """
    Algo outbids a human: raises its bid to human_price + 1 (simulating aggressive MM)
    side: 'buy' (human buying) or 'sell' (human selling)
    option_type: 'call' or 'put'
    """
    key = (option_type, strike)
    q = st.session_state.algo_quotes.get(key, {"bid":0.0,"ask":0.0})
    # If human places buy order > algo bid, algo moves bid above
    if side == "buy":
        if human_price >= q["bid"]:
            q["bid"] = round(human_price + 1.0, 2)
    # If human sells below algo ask, algo lowers ask
    else:
        if human_price <= q["ask"]:
            q["ask"] = round(max(0.0, human_price - 1.0), 2)
    st.session_state.algo_quotes[key] = q
    st.session_state.activity.append({"ts": now_ts(), "strike": strike, "type": option_type, "action": "mm_adjust", "price": human_price})

# ----------------------------
# Mispricing detector and threshold logic
# ----------------------------
def check_mispricing(human_price, fair_price, threshold_factor=1.2):
    """
    returns (is_mispriced, threshold_price)
    If human_price >= threshold_factor * fair_price -> mispriced (algo will sell to human)
    """
    threshold_price = round(fair_price * threshold_factor, 2)
    return (human_price >= threshold_price, threshold_price)

# ----------------------------
# Order book and matching (simple)
# ----------------------------
def place_order(user, side, option_type, strike, price, qty=1, order_type="limit"):
    """
    side: 'buy' or 'sell' from the human perspective
    """
    order = {
        "id": new_order_id(), "user": user, "side": side, "type": order_type,
        "option_type": option_type, "strike": strike, "price": float(price),
        "qty": int(qty), "time": now_ts()
    }
    # Add to book (simple queue)
    if side == "buy":
        st.session_state.order_book["bids"].append(order)
    else:
        st.session_state.order_book["asks"].append(order)
    st.session_state.activity.append({"ts": now_ts(), "strike": strike, "type": option_type, "action": "order_placed", "side": side, "price": price})
    match_orders(option_type, strike)
    return order

def match_orders(option_type, strike):
    """
    Very simple FIFO matching between top bid and ask for a given strike & option_type.
    Also allows algorithm to be a counterparty using algo_quotes.
    """
    bids = [o for o in st.session_state.order_book["bids"] if o["strike"]==strike and o["option_type"]==option_type]
    asks = [o for o in st.session_state.order_book["asks"] if o["strike"]==strike and o["option_type"]==option_type]
    # sort by time (FIFO)
    bids = sorted(bids, key=lambda x: x["time"])
    asks = sorted(asks, key=lambda x: x["time"])
    key = (option_type, strike)
    algo_q = st.session_state.algo_quotes.get(key, {"bid":0.0,"ask":1e9})
    # try matching human vs human first
    while bids and asks:
        top_b = bids[0]
        top_a = asks[0]
        # match if bid >= ask
        if top_b["price"] >= top_a["price"]:
            qty = min(top_b["qty"], top_a["qty"])
            trade_price = round((top_b["price"] + top_a["price"]) / 2.0, 2)
            do_trade(top_b, top_a, trade_price, qty)
            top_b["qty"] -= qty
            top_a["qty"] -= qty
            if top_b["qty"] == 0:
                st.session_state.order_book["bids"].remove(top_b)
            if top_a["qty"] == 0:
                st.session_state.order_book["asks"].remove(top_a)
            bids = [o for o in st.session_state.order_book["bids"] if o["strike"]==strike and o["option_type"]==option_type]
            asks = [o for o in st.session_state.order_book["asks"] if o["strike"]==strike and o["option_type"]==option_type]
            bids = sorted(bids, key=lambda x: x["time"])
            asks = sorted(asks, key=lambda x: x["time"])
        else:
            break

    # If no HH match, allow algo to take other side if prices cross algo quotes
    # Human buy and algo sells at algo.ask if human price >= algo.ask
    for b in bids[:]:
        if b["price"] >= algo_q["ask"]:
            # sell from algo at algo.ask
            trade_price = algo_q["ask"]
            algo_order = {"id": 0, "user": "algo", "side": "sell", "option_type": option_type, "strike": strike, "price": trade_price, "qty": b["qty"], "time": now_ts()}
            do_trade(b, algo_order, trade_price, b["qty"])
            st.session_state.order_book["bids"].remove(b)
            # after selling to human at high price, reset quotes aggressively (create hostile ask)
            st.session_state.algo_quotes[key] = {"bid": round(max(0.0, algo_q["bid"]-2),2), "ask": round(max(algo_q["ask"]*2, algo_q["ask"]+20),2)}
            st.session_state.activity.append({"ts": now_ts(), "strike": strike, "type": option_type, "action": "algo_sell_to_human", "price": trade_price})
    # Human sell and algo buys at algo.bid if human price <= algo.bid
    for a in asks[:]:
        if a["price"] <= algo_q["bid"]:
            trade_price = algo_q["bid"]
            algo_order = {"id": 0, "user": "algo", "side": "buy", "option_type": option_type, "strike": strike, "price": trade_price, "qty": a["qty"], "time": now_ts()}
            do_trade(algo_order, a, trade_price, a["qty"])
            st.session_state.order_book["asks"].remove(a)
            # after buying from human at low price, algo tightens spread slightly
            st.session_state.algo_quotes[key] = {"bid": round(algo_q["bid"]+2,2), "ask": round(max(algo_q["ask"], algo_q["bid"]+5),2)}
            st.session_state.activity.append({"ts": now_ts(), "strike": strike, "type": option_type, "action": "algo_buy_from_human", "price": trade_price})

def do_trade(buy_order, sell_order, trade_price, qty):
    """
    Execute a trade and update positions + transactions.
    buy_order and sell_order are dicts (user fields). qty is int.
    """
    t = {
        "time": now_ts(), "strike": buy_order["strike"], "option_type": buy_order["option_type"],
        "buyer": buy_order["user"], "seller": sell_order["user"], "price": float(trade_price), "qty": int(qty)
    }
    st.session_state.transactions.append(t)
    # update positions for buyer (+qty) and seller (-qty)
    # for simplicity treat option as one sided instrument (call only) and store positions per user
    for role, user, sign in [("buyer", buy_order["user"], 1), ("seller", sell_order["user"], -1)]:
        if user == "algo":
            continue  # do not track algo positions for now
        # find position record
        found = None
        for pos in st.session_state.positions:
            if pos["user"] == user and pos["option_type"] == buy_order["option_type"] and pos["strike"] == buy_order["strike"]:
                found = pos
                break
        if found:
            # update avg price
            prev_qty = found["qty"]
            prev_avg = found["avg_price"]
            new_qty = prev_qty + sign * qty
            if new_qty == 0:
                # position closed -> realized P/L not separately tracked here
                st.session_state.positions.remove(found)
            else:
                # update weighted avg only if same side
                if sign * prev_qty > 0:
                    # same side - increase or decrease; compute avg only for increases
                    if sign > 0:
                        found["avg_price"] = (prev_avg*prev_qty + trade_price*qty) / (prev_qty + qty)
                    # if reducing size, keep avg
                    found["qty"] = new_qty
                else:
                    # opposite side -> reduce
                    found["qty"] = new_qty
        else:
            # new position
            if sign > 0:
                st.session_state.positions.append({"user": user, "option_type": buy_order["option_type"], "strike": buy_order["strike"], "qty": qty, "avg_price": float(trade_price)})
            else:
                # negative quantity (short) allowed
                st.session_state.positions.append({"user": user, "option_type": buy_order["option_type"], "strike": buy_order["strike"], "qty": -qty, "avg_price": float(trade_price)})

    st.session_state.activity.append({"ts": now_ts(), "strike": buy_order["strike"], "type": buy_order["option_type"], "action": "trade", "price": trade_price, "qty": qty, "buyer": buy_order["user"], "seller": sell_order["user"]})

# ----------------------------
# Delta-hedging (simple)
# ----------------------------
def delta_hedge_for_algo(strike, option_type):
    """
    If algo sold call to human, it hedges by buying underlying proportional to delta.
    This is a simple simulation where we compute delta * qty and log a hedge trade.
    """
    # compute last transaction at strike where algo was seller
    txs = [t for t in st.session_state.transactions if t["strike"]==strike and t["option_type"]==option_type]
    if not txs:
        return None
    last = txs[-1]
    if last["seller"] != "algo":
        return None
    # assume delta from current BS
    t = max(st.session_state.expiry_days,0)/365.0
    K = strike
    c = bs_price_greeks(st.session_state.underlying, K, st.session_state.r, st.session_state.q, st.session_state.iv, t, option_type)
    delta = c["delta"]
    hedge_qty = round(delta * last["qty"], 2)
    # log hedge as activity (algo bought underlying)
    st.session_state.activity.append({"ts": now_ts(), "strike": strike, "action": "delta_hedge", "hedge_qty": hedge_qty, "delta": delta})
    return {"delta": delta, "hedge_qty": hedge_qty}

# ----------------------------
# UI: controls left column
# ----------------------------
st.sidebar.header("Market Controls")
st.sidebar.slider("Underlying price", 500, 1500, int(st.session_state.underlying), step=5, key="underlying")
st.sidebar.slider("IV (annual)", 0.05, 1.0, float(st.session_state.iv), step=0.01, key="iv")
st.sidebar.number_input("Expiry (days)", min_value=1, max_value=365, value=int(st.session_state.expiry_days), key="expiry_days")
st.sidebar.number_input("Risk-free rate (r)", min_value=0.0, max_value=0.2, value=float(st.session_state.r), step=0.001, key="r")
st.sidebar.number_input("Dividend yield (q)", min_value=0.0, max_value=0.2, value=float(st.session_state.q), step=0.001, key="q")
if st.sidebar.button("Rebuild strikes"):
    base = round(st.session_state.underlying / 50) * 50
    st.session_state.strikes = [base + i * 50 for i in range(-6, 7)]

# Refresh chain whenever market controls change
chain_df = build_chain()

# ----------------------------
# Main layout
# ----------------------------
st.title("Option Chain + Algo Simulator Dashboard")
left, mid, right = st.columns([1.2, 1, 1.2])

# -- Left: Option chain, MM & synthetic liquidity controls
with left:
    st.subheader("Option Chain (Calls)")
    st.dataframe(chain_df.style.format("{:.2f}"), height=400)
    st.markdown("**Algo Quotes** (editable):")
    # show editable quotes
    quotes_df = []
    for k, v in st.session_state.algo_quotes.items():
        quotes_df.append({"option_type": k[0], "strike": k[1], "bid": v["bid"], "ask": v["ask"]})
    qdf = pd.DataFrame(quotes_df).sort_values(["strike","option_type"])
    edited = st.experimental_data_editor(qdf, num_rows="never", use_container_width=True)
    # save edits back
    if st.button("Apply edited algo quotes"):
        for _, r in edited.iterrows():
            st.session_state.algo_quotes[(r["option_type"], int(r["strike"]))] = {"bid": float(r["bid"]), "ask": float(r["ask"])}
        st.success("Algo quotes updated.")

    st.markdown("---")
    st.subheader("Synthetic Liquidity / Market-Maker params")
    spread_tightness = st.slider("Spread tightness (lower → tighter)", 0.5, 5.0, 1.0, step=0.1)
    aggressiveness = st.slider("MM aggressiveness (how much > human)", 0.5, 5.0, 1.0, step=0.1)
    # button to run a synthetic flow that widens or narrows spreads
    if st.button("Simulate liquidity change"):
        for k in st.session_state.algo_quotes.keys():
            bid = st.session_state.algo_quotes[k]["bid"]
            ask = st.session_state.algo_quotes[k]["ask"]
            # nearer ATM tighter spreads
            dist = abs(k[1] - st.session_state.underlying)
            factor = 1 + (dist / 500) * spread_tightness
            st.session_state.algo_quotes[k]["bid"] = round(max(0.01, bid / factor),2)
            st.session_state.algo_quotes[k]["ask"] = round(ask * factor,2)
        st.success("Synthetic liquidity simulation applied.")

# -- Middle: Order entry, matching, mispricing detector
with mid:
    st.subheader("Order Entry & Matching Engine")
    with st.form("order_form"):
        user_name = st.text_input("User name", value="human1")
        opt_type = st.selectbox("Option", ["call","put"])
        strike_sel = st.selectbox("Strike", st.session_state.strikes)
        side = st.selectbox("Side", ["buy","sell"])
        price = st.number_input("Price (₹)", min_value=0.0, value=10.0, step=0.5)
        qty = st.number_input("Qty", min_value=1, value=1, step=1)
        submitted = st.form_submit_button("Place Order")
    if submitted:
        # check mispricing against fair price
        t = max(st.session_state.expiry_days,0)/365.0
        fv = bs_price_greeks(st.session_state.underlying, strike_sel, st.session_state.r, st.session_state.q, st.session_state.iv, t, opt_type)["price"]
        mis, thresh = check_mispricing(price, fv, threshold_factor=1.2)
        if mis:
            st.warning(f"Order is ≥20% above fair value (threshold {thresh}). Algorithm will try to sell to buyer if aggressive.")
        # mm reacts: outbid/adjust quotes
        mm_outbid(side, strike_sel, price, opt_type)
        o = place_order(user_name, side, opt_type, strike_sel, price, qty)
        st.success(f"Placed order ID {o['id']} ({side} {opt_type} {strike_sel} @ {price} x{qty})")

    st.markdown("**Order Book (Top 10)**")
    bids_df = pd.DataFrame(st.session_state.order_book["bids"])[["id","user","side","option_type","strike","price","qty","time"]] if st.session_state.order_book["bids"] else pd.DataFrame(columns=["id","user"])
    asks_df = pd.DataFrame(st.session_state.order_book["asks"])[["id","user","side","option_type","strike","price","qty","time"]] if st.session_state.order_book["asks"] else pd.DataFrame(columns=["id","user"])
    st.write("Bids")
    st.dataframe(bids_df.sort_values("price", ascending=False).head(10))
    st.write("Asks")
    st.dataframe(asks_df.sort_values("price", ascending=True).head(10))

# -- Right: Transactions, positions, hedging
with right:
    st.subheader("Transactions & Positions")
    tx_df = pd.DataFrame(st.session_state.transactions)
    if not tx_df.empty:
        st.dataframe(tx_df.tail(10))
    else:
        st.write("No transactions yet.")
    st.markdown("**Positions (per user instrument)**")
    if st.session_state.positions:
        st.dataframe(pd.DataFrame(st.session_state.positions))
    else:
        st.write("No positions yet.")

    if st.button("Run simple delta-hedge for last trade"):
        if st.session_state.transactions:
            last = st.session_state.transactions[-1]
            hed = delta_hedge_for_algo(last["strike"], last["option_type"])
            if hed:
                st.success(f"Algo hedged {hed['hedge_qty']} underlying units (delta {hed['delta']:.3f}) for strike {last['strike']}.")
            else:
                st.info("No recent trade where algo was seller to hedge against.")

# ----------------------------
# Bottom panels: Greeks visual, heatmap, arbitrage finder
# ----------------------------
st.markdown("---")
col1, col2 = st.columns([1.3,1])

with col1:
    st.subheader("Greeks & Volatility Shock")
    strike_choice = st.selectbox("Choose strike for Greeks", st.session_state.strikes, key="greeks_strike")
    t = max(st.session_state.expiry_days,0)/365.0
    greeks = bs_price_greeks(st.session_state.underlying, strike_choice, st.session_state.r, st.session_state.q, st.session_state.iv, t, "call")
    st.metric("Call Fair Value", f"₹{greeks['price']:.2f}")
    st.write(pd.DataFrame([greeks]).T.rename(columns={0:"value"}))
    st.markdown("**IV Shock Simulator**")
    shock = st.slider("IV shock (%) ±", -20, 100, 0, step=5)
    shocked_iv = max(0.01, st.session_state.iv * (1 + shock/100.0))
    shocked = bs_price_greeks(st.session_state.underlying, strike_choice, st.session_state.r, st.session_state.q, shocked_iv, t, "call")
    st.write(f"Shocked IV: {shocked_iv:.2%} → FV: ₹{shocked['price']:.2f}")
    if st.button("Apply shocked IV globally"):
        st.session_state.iv = shocked_iv
        st.experimental_rerun()

with col2:
    st.subheader("Activity Heatmap & Arbitrage")
    # activity heatmap by strike vs action counts
    act_df = pd.DataFrame(st.session_state.activity)
    if not act_df.empty:
        heat = act_df.groupby(["strike","action"]).size().unstack(fill_value=0)
        heat = heat.reindex(index=sorted(heat.index))
        fig = go.Figure(data=go.Heatmap(z=heat.values, x=list(heat.columns), y=list(heat.index), hoverongaps=False))
        fig.update_layout(height=300, yaxis_title="Strike", xaxis_title="Action")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No activity yet to show heatmap.")

    st.markdown("**Arbitrage Finder**")
    # simple check: find human orders that are significantly away from fair value
    suspicious = []
    for ob in st.session_state.order_book["bids"] + st.session_state.order_book["asks"]:
        t = max(st.session_state.expiry_days,0)/365.0
        fv = bs_price_greeks(st.session_state.underlying, ob["strike"], st.session_state.r, st.session_state.q, st.session_state.iv, t, ob["option_type"])["price"]
        if ob["side"] == "buy" and ob["price"] > fv * 1.05:
            suspicious.append({**ob, "fv": round(fv,2), "label":"buy_above_fv"})
        if ob["side"] == "sell" and ob["price"] < fv * 0.95:
            suspicious.append({**ob, "fv": round(fv,2), "label":"sell_below_fv"})
    if suspicious:
        st.write(pd.DataFrame(suspicious))
    else:
        st.write("No arbitrage opportunities detected (simple rules).")

# ----------------------------
# Footer: quick actions
# ----------------------------
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns([1,1,1])
with footer_col1:
    if st.button("Reset all orders & positions"):
        st.session_state.order_book = {"bids": [], "asks": []}
        st.session_state.positions = []
        st.session_state.transactions = []
        st.session_state.activity = []
        st.success("Reset done.")
with footer_col2:
    if st.button("Reset algo quotes to fair value"):
        # reset algo quotes based on fair values
        t = max(st.session_state.expiry_days,0)/365.0
        for K in st.session_state.strikes:
            fv = bs_price_greeks(st.session_state.underlying, K, st.session_state.r, st.session_state.q, st.session_state.iv, t, "call")["price"]
            st.session_state.algo_quotes[("call",K)] = {"bid": max(0.01, round(fv*0.8,2)), "ask": round(fv*1.2 + 1,2)}
            pfv = bs_price_greeks(st.session_state.underlying, K, st.session_state.r, st.session_state.q, st.session_state.iv, t, "put")["price"]
            st.session_state.algo_quotes[("put",K)] = {"bid": max(0.01, round(pfv*0.8,2)), "ask": round(pfv*1.2 + 1,2)}
        st.success("Algo quotes reset to fair-based levels.")
with footer_col3:
    st.write("Made for demo — extend as needed. Save to GitHub and deploy on Streamlit Cloud.")

# End of app.py

