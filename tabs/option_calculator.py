# tabs/option_calculator.py

import streamlit as st
import pandas as pd
from Option_pricer import OptionPricer

def render(shared_inputs):
    # Sidebar Inputs
    spot_price = shared_inputs["spot_price"]
    strike_price = shared_inputs["strike_price"]
    TTM = shared_inputs["TTM"]
    interest_rate = shared_inputs["interest_rate"]
    volatility = shared_inputs["volatility"]
    show_greeks = st.sidebar.checkbox("📐 Show Greeks", value=True)
    pricer = OptionPricer(spot_price, strike_price, interest_rate, volatility, TTM)
    pricer.run()

    st.markdown("### 💰 Option Prices")
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("### 🟩 Call Option Price")
        st.markdown(f"""
        <div style='background-color:#d4f4dd; padding:20px; border-radius:10px; font-size:24px; text-align:center; color:#111'>
        <b>${pricer.call_price:.4f}</b>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("### 🟥 Put Option Price")
        st.markdown(f"""
        <div style='background-color:#f8d7da; padding:20px; border-radius:10px; font-size:24px; text-align:center; color:#111'>
        <b>${pricer.put_price:.4f}</b>
        </div>
        """, unsafe_allow_html=True)

    # Option Greeks
    if show_greeks:
        st.markdown("### 📊 Option Greeks")
        greeks = pricer.greeks()

        col_call, col_put = st.columns(2)

        call_df = pd.DataFrame({
            "Greek": ["Δ Delta", "Γ Gamma", "Θ Theta/day", "ν Vega (per 1%)", "ρ Rho (per 1%)"],
            "Value": [
                round(greeks["Delta (Call)"], 4),
                round(greeks["Gamma"], 4),
                round(greeks["Theta (Call)"], 4),
                round(greeks["Vega"], 4),
                round(greeks["Rho (Call)"], 4),
            ]})

        put_df = pd.DataFrame({
            "Greek": ["Δ Delta", "Γ Gamma", "Θ Theta/day", "ν Vega (per 1%)", "ρ Rho (per 1%)"],
            "Value": [
                round(greeks["Delta (Put)"], 4),
                round(greeks["Gamma"], 4),
                round(greeks["Theta (Put)"], 4),
                round(greeks["Vega"], 4),
                round(greeks["Rho (Put)"], 4),
            ]})

        with st.sidebar.expander("📘 Greek Descriptions"):
            st.markdown("""
            <ul style='padding-left:1em;'>
                <li><b>Δ Delta</b>: Measures how much the option price changes for a $1 change in the underlying asset.</li>
                <li><b>Γ Gamma</b>: Measures the rate of change in Delta per $1 change in the underlying.</li>
                <li><b>Θ Theta/day</b>: Measures the daily time decay of the option price.</li>
                <li><b>ν Vega (per 1%)</b>: Measures sensitivity of option price to 1% change in volatility.</li>
                <li><b>ρ Rho (per 1%)</b>: Measures sensitivity of option price to 1% change in interest rate.</li>
            </ul>
            """, unsafe_allow_html=True)

        with col_call:
            st.markdown("#### 🟩 Call Option Greeks")
            st.markdown("<div style='background-color:#d8f3dc;padding:1em;border-radius:10px;'>", unsafe_allow_html=True)
            st.table(call_df.set_index("Greek"))
            st.markdown("</div>", unsafe_allow_html=True)

        with col_put:
            st.markdown("#### 🟥 Put Option Greeks")
            st.markdown("<div style='background-color:#ffe5ec;padding:1em;border-radius:10px;'>", unsafe_allow_html=True)
            st.table(put_df.set_index("Greek"))
            st.markdown("</div>", unsafe_allow_html=True)
