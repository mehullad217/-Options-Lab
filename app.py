import streamlit as st
from Option_pricer import OptionPricer  # import your class
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime, timedelta
import pytz
from services.market import (compute_ttm_years, get_risk_free_rate, get_spot,get_expiries, get_option_chain, nearest_index)
from core.heatmaps import price_grid, pnl_grid
from core.strategies import (straddle_pnl_curve, strangle_pnl_curve, iron_condor_net_credit,iron_condor_payoff_at_expiry,butterfly_pnl_curve)



st.set_page_config(page_title="Options Pricer", page_icon="📈" , initial_sidebar_state= 'expanded' , layout= 'wide')
st.title("📈 Options Pricer - Black-Scholes Model")
st.markdown("This app calculates theoretical **call** and **put** option prices.")


st.sidebar.header("⚙️ Settings")
live_mode = st.sidebar.toggle("📡 Use Live Market Data", value=False)

if live_mode:
    st.sidebar.subheader("Live Data Config")
    ticker_symbol = st.sidebar.text_input("Enter Ticker Symbol", value="AAPL")
    
    try:
    # Spot price
        spot = get_spot(ticker_symbol)
        if spot is None:
            st.sidebar.error("No price history available for this symbol.")
            live_mode = False
        else:
            spot_price = round(float(spot), 2)
            st.sidebar.success(f"📈 Spot Price: ${spot_price}")

        if live_mode:
            expiries = get_expiries(ticker_symbol)
            if not expiries:
                st.sidebar.error("No listed expirations for this symbol.")
                live_mode = False
            else:
                expiry = st.sidebar.selectbox("Select Expiration", expiries)
                MIN_TTM = 1e-6
                TTM = max(compute_ttm_years(expiry), MIN_TTM)
                interest_rate = round(get_risk_free_rate(TTM), 4)
                st.sidebar.success(f"Risk-Free Rate: {interest_rate:.3%}")

        if live_mode:
            chain = get_option_chain(ticker_symbol, expiry)
            calls = chain.calls[["strike","impliedVolatility","bid","ask","lastPrice"]].dropna(subset=["strike"])
            available_strikes = sorted(calls["strike"].astype(float).unique())

            # keep your existing session_state logic
            if "selected_strike" not in st.session_state:
                st.session_state.selected_strike = None
            if "prev_expiry" not in st.session_state:
                st.session_state.prev_expiry = None

            target_strike = st.session_state.selected_strike or spot_price
            if st.session_state.prev_expiry != expiry:
                target_strike = st.session_state.selected_strike or spot_price

            default_idx = nearest_index(available_strikes, target_strike)

            strike_price = st.sidebar.selectbox(
                "Select Strike Price",
                available_strikes,
                index=default_idx,
                key="strike_select"
            )
            st.session_state.selected_strike = float(strike_price)
            st.session_state.prev_expiry = expiry

            if abs(target_strike - strike_price) > 0.0001:
                st.sidebar.caption(
                    f"Note: {target_strike:.2f} not listed for {expiry}; snapped to nearest {strike_price:.2f}."
                )

            row = calls.loc[calls["strike"] == strike_price]
            iv = float(row["impliedVolatility"].iloc[0]) if not row.empty else float("nan")
            MIN_VOL = 1e-4

            if not np.isfinite(iv) or iv <= MIN_VOL:
                volatility = st.sidebar.number_input(
                    "Volatility (Manual Fallback)", value=0.20, min_value=0.01, step=0.01
                )
            else:
                volatility = round(iv, 4)

            if np.isfinite(iv) and iv > MIN_VOL:
                st.sidebar.success(f"Implied Volatility (selected strike): {iv:.2%}")
            else:
                st.sidebar.warning("No IV available for this strike; using manual volatility.")
                st.sidebar.info(f"Volatility used for pricing: {volatility:.2%}")

    except Exception as e:
        st.sidebar.error(f"Failed to fetch data: {e}")
        live_mode = False  # fallback
if not live_mode:
    #Sidebar_inputs
        st.sidebar.header("📥Input Parameters")
        spot_price  = st.sidebar.number_input("Spot_Price(S)", value = 100)
        strike_price = st.sidebar.number_input("Strike_Price(K)", value = 100)
        TTM = st.sidebar.number_input("Time To Maturity (T, in years) ", min_value= 0.01 ,value = 1.0, step = 0.01)
        interest_rate = st.sidebar.number_input("Risk_Free_Rate(r, indecimal)",value =0.05)
        volatility = st.sidebar.number_input("Volatility(σ , in decimal)",value = 0.20)

# Create tabs
tab1, tab2, tab3,tab4, tab5= st.tabs(["📈 Option Calculator", "🌡️ Scenario Heatmap", " 💹 P&L Scenario Heatmap","🧩 Basic Options Strategy","📐 Advanced Options Strategies" ])

# --- TAB 1 Placeholder ---
with tab1:
    # if not live_mode:
    # #Sidebar_inputs
    #     st.sidebar.header("📥Input Parameters")
    #     spot_price  = st.sidebar.number_input("Spot_Price(S)", value = 100)
    #     strike_price = st.sidebar.number_input("Strike_Price(K)", value = 100)
    #     TTM = st.sidebar.number_input("Time To Maturity (T, in years) ", min_value= 0.01 ,value = 1.0, step = 0.01)
    #     interest_rate = st.sidebar.number_input("Risk_Free_Rate(r, indecimal)",value =0.05)
    #     volatility = st.sidebar.number_input("Volatility(σ , in decimal)",value = 0.20)
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
        # greek_tooltips = {
        #     "Delta": "Measures how much the option price changes for a $1 change in the underlying asset.",
        #     "Gamma": "Measures the rate of change in Delta per $1 change in the underlying.",
        #     "Theta/day": "Measures the daily time decay of the option price.",
        #     "Vega (per 1%)": "Measures sensitivity of option price to 1% change in volatility.",
        #     "Rho (per 1%)": "Measures sensitivity of option price to 1% change in interest rate."
        # }
        # Create DataFrames for each type
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
            st.markdown(
                "<div style='background-color:#d8f3dc;padding:1em;border-radius:10px;'>",
                unsafe_allow_html=True
            )
            st.table(call_df.set_index("Greek"))
            st.markdown("</div>", unsafe_allow_html=True)

        with col_put:
            st.markdown("#### 🟥 Put Option Greeks")
            st.markdown(
                "<div style='background-color:#ffe5ec;padding:1em;border-radius:10px;'>",
                unsafe_allow_html=True
            )
            st.table(put_df.set_index("Greek"))
            st.markdown("</div>", unsafe_allow_html=True)


# --- TAB 2 Placeholder ---
with tab2:
    st.markdown("## 🌡️ Scenario Heatmap")

    # Toggle to enable custom axis configuration
    enable_custom_axis = st.sidebar.checkbox("Enable Custom Axis Configuration")
    #heatmap_resolution = st.sidebar.slider("Heatmap Resolution", min_value=5, max_value=30, value=15, step=1)
    st.sidebar.header("📊 Heatmap Axis Configuration")

    param_options = ["Spot Price (S)", "Volatility (σ)", "Time to Maturity (T)", "Risk-Free Rate (r)", "Strike Price (K)"]

    x_axis_param = st.sidebar.selectbox("📈 X-Axis Parameter", param_options, index=0)
    y_axis_param = st.sidebar.selectbox("📉 Y-Axis Parameter", [p for p in param_options if p != x_axis_param], index=0)

    param_defaults = {
            "Spot Price (S)": spot_price,
            "Volatility (σ)": volatility,
            "Time to Maturity (T)": TTM,
            "Risk-Free Rate (r)": interest_rate,
            "Strike Price (K)": strike_price
        }

    param_ranges = {}
    for param in [x_axis_param, y_axis_param]:
            base = param_defaults[param]
            pmin = st.sidebar.number_input(f"Min {param}", value=(base - (0.2*base)), key=f"min_{param}")
            pmax = st.sidebar.number_input(f"Max {param}", value=(base + (0.2*base)), key=f"max_{param}")
            param_ranges[param] = np.linspace(pmin, pmax, 10)

    fixed_params = param_defaults.copy()
    fixed_params.pop(x_axis_param)
    fixed_params.pop(y_axis_param)
    if enable_custom_axis:



        x_vals = param_ranges[x_axis_param]
        y_vals = param_ranges[y_axis_param]
        # call_data, put_data = dynamic_heatmap(x_axis_param, y_axis_param, x_vals, y_vals, fixed_params)
        call_data, put_data = price_grid(x_axis_param, y_axis_param, x_vals, y_vals, fixed_params)

        col5, col6 = st.columns(2, gap='small')

        with col5:
            st.subheader('Call Price Heatmap')
            fig, ax = plt.subplots(figsize=(10, 8), dpi=100, constrained_layout=True)
            sns.heatmap(
                call_data,
                xticklabels=np.round(x_vals, 2),
                yticklabels=np.round(y_vals, 2),
                cmap="RdYlGn",
                annot=True,
                fmt=".2f",
                ax=ax
            )
            ax.set_xlabel(x_axis_param)
            ax.set_ylabel(y_axis_param)
            ax.set_title('CALL')
            st.pyplot(fig)

        with col6:
            st.subheader('Put Price Heatmap')
            fig, ax = plt.subplots(figsize=(10, 8), dpi=100, constrained_layout=True)
            sns.heatmap(
                put_data,
                xticklabels=np.round(x_vals, 2),
                yticklabels=np.round(y_vals, 2),
                cmap="RdYlGn",
                annot=True,
                fmt=".2f",
                ax=ax
            )
            ax.set_xlabel(x_axis_param)
            ax.set_ylabel(y_axis_param)
            ax.set_title('PUT')
            st.pyplot(fig)
        

    else:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<div style='max-width: 300px;'>", unsafe_allow_html=True)
            min_S = st.number_input("🟢 Minimum Spot Price", value = spot_price - (0.2*spot_price))
            max_S = st.number_input("🔴 Maximum Spot Price", value = spot_price +(0.2*spot_price))
            st.markdown("</div>", unsafe_allow_html=True)
        with col2:
            st.markdown("<div style='max-width: 300px;'>", unsafe_allow_html=True)
            min_sigma = st.slider("📉 Minimum Volatility (σ)",  min_value=0.01, max_value=1.0, value= (volatility-0.5*volatility), step=0.01)
            max_sigma = st.slider("📈 Maximum Volatility (σ)", min_value=0.01, max_value=1.0, value=(volatility+0.5*volatility), step=0.01)
            st.markdown("</div>", unsafe_allow_html=True)

        spot_range = np.linspace(min_S ,max_S ,10)
        vol_range = np.linspace(min_sigma,max_sigma,10)
        constants = {
            "Spot Price (S)": pricer.S,
            "Strike Price (K)": pricer.K,
            "Time to Maturity (T)": pricer.t,
            "Risk-Free Rate (r)": pricer.r,
            "Volatility (σ)": pricer.sigma,
        }
        call_data, put_data = price_grid("Spot Price (S)", "Volatility (σ)", spot_range, vol_range, constants)

        col5, col6 = st.columns(2, gap='small')

        with col5:
            st.subheader('Call Price Heatmap')
            fig, ax = plt.subplots(figsize=(10, 8), dpi=100, constrained_layout=True)
            sns.heatmap(
                call_data,
                xticklabels=np.round(spot_range, 2),
                yticklabels=np.round(vol_range, 2),
                cmap="RdYlGn",
                annot=True,
                fmt=".2f",
                ax=ax
            )
            ax.set_xlabel("Spot Price")
            ax.set_ylabel("Volatility")
            ax.set_title('CALL')
            st.pyplot(fig)

        with col6:
            st.subheader('Put Price Heatmap')
            fig, ax = plt.subplots(figsize=(10, 8), dpi=100, constrained_layout=True)
            sns.heatmap(
                put_data,
                xticklabels=np.round(spot_range, 2),
                yticklabels=np.round(vol_range, 2),
                cmap="RdYlGn",
                annot=True,
                fmt=".2f",
                ax=ax
            )
            ax.set_xlabel("Spot Price")
            ax.set_ylabel("Volatility")
            ax.set_title('PUT')
            st.pyplot(fig)

shared_inputs = {
        "x_axis_param": x_axis_param,
        "y_axis_param": y_axis_param,
        "param_ranges": param_ranges,
        "param_defaults": param_defaults
    }

with tab3:
    st.markdown("## 💹 P&L Scenario Heatmap")
    use_same_price = st.checkbox("☑️ Use same purchase price for both Call and Put", value=False)
    if use_same_price:
        common_price = st.number_input("💰 Common Purchase Price", value=5.0, min_value=0.01, step=0.1)
        call_price_paid = put_price_paid = common_price
    else:
        call_price_paid = st.number_input("💰 Purchase Price for Call Option", value=0.5, min_value=0.01, step=0.1)
        put_price_paid = st.number_input("💰 Purchase Price for Put Option", value=0.5, min_value=0.01, step=0.1)

    x_vals = shared_inputs["param_ranges"][shared_inputs["x_axis_param"]]
    y_vals = shared_inputs["param_ranges"][shared_inputs["y_axis_param"]]
    fixed_vals = shared_inputs["param_defaults"].copy()
    fixed_vals.pop(shared_inputs["x_axis_param"])
    fixed_vals.pop(shared_inputs["y_axis_param"])


    x_vals = param_ranges[x_axis_param]
    y_vals = param_ranges[y_axis_param]
    pnl_call_data, pnl_put_data = pnl_grid(
    x_axis_param, y_axis_param, x_vals, y_vals, fixed_params, call_price_paid, put_price_paid)

    col7, col8 = st.columns(2, gap='small')
    # Plotting P&L Heatmap
    with col7:
        st.subheader('Call PNL Heatmap')
        fig, ax = plt.subplots(figsize=(10, 8), dpi=100, constrained_layout=True)
        sns.heatmap(pnl_call_data, xticklabels=np.round(x_vals, 2), yticklabels=np.round(y_vals, 2),
                    cmap="RdYlGn", linewidths=0.2, linecolor='gray',annot=True,fmt=".2f",
                    cbar_kws={"label": "_P&L"}, ax=ax)
        ax.set_xlabel(x_axis_param)
        ax.set_ylabel(y_axis_param)
        ax.set_title(f"Call Option P&L Heatmap")
 
        st.pyplot(fig)
    with col8:
        st.subheader('Put PNL Heatmap')
        fig, ax = plt.subplots(figsize=(10, 8), dpi=100, constrained_layout=True)
        sns.heatmap(pnl_put_data, xticklabels=np.round(x_vals, 2), yticklabels=np.round(y_vals, 2),
                    cmap="RdYlGn", linewidths=0.2, linecolor='gray',annot=True,fmt=".2f",
                    cbar_kws={"label": "_P&L"}, ax=ax)
        ax.set_xlabel(x_axis_param)
        ax.set_ylabel(y_axis_param)
        ax.set_title(f"Put Option P&L Heatmap")

        st.pyplot(fig)

with tab4:
    st.markdown("## 🧩 Strategy P&L: Call + Put Combination")
    view_mode = st.radio("Choose Visualization Mode",["📈 Line Payoff Chart", "🗺️ P&L Heatmap"],horizontal=True)

    strategy_type = st.selectbox("Select Strategy", ["Straddle", "Strangle", "Custom"])
    position_type = st.radio("Select Position Type", ["Long", "Short"], horizontal=True)
    if strategy_type == "Straddle":
        strike_price_common = st.number_input("📍 Common Strike Price (Straddle)", value=strike_price)
        call_strike = put_strike = strike_price_common
    else:
        col1, col2 = st.columns(2)
        with col1:
            call_strike = st.number_input("📈 Call Strike Price", value=strike_price)
        with col2:
            put_strike = st.number_input("📉 Put Strike Price", value=strike_price)

    col3, col4 = st.columns(2)
    with col3:
        call_price_paid = st.number_input("💰 Call Purchase Price", value=1.0)
    with col4:
        put_price_paid = st.number_input("💰 Put Purchase Price", value=1.0)
    is_short = (position_type == "Short")
    if view_mode == "📈 Line Payoff Chart":
        S_range = np.linspace(spot_price * 0.5, spot_price * 1.5, 200)
        if strategy_type == "Straddle":
            pnl = straddle_pnl_curve(S_range, call_strike, put_strike, call_price_paid, put_price_paid, short=is_short)

        elif strategy_type == "Strangle":
            pnl = strangle_pnl_curve(S_range, call_strike, put_strike, call_price_paid, put_price_paid, short=is_short)

        else:  # Custom
            pnl = []
            for S in S_range:
                call_pnl = max(S - call_strike, 0) - call_price_paid
                put_pnl = max(put_strike - S, 0) - put_price_paid
                total = call_pnl + put_pnl
                pnl.append(-total if is_short else total)
            pnl = np.array(pnl)

        #pnl = []
        # for S in S_range:
        #     call_pnl = max(S - call_strike, 0) - call_price_paid
        #     put_pnl = max(put_strike - S, 0) - put_price_paid
        #     total = call_pnl + put_pnl
        #     pnl.append(-total if is_short else total)


        spacer_left, main, spacer_right = st.columns([1, 5, 1])
        with main:
            fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
            ax.plot(S_range, pnl, color="blue", linewidth=2)
            ax.axhline(0, color="black", linestyle="--")
            ax.set_xlabel("Spot Price at Expiration")
            ax.set_ylabel("Net P&L")
            ax.set_title(f"{strategy_type} Payoff Curve")
            st.pyplot(fig)

    else:        
        x_vals = shared_inputs["param_ranges"][shared_inputs["x_axis_param"]]
        y_vals = shared_inputs["param_ranges"][shared_inputs["y_axis_param"]]
        fixed_vals = shared_inputs["param_defaults"].copy()
        fixed_vals.pop(shared_inputs["x_axis_param"])
        fixed_vals.pop(shared_inputs["y_axis_param"])



        def combined_strategy_heatmap(x_name, y_name, x_vals, y_vals, constants, call_strike, put_strike, call_cost, put_cost, is_short=False):
                pnl_matrix = np.zeros((len(y_vals), len(x_vals)))
                for i, y in enumerate(y_vals):
                    for j, x in enumerate(x_vals):
                        params = constants.copy()
                        params[x_name] = x
                        params[y_name] = y

                        spot = params["Spot Price (S)"]
                        ttm = params["Time to Maturity (T)"]
                        vol = params["Volatility (σ)"]
                        rate = params["Risk-Free Rate (r)"]

                        call_leg = OptionPricer(spot, call_strike, rate, vol,ttm)
                        put_leg = OptionPricer(spot, put_strike, rate, vol,ttm)
                        call_leg.run()
                        put_leg.run()

                        if is_short:
                            net_pnl = (call_cost - call_leg.call_price) + (put_cost - put_leg.put_price)
                        else:
                            net_pnl = (call_leg.call_price - call_cost) + (put_leg.put_price - put_cost)

                        pnl_matrix[i, j] = net_pnl
                return pnl_matrix



        strat_pnl_matrix = combined_strategy_heatmap(shared_inputs["x_axis_param"], shared_inputs["y_axis_param"], x_vals, y_vals, fixed_vals,
                                                    call_strike, put_strike, call_price_paid, put_price_paid, is_short=is_short)


        spacer_left, main, spacer_right = st.columns([1, 5, 1])

        with main:
            fig, ax = plt.subplots(figsize=(10, 8), dpi=100, constrained_layout=True)
            sns.heatmap(strat_pnl_matrix, xticklabels=np.round(x_vals, 2), yticklabels=np.round(y_vals, 2),
                            cmap="RdYlGn", linewidths=0.2, linecolor='gray',annot=True,fmt=".2f",
                        cbar_kws={"label": "Net Strategy P&L"}, ax=ax)
            ax.set_xlabel(shared_inputs["x_axis_param"])
            ax.set_ylabel(shared_inputs["y_axis_param"])
            ax.set_title("Strategy P&L Heatmap")
            st.pyplot(fig)


with tab5:
    def recommend_strategy(volatility, TTM, spot_price, strike_price):
        # if volatility < 0.15:
        #     return "Credit Spread", "Low volatility → safer income strategies"
        if volatility > 0.4:
            return "Straddle", "High volatility → expect big movement"
        elif abs(spot_price - strike_price) < 0.05 * strike_price and TTM > 0.3:
            return "Iron Condor", "Neutral market, decent TTM, range-bound"
        elif TTM < 0.15 and volatility > 0.3:
            return "Butterfly Spread", "Short TTM, volatile → directional bet"
        else:
            return "Iron Condor", "Defaulting to risk-defined neutral strategy"
        

    st.markdown("## 🎯 Strategy Recommender and Payoff Visualizer")
    suggested, reasoning = recommend_strategy(volatility, TTM, spot_price, strike_price)
    st.info(f"**Recommended Strategy:** {suggested}\n\n💡 *{reasoning}*")

    st.markdown("Iron Condor and Butterfly Options Strategy")
    strategy=  st.selectbox('📌 Select Strategy', ['Iron Condor' ,'Reverse Iron Condor' ,'Butterfly_Spread','Reverse Butterfly_Spread'])

    view_mode = st.radio(
                "Choose Visualization Mode",
                ["📈 Line Payoff Chart", "🗺️ Payoff Heatmap"],
    horizontal=True)
    strike_spread = st.number_input("Strike Spread Distance", min_value=1.0, value=5.0, step=1.0)
    base_strike = strike_price
    premium_input = st.checkbox('Manual Premium Input')
    expected_move_pct  = volatility*np.sqrt(TTM)
    S_vals = np.linspace(spot_price*(1-2*expected_move_pct) ,spot_price*(1+2*expected_move_pct),50)
    call_lower =  base_strike-strike_spread
    call_upper = base_strike +strike_spread

    if strategy == "Iron Condor" or strategy ==  'Reverse Iron Condor':
        #Sell OTM Put , Buy OTM Put (SP< short Put SP) , Sell OTM Call ,  Buy OTM Call(SP> short Call SP)
        put_lower = base_strike -strike_spread
        put_outer  = put_lower -strike_spread 
        call_upper = base_strike +strike_spread
        call_outer  = call_upper+strike_spread

        if premium_input:
            buy_put = st.number_input("Buy Put (LL) Premium", value=1.0)
            sell_put = st.number_input("Sell Put (L) Premium", value=2.0)
            sell_call = st.number_input("Sell Call (H) Premium", value=2.0)
            buy_call = st.number_input("Buy Call (HH) Premium", value=1.0)

        else:
            pricer_buy_put = OptionPricer(spot_price, put_outer, interest_rate, volatility, TTM)
            pricer_buy_put.run()
            buy_put = pricer_buy_put.put_price

            pricer_sell_put = OptionPricer(spot_price, put_lower, interest_rate, volatility, TTM)
            pricer_sell_put.run()
            sell_put = pricer_sell_put.put_price

            pricer_sell_call = OptionPricer(spot_price, call_upper, interest_rate, volatility, TTM)
            pricer_sell_call.run()
            sell_call = pricer_sell_call.call_price

            pricer_buy_call = OptionPricer(spot_price, call_outer, interest_rate, volatility, TTM)
            pricer_buy_call.run()
            buy_call = pricer_buy_call.call_price

        net_credit = sell_put + sell_call - (buy_put + buy_call)
        if view_mode == "📈 Line Payoff Chart":
            # def iron_condor_pnl(S_val):
            #     payoff = (-max(put_lower -S_val,0) + max(put_outer-S_val,0) - max(S_val-call_upper,0) + max(S_val-call_outer,0))
            #     return payoff+net_credit if strategy == "Iron Condor" else -payoff - net_credit
            
            # # expected_move_pct  = volatility*np.sqrt(TTM)
            # # S_vals = np.linspace(spot_price*(1-2*expected_move_pct) ,spot_price*(1+2*expected_move_pct),50)
            # pnl_vals = [iron_condor_pnl(s) for s in S_vals] 
            # spacer_left, main, spacer_right = st.columns([1, 5, 1])
            # with main:
            #     fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
            #     ax.plot(S_vals, pnl_vals, label="Iron Condor P&L", color="blue", linewidth=2)
            #     ax.axhline(0, color="black", linestyle="--")
            #     ax.set_xlabel("Spot Price at Expiration")
            #     ax.set_ylabel("P&L")
            #     ax.set_title("Iron Condor Strategy Payoff")
            #     ax.legend()
            #     st.pyplot(fig)
            def iron_condor_total_pnl(S_val):
                base_payoff = iron_condor_payoff_at_expiry(S_val, put_lower, put_outer, call_upper, call_outer)
                # Net credit computed at current spot/vol/rate/TTM
                net_credit_local = iron_condor_net_credit(spot_price, interest_rate, volatility, TTM,
                                                        put_outer, put_lower, call_upper, call_outer)
                if strategy == "Iron Condor":
                    return base_payoff + net_credit_local
                else:  # Reverse Iron Condor
                    return -(base_payoff + net_credit_local)
            pnl_vals = [iron_condor_total_pnl(s) for s in S_vals]

            spacer_left, main, spacer_right = st.columns([1, 5, 1])
            with main:
                fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
                ax.plot(S_vals, pnl_vals, label=f"{strategy} P&L", linewidth=2)
                ax.axhline(0, color="black", linestyle="--")
                ax.set_xlabel("Spot Price at Expiration")
                ax.set_ylabel("P&L")
                ax.set_title(f"{strategy} Strategy Payoff")
                ax.legend()
                st.pyplot(fig)

        elif view_mode == "🗺️ Payoff Heatmap":
        #     strike_spreads = np.arange(2, 11, 1)

        #     payoff_matrix = np.zeros((len(strike_spreads), len(S_vals)))
        #     for i, spread in enumerate(strike_spreads):
        #         pl = strike_price - spread
        #         po = pl - spread
        #         cu = strike_price + spread
        #         co = cu + spread
        #         buy_put = OptionPricer(spot_price, po, interest_rate, volatility, TTM); buy_put.run()
        #         sell_put = OptionPricer(spot_price, pl, interest_rate, volatility, TTM); sell_put.run()
        #         sell_call = OptionPricer(spot_price, cu, interest_rate, volatility, TTM); sell_call.run()
        #         buy_call = OptionPricer(spot_price, co, interest_rate, volatility, TTM); buy_call.run()
        #         credit = sell_put.put_price + sell_call.call_price - (buy_put.put_price + buy_call.call_price)

        #         for j, S_exp in enumerate(S_vals):
        #             payoff = (-max(pl - S_exp, 0) + max(po - S_exp, 0) - max(S_exp - cu, 0) + max(S_exp - co, 0))
        #             payoff_matrix[i, j] = payoff + credit if strategy == "Iron Condor" else -payoff - credit

        #     # Create 3-column layout to center content
        #     spread_data = []
        #     for spread in np.arange(2, 11, 1):
        #         pl = strike_price - spread
        #         po = pl - spread
        #         cu = strike_price + spread
        #         co = cu + spread

        # # Use OptionPricer to get premiums
        #         buy_put = OptionPricer(spot_price, po, interest_rate, volatility, TTM); buy_put.run()
        #         sell_put = OptionPricer(spot_price, pl, interest_rate, volatility, TTM); sell_put.run()
        #         sell_call = OptionPricer(spot_price, cu, interest_rate, volatility, TTM); sell_call.run()
        #         buy_call = OptionPricer(spot_price, co, interest_rate, volatility, TTM); buy_call.run()

        #         net_credit = sell_put.put_price + sell_call.call_price - (buy_put.put_price + buy_call.call_price)

        #         if strategy == "Iron Condor":
        #             max_profit = round(net_credit, 2)
        #             max_loss = round(spread - net_credit, 2)
        #             breakeven_low = round(pl - net_credit, 2)
        #             breakeven_high = round(cu + net_credit, 2)
        #         else:  # Reverse Iron Condor
        #             max_profit = round(spread - net_credit, 2)
        #             max_loss = round(net_credit, 2)
        #             breakeven_low = round(pl - net_credit, 2)
        #             breakeven_high = round(cu + net_credit, 2)

        #         spread_data.append({
        #             "Spread Width": spread,
        #             "Put Spread": f"{po}-{pl}",
        #             "Call Spread": f"{cu}-{co}",
        #             "Max Profit ($)": max_profit,
        #             "Max Loss ($)": max_loss,
        #             "Breakeven Low": breakeven_low,
        #             "Breakeven High": breakeven_high
        #         })
        #     spacer_left, main, spacer_right = st.columns([1, 5, 1])

        #     with main:
        #         st.markdown("### 📊 Iron Condor P&L Heatmap (Volatility-Adjusted)")

        #         fig, ax = plt.subplots(figsize=(8, 4), dpi=100)  # Shrunk size
        #         sns.heatmap(
        #             payoff_matrix,
        #             xticklabels=np.round(S_vals, 1),
        #             yticklabels=strike_spreads,
        #             cmap="RdYlGn",
        #             annot=False,
        #             fmt=".2f",
        #             cbar_kws={"label": "Net P&L"},
        #             ax=ax
        #         )
        #         ax.set_xlabel("Spot Price at Expiration (σ√T adjusted)")
        #         ax.set_ylabel("Strike Spread Width")
        #         ax.set_title(f"{strategy} P&L Heatmap (Volatility-Adjusted)")
        #         ax.set_xticks(ax.get_xticks()[::5])  # adjust tick density
        #         plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
        #         st.pyplot(fig)
        #         st.markdown("### 📋 Iron Condor Summary Table (Strike Spread vs. Profit/Loss)")


        #         df_summary = pd.DataFrame(spread_data)
        #         st.dataframe(df_summary.style.format(precision=2), use_container_width=True)
        # --- NEW Iron Condor heatmap using helpers ---
            strike_spreads = np.arange(2, 11, 1)
            payoff_matrix = np.zeros((len(strike_spreads), len(S_vals)))
            spread_data = []

            for i, spread in enumerate(strike_spreads):
                pl = strike_price - spread       # put inner (short put strike)
                po = pl - spread                 # put outer (long put strike)
                cu = strike_price + spread       # call inner (short call strike)
                co = cu + spread                 # call outer (long call strike)

                # Net credit at current spot/r/vol/T
                credit_i = iron_condor_net_credit(spot_price, interest_rate, volatility, TTM, po, pl, cu, co)

                # Fill payoff across S
                for j, S_exp in enumerate(S_vals):
                    base_payoff = iron_condor_payoff_at_expiry(S_exp, pl, po, cu, co)
                    payoff = base_payoff + credit_i
                    payoff_matrix[i, j] = payoff if strategy == "Iron Condor" else -payoff

                # Summary metrics (same formulas as before, but use credit_i)
                if strategy == "Iron Condor":
                    max_profit = round(credit_i, 2)
                    max_loss   = round(spread - credit_i, 2)
                    breakeven_low  = round(pl - credit_i, 2)
                    breakeven_high = round(cu + credit_i, 2)
                    net_label = "Net Credit ($)"
                    net_value = round(credit_i, 2)
                else:  # Reverse Iron Condor
                    max_profit = round(spread - credit_i, 2)
                    max_loss   = round(credit_i, 2)
                    breakeven_low  = round(pl - credit_i, 2)
                    breakeven_high = round(cu + credit_i, 2)
                    net_label = "Net Debit ($)"   # reverse usually costs a debit
                    net_value = round(credit_i, 2)  # keep sign as computed; label clarifies

                spread_data.append({
                    "Spread Width": spread,
                    "Put Spread": f"{po}-{pl}",
                    "Call Spread": f"{cu}-{co}",
                    net_label: net_value,
                    "Max Profit ($)": max_profit,
                    "Max Loss ($)": max_loss,
                    "Breakeven Low": breakeven_low,
                    "Breakeven High": breakeven_high
                })

            spacer_left, main, spacer_right = st.columns([1, 5, 1])

            with main:
                st.markdown("### 📊 Iron Condor P&L Heatmap (Volatility-Adjusted)")
                fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
                sns.heatmap(
                    payoff_matrix,
                    xticklabels=np.round(S_vals, 1),
                    yticklabels=strike_spreads,
                    cmap="RdYlGn",
                    annot=False,
                    fmt=".2f",
                    cbar_kws={"label": "Net P&L"},
                    ax=ax
                )
                ax.set_xlabel("Spot Price at Expiration (σ√T adjusted)")
                ax.set_ylabel("Strike Spread Width")
                ax.set_title(f"{strategy} P&L Heatmap (Volatility-Adjusted)")
                if len(S_vals) > 20:
                    ax.set_xticks(ax.get_xticks()[::5])
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
                st.pyplot(fig)

                st.markdown("### 📋 Iron Condor Summary Table (Strike Spread vs. Profit/Loss)")
                df_summary = pd.DataFrame(spread_data)
                st.dataframe(df_summary.style.format(precision=2), use_container_width=True)
    elif strategy in ["Butterfly_Spread", "Reverse Butterfly_Spread"]:
        lower = base_strike- strike_spread
        middle =base_strike
        upper = base_strike +strike_spread


        if premium_input:
            buy_lower = st.number_input('But Lower Strike Premium' , value = 1.0)
            sell_middle = st.number_input('Buy Middle Strike Premium' , value =2.0 )
            buy_upper  = st.number_input('Buy Upper Strike Premium' , value= 1.0)

        else:
            pricer_buy_lower = OptionPricer(spot_price, lower, interest_rate, volatility, TTM)
            pricer_buy_lower.run()
            buy_lower = pricer_buy_lower.call_price

            pricer_sell_middle = OptionPricer(spot_price, middle, interest_rate, volatility, TTM)
            pricer_sell_middle.run()
            sell_middle = pricer_sell_middle.call_price

            pricer_buy_upper = OptionPricer(spot_price, upper, interest_rate, volatility, TTM)
            pricer_buy_upper.run()
            buy_upper = pricer_buy_upper.call_price

        net_debit = buy_lower + buy_upper - 2*sell_middle
        
        def butterfly_pnl(S):
            pnl = max(S - lower, 0) - 2 * max(S - middle, 0) + max(S - upper, 0)
            return pnl-net_debit
  
        pnl_vals_b = [butterfly_pnl(s) for s in S_vals] 
        spacer_left, main, spacer_right = st.columns([1, 5, 1])
        if view_mode == "📈 Line Payoff Chart":
            # with main:
            #     fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
            #     ax.plot(S_vals, pnl_vals_b, label="Butterfly P&L", color="blue", linewidth=2)
            #     ax.axhline(0, color="black", linestyle="--")
            #     ax.set_xlabel("Spot Price at Expiration")
            #     ax.set_ylabel("P&L")
            #     ax.set_title("Butterfly Strategy Payoff")
            #     ax.legend()
            #     st.pyplot(fig)
            # --- NEW Butterfly line payoff using helper ---
            pnl_vals_b = butterfly_pnl_curve(
                S_vals,
                lower,   # base_strike - strike_spread
                middle,  # base_strike
                upper,   # base_strike + strike_spread
                net_debit,  # buy_lower + buy_upper - 2*sell_middle
                short=(strategy == "Reverse Butterfly_Spread")
            )

            spacer_left, main, spacer_right = st.columns([1, 5, 1])
            with main:
                fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
                ax.plot(S_vals, pnl_vals_b, label=f"{strategy} P&L", linewidth=2)
                ax.axhline(0, color="black", linestyle="--")
                ax.set_xlabel("Spot Price at Expiration")
                ax.set_ylabel("P&L")
                ax.set_title(f"{strategy} Strategy Payoff")
                ax.legend()
                st.pyplot(fig)

        elif view_mode == "🗺️ Payoff Heatmap":
                # strike_spreads = np.arange(2,11,1)
                # payoff_matrix = np.zeros((len(strike_spreads),len(S_vals)))
                # spread_data_butterfly = []
                # for i,spread in enumerate(strike_spreads):
                #     l = base_strike - spread
                #     m = base_strike
                #     u = base_strike + spread

                #     pricer_l = OptionPricer(spot_price, l, interest_rate, volatility, TTM);pricer_l.run()
                #     pricer_m = OptionPricer(spot_price, m, interest_rate, volatility, TTM);pricer_m.run()
                #     pricer_u = OptionPricer(spot_price, u, interest_rate, volatility, TTM);pricer_u.run()


                #     net_premium = pricer_l.call_price + pricer_u.call_price -2*pricer_m.call_price
            
                #     for j,S_exp in enumerate(S_vals):
                #         pnl = max(S_exp - l, 0) - 2 * max(S_exp - m, 0) + max(S_exp - u, 0)
                #         payoff = pnl-net_premium
                #         payoff_matrix[i,j] = payoff if strategy == "Butterfly_Spread" else -payoff
                


                #     if strategy == "Butterfly_Spread":
                #         max_profit = round(max(spread - net_premium, 0), 2)
                #         max_loss = round(net_premium, 2)
                #         breakeven_low = round(l + net_premium, 2)
                #         breakeven_high = round(u - net_premium, 2)
                #     else:  # Reverse Butterfly
                #         max_profit = round(net_premium, 2)
                #         max_loss = round(max(spread - net_premium, 0), 2)
                #         breakeven_low = round(l + (spread - net_premium), 2)
                #         breakeven_high = round(u - (spread - net_premium), 2)
                #     net_label = "Net Debit ($)" if strategy == "Butterfly_Spread" else "Net Credit ($)"
                #     spread_data_butterfly.append({
                #         "Spread Width": spread,
                #         "Strikes": f"{l}-{m}-{u}",
                #         net_label:round(net_premium, 2),
                #         "Max Profit ($)": max_profit,
                #         "Max Loss ($)": max_loss,
                #         "Breakeven Low": breakeven_low,
                #         "Breakeven High": breakeven_high})

                # with main:
                #     fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
                #     sns.heatmap(
                #         payoff_matrix,
                #         xticklabels=np.round(S_vals, 1),
                #         yticklabels=strike_spreads,
                #         cmap="RdYlGn",
                #         annot=False,
                #         fmt=".2f",
                #         cbar_kws={"label": "Net P&L"},
                #         ax=ax
                #     )
                #     ax.set_xlabel("Spot Price at Expiration")
                #     ax.set_ylabel("Strike Spread Width")
                #     ax.set_title(f"{strategy} P&L Heatmap (Volatility-Adjusted)")
                #     if len(S_vals) > 20:
                #         ax.set_xticks(ax.get_xticks()[::5])
                #     #ax.set_xticks(ax.get_xticks()[::5])
                #     plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
                #     st.pyplot(fig)  
                #     st.markdown("### 📋 Butterfly Spread Summary Table (Strike Spread vs. Profit/Loss)")
                #     df_butterfly_summary = pd.DataFrame(spread_data_butterfly)
                #     st.dataframe(df_butterfly_summary.style.format(precision=2), use_container_width=True)

                # --- NEW Butterfly heatmap (volatility-adjusted S grid) + summary table ---
                strike_spreads = np.arange(2, 11, 1)             # widths to sweep (L/M/U distance)
                payoff_matrix = np.zeros((len(strike_spreads), len(S_vals)))
                spread_data_butterfly = []

                for i, spread in enumerate(strike_spreads):
                    l = base_strike - spread
                    m = base_strike
                    u = base_strike + spread

                    # Premiums at current spot/r/vol/T (use calls, symmetric construction)
                    pricer_l = OptionPricer(spot_price, l, interest_rate, volatility, TTM); pricer_l.run()
                    pricer_m = OptionPricer(spot_price, m, interest_rate, volatility, TTM); pricer_m.run()
                    pricer_u = OptionPricer(spot_price, u, interest_rate, volatility, TTM); pricer_u.run()

                    net_premium = pricer_l.call_price + pricer_u.call_price - 2 * pricer_m.call_price  # net_debit for standard butterfly

                    # Fill payoff across S (expiry)
                    for j, S_exp in enumerate(S_vals):
                        base_payoff = (max(S_exp - l, 0) - 2 * max(S_exp - m, 0) + max(S_exp - u, 0))
                        payoff = base_payoff - net_premium
                        payoff_matrix[i, j] = payoff if strategy == "Butterfly_Spread" else -payoff

                    # Summary metrics
                    if strategy == "Butterfly_Spread":
                        # Long butterfly: typically a net debit
                        max_profit     = round(max(spread - net_premium, 0), 2)
                        max_loss       = round(max(net_premium, 0), 2)
                        breakeven_low  = round(l + net_premium, 2)
                        breakeven_high = round(u - net_premium, 2)
                        net_label, net_value = "Net Debit ($)", round(net_premium, 2)
                    else:
                        # Reverse butterfly: typically a net credit (profit on wings, risk near middle)
                        max_profit     = round(max(net_premium, 0), 2)
                        max_loss       = round(max(spread - net_premium, 0), 2)
                        breakeven_low  = round(l + (spread - net_premium), 2)
                        breakeven_high = round(u - (spread - net_premium), 2)
                        net_label, net_value = "Net Credit ($)", round(net_premium, 2)  # keep sign as computed; label clarifies

                    spread_data_butterfly.append({
                        "Spread Width": spread,
                        "Strikes": f"{l}-{m}-{u}",
                        net_label: net_value,
                        "Max Profit ($)": max_profit,
                        "Max Loss ($)": max_loss,
                        "Breakeven Low": breakeven_low,
                        "Breakeven High": breakeven_high
                    })

                spacer_left, main, spacer_right = st.columns([1, 5, 1])
                with main:
                    st.markdown(f"### 📊 {strategy} P&L Heatmap (Volatility-Adjusted)")
                    fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
                    sns.heatmap(
                        payoff_matrix,
                        xticklabels=np.round(S_vals, 1),
                        yticklabels=strike_spreads,
                        cmap="RdYlGn",
                        annot=False,
                        fmt=".2f",
                        cbar_kws={"label": "Net P&L"},
                        ax=ax
                    )
                    ax.set_xlabel("Spot Price at Expiration (σ√T adjusted)")
                    ax.set_ylabel("Strike Spread Width")
                    ax.set_title(f"{strategy} P&L Heatmap (Volatility-Adjusted)")
                    if len(S_vals) > 20:
                        ax.set_xticks(ax.get_xticks()[::5])
                    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
                    st.pyplot(fig)

                    st.markdown("### 📋 Butterfly Spread Summary Table (Strike Spread vs. Profit/Loss)")
                    df_butterfly_summary = pd.DataFrame(spread_data_butterfly)
                    st.dataframe(df_butterfly_summary.style.format(precision=2), use_container_width=True)
