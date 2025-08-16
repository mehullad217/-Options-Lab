import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from Option_pricer import OptionPricer

def render(shared_inputs):
    st.markdown("## 📉 P&L Scenario Heatmap")

    x_axis_param = shared_inputs["x_axis_param"]
    y_axis_param = shared_inputs["y_axis_param"]
    param_ranges = shared_inputs["param_ranges"]

    param_defaults = {
        "Spot Price (S)": shared_inputs["spot_price"],
        "Volatility (σ)": shared_inputs["volatility"],
        "Time to Maturity (T)": shared_inputs["TTM"],
        "Risk-Free Rate (r)": shared_inputs["interest_rate"],
        "Strike Price (K)": shared_inputs["strike_price"]
    }

    fixed_vals = param_defaults.copy()
    fixed_vals.pop(x_axis_param)
    fixed_vals.pop(y_axis_param)

    use_same_price = st.checkbox("☑️ Use same purchase price for both Call and Put", value=False)
    if use_same_price:
        common_price = st.number_input("💰 Common Purchase Price", value=5.0, min_value=0.01, step=0.1)
        call_price_paid = put_price_paid = common_price
    else:
        call_price_paid = st.number_input("💰 Purchase Price for Call Option", value=0.5, min_value=0.01, step=0.1)
        put_price_paid = st.number_input("💰 Purchase Price for Put Option", value=0.5, min_value=0.01, step=0.1)

    # Heatmap generation function for P&L
    def pnl_heatmap(x_name, y_name, x_vals, y_vals, constants, call_price_paid, put_price_paid):
        pnl_call_matrix = np.zeros((len(y_vals), len(x_vals)))
        pnl_put_matrix = np.zeros((len(y_vals), len(x_vals)))
        for i, y in enumerate(y_vals):
            for j, x in enumerate(x_vals):
                inputs = constants.copy()
                inputs[x_name] = x
                inputs[y_name] = y

                S_ = inputs["Spot Price (S)"]
                K_ = inputs["Strike Price (K)"]
                t_ = inputs["Time to Maturity (T)"]
                r_ = inputs["Risk-Free Rate (r)"]
                sigma_ = inputs["Volatility (σ)"]

                pricer = OptionPricer(S_, K_, r_, sigma_, t_)
                pricer.run()
                pnl_call = pricer.call_price - call_price_paid
                pnl_put = pricer.put_price - put_price_paid
                pnl_call_matrix[i, j] = pnl_call
                pnl_put_matrix[i, j] = pnl_put

        return pnl_call_matrix, pnl_put_matrix

    x_vals = param_ranges[x_axis_param]
    y_vals = param_ranges[y_axis_param]

    pnl_call_data, pnl_put_data = pnl_heatmap(
        x_axis_param,
        y_axis_param,
        x_vals,
        y_vals,
        fixed_vals,
        call_price_paid,
        put_price_paid
    )

    col7, col8 = st.columns(2, gap='small')

    with col7:
        st.subheader('Call PNL Heatmap')
        fig, ax = plt.subplots(figsize=(10, 8), dpi=100, constrained_layout=True)
        sns.heatmap(pnl_call_data, xticklabels=np.round(x_vals, 2), yticklabels=np.round(y_vals, 2),
                    cmap="RdYlGn", linewidths=0.2, linecolor='gray', annot=True, fmt=".2f",
                    cbar_kws={"label": "_P&L"}, ax=ax)
        ax.set_xlabel(x_axis_param)
        ax.set_ylabel(y_axis_param)
        ax.set_title(f"Call Option P&L Heatmap")
        st.pyplot(fig)

    with col8:
        st.subheader('Put PNL Heatmap')
        fig, ax = plt.subplots(figsize=(10, 8), dpi=100, constrained_layout=True)
        sns.heatmap(pnl_put_data, xticklabels=np.round(x_vals, 2), yticklabels=np.round(y_vals, 2),
                    cmap="RdYlGn", linewidths=0.2, linecolor='gray', annot=True, fmt=".2f",
                    cbar_kws={"label": "_P&L"}, ax=ax)
        ax.set_xlabel(x_axis_param)
        ax.set_ylabel(y_axis_param)
        ax.set_title(f"Put Option P&L Heatmap")
        st.pyplot(fig)
