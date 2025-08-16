# tabs/scenario_heatmap.py

import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from Option_pricer import OptionPricer

def render(shared_inputs):
    st.markdown("## 🌡️ Scenario Heatmap")

    spot_price = shared_inputs["spot_price"]
    strike_price = shared_inputs["strike_price"]
    TTM = shared_inputs["TTM"]
    interest_rate = shared_inputs["interest_rate"]
    volatility = shared_inputs["volatility"]

    x_axis_param = shared_inputs["x_axis_param"]
    y_axis_param = shared_inputs["y_axis_param"]
    param_ranges = shared_inputs["param_ranges"]

    param_defaults = {
        "Spot Price (S)": spot_price,
        "Volatility (σ)": volatility,
        "Time to Maturity (T)": TTM,
        "Risk-Free Rate (r)": interest_rate,
        "Strike Price (K)": strike_price
    }

    fixed_params = param_defaults.copy()
    fixed_params.pop(x_axis_param)
    fixed_params.pop(y_axis_param)

    def dynamic_heatmap(x_name, y_name, x_vals, y_vals, constants):
        call_matrix = np.zeros((len(y_vals), len(x_vals)))
        put_matrix = np.zeros((len(y_vals), len(x_vals)))
        for i, y in enumerate(y_vals):
            for j, x in enumerate(x_vals):
                inputs = constants.copy()
                inputs[x_name] = x
                inputs[y_name] = y
                pricer = OptionPricer(
                    inputs["Spot Price (S)"],
                    inputs["Strike Price (K)"],
                    inputs["Risk-Free Rate (r)"],
                    inputs["Volatility (σ)"],
                    inputs["Time to Maturity (T)"]
                )
                pricer.run()
                call_matrix[i, j] = pricer.call_price
                put_matrix[i, j] = pricer.put_price
        return call_matrix, put_matrix

    x_vals = param_ranges[x_axis_param]
    y_vals = param_ranges[y_axis_param]
    call_data, put_data = dynamic_heatmap(x_axis_param, y_axis_param, x_vals, y_vals, fixed_params)

    col1, col2 = st.columns(2, gap='small')
    with col1:
        st.subheader("Call Price Heatmap")
        fig, ax = plt.subplots(figsize=(10, 8), dpi=100, constrained_layout=True)
        sns.heatmap(call_data, xticklabels=np.round(x_vals, 2), yticklabels=np.round(y_vals, 2),
                    cmap="RdYlGn", annot=True, fmt=".2f", ax=ax)
        ax.set_xlabel(x_axis_param)
        ax.set_ylabel(y_axis_param)
        ax.set_title("CALL")
        st.pyplot(fig)

    with col2:
        st.subheader("Put Price Heatmap")
        fig, ax = plt.subplots(figsize=(10, 8), dpi=100, constrained_layout=True)
        sns.heatmap(put_data, xticklabels=np.round(x_vals, 2), yticklabels=np.round(y_vals, 2),
                    cmap="RdYlGn", annot=True, fmt=".2f", ax=ax)
        ax.set_xlabel(x_axis_param)
        ax.set_ylabel(y_axis_param)
        ax.set_title("PUT")
        st.pyplot(fig)
