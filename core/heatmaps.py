# core/heatmaps.py
import numpy as np
from Option_pricer import OptionPricer

def price_grid(x_name: str, y_name: str, x_vals, y_vals, constants: dict):
    """
    Returns (call_matrix, put_matrix) for a grid over (x_vals, y_vals).
    constants must include keys:
      "Spot Price (S)", "Strike Price (K)", "Time to Maturity (T)",
      "Risk-Free Rate (r)", "Volatility (σ)"
    """
    call = np.zeros((len(y_vals), len(x_vals)))
    put  = np.zeros_like(call)

    for i, y in enumerate(y_vals):
        for j, x in enumerate(x_vals):
            params = constants.copy()
            params[x_name] = x
            params[y_name] = y

            S_ = params["Spot Price (S)"]
            K_ = params["Strike Price (K)"]
            T_ = params["Time to Maturity (T)"]
            r_ = params["Risk-Free Rate (r)"]
            v_ = params["Volatility (σ)"]

            pr = OptionPricer(S_, K_, r_, v_, T_); pr.run()
            call[i, j] = pr.call_price
            put[i, j]  = pr.put_price

    return call, put

def pnl_grid(x_name: str, y_name: str, x_vals, y_vals, constants: dict,
             call_price_paid: float, put_price_paid: float):
    """
    Returns (pnl_call_matrix, pnl_put_matrix) using model price minus purchase price.
    """
    pnl_call = np.zeros((len(y_vals), len(x_vals)))
    pnl_put  = np.zeros_like(pnl_call)

    for i, y in enumerate(y_vals):
        for j, x in enumerate(x_vals):
            params = constants.copy()
            params[x_name] = x
            params[y_name] = y

            S_ = params["Spot Price (S)"]
            K_ = params["Strike Price (K)"]
            T_ = params["Time to Maturity (T)"]
            r_ = params["Risk-Free Rate (r)"]
            v_ = params["Volatility (σ)"]

            pr = OptionPricer(S_, K_, r_, v_, T_); pr.run()
            pnl_call[i, j] = pr.call_price - call_price_paid
            pnl_put[i, j]  = pr.put_price  - put_price_paid

    return pnl_call, pnl_put
