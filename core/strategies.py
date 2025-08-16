# core/strategies.py
import numpy as np
from Option_pricer import OptionPricer

# -------------------------------
# Basic Strategies
# -------------------------------

def straddle_pnl_curve(S_range, Kc, Kp, call_cost, put_cost, short=False):
    """Payoff curve for Straddle (long or short)."""
    pnl = []
    for S in S_range:
        call = max(S - Kc, 0) - call_cost
        put  = max(Kp - S, 0) - put_cost
        total = call + put
        pnl.append(-total if short else total)
    return np.array(pnl)


def strangle_pnl_curve(S_range, call_strike, put_strike, call_cost, put_cost, short=False):
    """Payoff curve for Strangle (long or short)."""
    pnl = []
    for S in S_range:
        call = max(S - call_strike, 0) - call_cost
        put  = max(put_strike - S, 0) - put_cost
        total = call + put
        pnl.append(-total if short else total)
    return np.array(pnl)

# -------------------------------
# Iron Condor
# -------------------------------

def iron_condor_net_credit(S0, r, v, T, put_outer, put_inner, call_inner, call_outer):
    """Net credit (premiums) for Iron Condor setup."""
    p_buy = OptionPricer(S0, put_outer, r, v, T); p_buy.run()
    p_sell= OptionPricer(S0, put_inner, r, v, T); p_sell.run()
    c_sell= OptionPricer(S0, call_inner, r, v, T); c_sell.run()
    c_buy = OptionPricer(S0, call_outer, r, v, T); c_buy.run()
    return (p_sell.put_price + c_sell.call_price) - (p_buy.put_price + c_buy.call_price)

def iron_condor_payoff_at_expiry(S, pl, po, cu, co):
    """Payoff at expiry for Iron Condor legs."""
    return (-max(pl - S, 0) + max(po - S, 0)
            - max(S - cu, 0) + max(S - co, 0))

# -------------------------------
# Butterfly Spread
# -------------------------------

def butterfly_pnl_curve(S_range, lower, middle, upper, net_debit, short=False):
    """Payoff curve for Butterfly or Reverse Butterfly."""
    pnl = []
    for S in S_range:
        payoff = (max(S - lower, 0)
                  - 2 * max(S - middle, 0)
                  + max(S - upper, 0))
        result = payoff - net_debit
        pnl.append(-result if short else result)
    return np.array(pnl)
