import pandas as pd
import yfinance as yf
import numpy as np
import scipy.stats as si
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from scipy.stats import norm


class OptionPricer:
    def __init__(self, spot_price, strike_price, interest_rate, volatility, TTM):
        self.S = spot_price
        self.K = strike_price
        self.r = interest_rate
        self.sigma = volatility
        self.t = max(TTM, 1e-6)

    def calc_d1_d2(self):
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma**2) * self.t) / (self.sigma * np.sqrt(self.t))
        d2 = d1 - self.sigma * np.sqrt(self.t)
        return d1, d2

    def run(self):
        d1, d2 = self.calc_d1_d2()
        call_option_price =  (norm.cdf(d1) * self.S) - (norm.cdf(d2) * self.K * np.exp(-self.r * self.t))
        put_option_price = (self.K * np.exp(-self.r * self.t) * norm.cdf(-d2)) - (self.S * norm.cdf(-d1))

        self.call_price =  call_option_price
        self.put_price = put_option_price

    def greeks(self):
        d1,d2 = self.calc_d1_d2()
        delta_call  = norm.cdf(d1)
        delta_put = norm.cdf(d1)-1
        gamma = norm.pdf(d1)*(1/(self.S*self.sigma*np.sqrt(self.t)))
        vega = self.S * norm.pdf(d1) * np.sqrt(self.t) / 100
        theta_call = (-self.S * norm.pdf(d1) * self.sigma / (2 * np.sqrt(self.t)) - self.r * self.K * np.exp(-self.r * self.t) * norm.cdf(d2)) / 365
        theta_put = (-self.S * norm.pdf(d1) * self.sigma / (2 * np.sqrt(self.t)) + self.r * self.K * np.exp(-self.r * self.t) * norm.cdf(-d2)) / 365
        rho_call = self.K* self.t*np.exp(-self.r*self.t)*(norm.cdf(d2))/100
        rho_put = -self.K * self.t * np.exp(-self.r * self.t) * norm.cdf(-d2)/100
        return {
            "Delta (Call)": delta_call,
            "Delta (Put)": delta_put,
            "Gamma": gamma,
            "Vega": vega,
            "Theta (Call)": theta_call,
            "Theta (Put)": theta_put,
            "Rho (Call)": rho_call,
            "Rho (Put)": rho_put}


if __name__ == "__main__":
    spot_price =293.65 
    strike_price =330.0 
    interest_rate= 0.05 
    volatility=0.00958252 
    TTM= 4 / 365

    # Black Scholes
    BS = OptionPricer(
        TTM =TTM,
        strike_price=strike_price,
        spot_price=spot_price,
        volatility=volatility,
        interest_rate=interest_rate)
    BS.run()
