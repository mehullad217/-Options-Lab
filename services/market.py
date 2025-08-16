# services/market.py
from __future__ import annotations
import pandas as pd
import numpy as np
import pytz
from datetime import datetime
import yfinance as yf

NY_TZ = pytz.timezone("America/New_York")

def compute_ttm_years(expiry_str: str) -> float:
    """ACT/365; assume 4:00pm America/New_York expiry; clamp to >=60s."""
    now_ny = datetime.now(NY_TZ)
    expiry_dt = pd.to_datetime(expiry_str)
    if expiry_dt.tzinfo is None:
        expiry_dt = NY_TZ.localize(expiry_dt)
    else:
        expiry_dt = expiry_dt.astimezone(NY_TZ)
    expiry_dt = expiry_dt.replace(hour=16, minute=0, second=0, microsecond=0)

    seconds = max((expiry_dt - now_ny).total_seconds(), 0.0)
    return max(seconds, 60.0) / (365.0 * 24 * 60 * 60)

def get_risk_free_rate(ttm_years: float) -> float:
    """Return decimal r from Treasury proxies; fallback 5%."""
    if ttm_years <= 0.25:   symbol = "^IRX"  # 13-week
    elif ttm_years <= 1:    symbol = "^FVX"  # 5-year
    elif ttm_years <= 5:    symbol = "^TNX"  # 10-year
    else:                   symbol = "^TYX"  # 30-year
    try:
        data = yf.Ticker(symbol).history(period="1d")
        last_yield_percent = float(data.iloc[-1]["Close"])
        return last_yield_percent / 100.0
    except Exception:
        return 0.05

def get_spot(ticker_symbol: str) -> float | None:
    try:
        hist = yf.Ticker(ticker_symbol).history(period="1d")
        return None if hist.empty else float(hist["Close"].iloc[-1])
    except Exception:
        return None

def get_expiries(ticker_symbol: str) -> list[str]:
    try:
        return yf.Ticker(ticker_symbol).options or []
    except Exception:
        return []

def get_option_chain(ticker_symbol: str, expiry: str):
    return yf.Ticker(ticker_symbol).option_chain(expiry)

def nearest_index(values, target):
    arr = np.array(values, dtype=float)
    return int(np.argmin(np.abs(arr - float(target))))
