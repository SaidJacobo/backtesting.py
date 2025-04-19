"""Data and utilities for testing."""

from __future__ import annotations

import pandas as pd


def _read_file(filename):
    from os.path import dirname, join

    return pd.read_csv(join(dirname(__file__), filename),
                       index_col=0, parse_dates=True)


BTCUSD = _read_file('BTCUSD.csv')
"""DataFrame of monthly BTC/USD histrical index data from 2012 through 2024 (12 years)."""

GOOG = _read_file('GOOG.csv')
"""DataFrame of daily NASDAQ:GOOG (Google/Alphabet) stock price data from 2004 to 2013."""

EURUSD = _read_file('EURUSD.csv')
"""DataFrame of hourly EUR/USD forex data from April 2017 to February 2018."""

USDJPY = _read_file('USDJPY.csv')
"""DataFrame of hourly USDJPY forex data from January 2020 to December 2024."""

GBPNZD = _read_file('GBPNZD.csv')
"""DataFrame of hourly GBPNZD forex data from January 2020 to December 2024."""

NZDUSD = _read_file('NZDUSD.csv')
"""DataFrame of hourly NZDUSD forex data from January 2020 to December 2024."""

def SMA(arr: pd.Series, n: int) -> pd.Series:
    """
    Returns `n`-period simple moving average of array `arr`.
    """
    return pd.Series(arr).rolling(n).mean()
