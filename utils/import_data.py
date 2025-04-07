import os
import sys
from typing import Callable

import pandas as pd
import requests
import yfinance as yf

from constants2 import (
    DATA_FILES_EXTENSION,
    LOCAL_FOLDER,
    TICKER_DATA_RAW_FILENAME_PREFIX,
    TICKER_DATA_W_FEATURES_FILENAME_PREFIX,
)

def import_ohlc_yfinance(
    ticker: str, period: str = "2y", interval: str = "1d"
) -> pd.DataFrame:
    """
    Get OHLC DataFrame with Volume from Yahoo Finance.
    Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max.
    Valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
    """
    res = yf.Ticker(ticker=ticker).history(period=period, interval=interval)

    # NOTE  If period and interval mismatch, Yahoo Finance returns empty DataFrame.
    # A mismatch is an interval too small for a long period.
    if res.shape[0] == 0:
        raise RuntimeError(
            f"import_yahoo_daily: Yahoo Finance returned empty Df for {ticker=}, mabe mismatch between {period=} and {interval=}"
        )

    # Ensure the index is timezone-unaware 
    if res.index.tz is not None:
        res.index = res.index.tz_convert(None) #Remove the timezone
        print(f"Timezone removed from index for {ticker=}, new timezone: {res.index.tz}")
    else:
        print(f"No timezone present in index for {ticker=}")

    return res[["Open", "High", "Low", "Close", "Volume"]]

def get_local_ticker_data_file_name(ticker: str, data_type: str = "raw") -> str:
    internal_ticker = ticker.upper()
    if data_type == "raw":
        return (
            LOCAL_FOLDER
            + TICKER_DATA_RAW_FILENAME_PREFIX
            + internal_ticker
            + DATA_FILES_EXTENSION
        )
    if data_type == "with_features":
        return (
            LOCAL_FOLDER
            + TICKER_DATA_W_FEATURES_FILENAME_PREFIX
            + internal_ticker
            + DATA_FILES_EXTENSION
        )
    raise ValueError(
        f"get_local_ticker_data_file_name: wrong {data_type=}, should be raw or with_features"
    )
