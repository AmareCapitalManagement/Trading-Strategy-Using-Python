# Trading-Strategy-Using-Python
Amare Capital Management (Pty) Ltd 

Amare Capital Management (Pty) Ltd is a systematic proprietary trading firm dedicated to developing and refining quantitative trading strategies. Our approach focuses in simplifying and enhancing trading strategies through rigorous statistical analysis and robust backtesting, leveraging a Python-based framework.

![ACM w color](https://github.com/user-attachments/assets/85e2320e-5494-4713-8c31-b8aeece758b5)

# PYTHON BACKTESTING: HAMMER REVERSAL WITH VOLATILITY FILTER 

This strategy identifies potential bullish reversals using the hammer candlestick pattern, filtered by the asset's position relative to its 200-day moving average and volatility conditions measured by the True Range Delta. It aims to enter long positions when a hammer candle forms under specific conditions and manage risk with stop-losses, profit-targets, and specific situation handling. The approach is backtested across multiple tickers to ensure robustness, reflecting Amare Capital Management's commitment to rigorous statistical validation.

STEP 1: DATA PREPARATION

(GATHER AND PROCESS OHLC(OPEN, HIGH, LOW, CLOSE) DATA FOR MULTIPLE TICKERS, ADDING DERIVED COLUMNS AND FEATURES)

    from typing import Callable, List 
    import pandas as pd
    import os
    import yfinance as yf
    from derivative_columns.atr import add_tr_delta_col_to_ohlc
    from utils.import_data import get_local_ticker_data_file_name 

    MUST_HAVE_DERIVATIVE_COLUMNS = {"tr", "tr_delta"}

    def import_yahoo_finance_daily(ticker:str) -> pd.DataFrame:
        stock = yf.Ticker(ticker)
        df = stock.history(start="2020-01-01", end="2025_04_06", interval="1d")
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        df.index = pd.to.datetime(df.index).tz_localize(None)
        return df 

    class TickersData:
       def __init__(self, tickers: list[str], add_features_cols_func: Callable, import_ohlc_func: Callable = import_yahoo_finance_daily):
          self.tickers_data_with_features = {}
          self.add_features_cols_func = add_features_cols_func
          self.import_ohlc_func = add_features_cols_func
          for ticker in tickers:
              df = self.get_df_with_features(ticker=ticker)
              for col in MUST_HAVE_DERIVATIVE_COLUMNS:
                  if col not in df.columns:
                     df = add_tr_delta_col_to_ohlc(ohlc_df=df)
              self.tickers_data_with_features[ticker] = df
            
       def get_df_with_features(self, ticker: str) -> pd.DataFrame:
           filename_with_features = get_local_ticker_data_file_name(ticker, "with_features")
           filename_raw = get_local_ticker_data_file_name(ticker, "raw")
           if os.path.exists(filename_with_features):
               return pd.read_excel

