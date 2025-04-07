# Trading-Strategy-Using-Python
Amare Capital Management (Pty) Ltd 

Amare Capital Management (Pty) Ltd is a systematic proprietary trading firm dedicated to developing and refining quantitative trading strategies. Our approach focuses in simplifying and enhancing trading strategies through rigorous statistical analysis and robust backtesting, leveraging a Python-based framework.

![ACM w color](https://github.com/user-attachments/assets/85e2320e-5494-4713-8c31-b8aeece758b5)

# PYTHON BACKTESTING: HAMMER REVERSAL WITH VOLATILITY FILTER 

This strategy identifies potential bullish reversals using the hammer candlestick pattern, filtered by the asset's position relative to its 200-day moving average and volatility conditions measured by the True Range Delta. It aims to enter long positions when a hammer candle forms under specific conditions and manage risk with stop-losses, profit-targets, and specific situation handling. The approach is backtested across multiple tickers to ensure robustness, reflecting Amare Capital Management's commitment to rigorous statistical validation.

**STEP 1: DATA PREPARATION**

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

Explanation

The implementation of the TickersData class enables efficient data retrieval from Yahoo Finance, with local caching in Excel files to ensure data integrity and reduce redundancy. By integrating key technical indicators such as the 200-day Moving Average (MA200), Average True Range (ATR), and hammer candle pattern detection, the firm streamlines its data management process, allowing greater focus on the development and refinement of trading strategies.

**STEP 2: FEATURE ENGINEERING**

(Create trading signals based on technical analysis)

    import pandas as pd
    from constants2 import FEATURE_COL_NAME_ADVANCED, FEATURE_COL_NAME_BASIC 
    from derivative_columns.atr import add_atr_col_to_df 
    from derivative_columns.ma import add_moving_average 
    from derivative_columns.hammer import add_col_is_hammer
    from derivative_columns.shooting_star import add_col_is_shooting_star

    MOVING_AVERAGE_N = 200
    REQUIRED_DERIVATIVE_COLUMNS_F_V1_BASIC = {"atr_14", f"ma_{MOVING_AVERAGE_N}", "is_hammer", "is_shooting_star"}

    def add_required_cols_for_f_v1_basic(df: pd.DataFrame) -> pd.DataFrame:
        df_columns = df.columns 
        internal_df = df.copy()
        if f"ma_{MOVING_AVERAGE_N}" not in df_columns:
            internal_df = add_moving_average(df=internal_df, n=MOVING_AVERAGE_N)
        if "atr_14" not in df_columns:
            internal_df = add_atr_col_to_df(df=internal_df, n=14, exponential=False)
        if "is_hammer" not in df_columns:
            internal_df = add_col_is_hammer(df=internal_df)
        if "is_shooting_star" not in df_columns:
            internal_df = add_col_is_shooting_star(df=internal_df)
        return internal_df 

    def add_features_v1_basic(df: pd.DataFrame, atr_multiplier_threshold: int = 6) -> pd.DataFrame:
        res = df.copy()
        for col in REQUIRED_DERIVATIVE_COLUMNS_F_V1_BASIC:
            if col not in res.columns:
                res = add_required_cols_for_f_v1_basic(df=res)
        res[FEATURE_COL_NAME_BASIC] = res["Close"] < res[f"ma_{MOVING_AVERAGE_N}"]
        res[FEATURE_COL_NAME_ADVANCED] = (res["ma_200"] - res["Close"]) >= (res["atr_14"] * atr_multiplier_threshold)
        return res 

Explanation

The add_features_v1_basic function is enhanced to incorporate a hammer candle signal (is_hammer) and to refine FEATURE_COL_NAME_ADVANCED to activate when the stock price is significantly below the 200-day Moving Average (MA200) with a confirmed hammer pattern. This transformation of complex market data into clear, actionable signals supports the firm's mission to elevate trading decisions through statistically grounded methodologies.

**STEP 3: POSITION SIZING LOGIC**

(Define rules for entering, holding, or exiting positions)

    from typing import Optional, Tuple
    from backtesting.backtesting import Strategy 
    from backtesting import set_bokeh_output
    set_bokeh_output(notebook=False)
    from constants2 import DPS_STUB, FEATURE_COL_NAME_ADVANCED
    from utils.strategy_exec.misc import get_current_position_size 

    def get_desired_current_position_size(strategy: Strategy) -> Tuple[Optional[float], float, str]:
        current_position_size = (
            get_current_position_size(
                shares_count=strategy.position.size,
                equity=strategy.equity,
                last_price=strategy._data.Open[-1],
            )
            if strategy.position.size != 0 
            else 0
       )
       is_hammer = strategy._data["is_hammer"][-1]
       price_below_ma200 = strategy._data[FEATURE_COL_NAME_ADVANCED][-1]
       volatility_ok = strategy.data["tr_delta"][-1] < 2.5 

       desired_position_size: Optional[float] = None 
       message = DPS_STUB

       if current_position_size != 0:
           desired_position_size = current_position_size
           message = "Maintain existing position"
           return desired_position_size, current_position_size, message

       if is_hammer and price_below_ma200 and volatility_ok:
           desired_position_size = 1.0
           message = "Enter Long: Hammer reversal below MA200 with moderate volatility"

       return desired_position_size, current_position_size, message 

Explanation 

The get_desired_current_position_size function is modified to initiate a 100% when a hammer candle forms, the price is at least six times the ATR_14 below the 200-day Moving Average (MA200), and volatility (True Range Delta) remains moderate (below 2.5). The function also represents existing positions or exits under specific conditions, reinforcing the firm's disciplined capital allocation framework - an essential pillar of its systematic and data-driven investment strategy.

**STEP 4: RISK MANAGEMENT**

(Protect capital with stop-losses and profit-targets)

    from backtesting import Strategy 
    from constants2 import SL_TIGHTENED 
    import numpy as np

    def _get_n_atr(strategy: Strategy) -> float:
        index = len(strategy.data) - 1
        if strategy.data.tr_delta[index] > 1.98 and strategy.trades and strategy.trades[-1].pl > 0:
            return 1.1 
        return strategy_parameters.stop_loss_default_atr_multiplier

    def update_stop_loss(strategy: Strategy):
        if not strategy.trades:
            return 
        n_atr = _get_n_atr(strategy)
        index = len(strategy.data) - 1
        for trade in strategy.trades:
            if trade.is_long:
                sl_price = max(trade.sl or -np.inf, strategy.data.Open[index] - strategy.data.atr_14[index] * n_atr)
            else:
                sl_price = min(trade.sl or np.inf, strategy.data.Open[index] + strategy.data.atr_14[index] * n_atr)
            if sl_price < 0:
                sl_price = None 
            if sl_price and trade.sl != sl-price: 
                trade.sl = sl-price 
                if n_atr == 1.1 and SL_TIGHTENED not in (trade.tag or ""):
                    setattr(trade, f"_{trade.__class__.__qualname__}__tag", (trade.tag or "") + SL_TIGHTENED)

    def check_set_profit_targets_long_trades(strategy: Strategy):
        last_price = strategy._data.Open[-1]
        min_profit_target_long = None 
        trades_long = [trade for trade in strategy.trades if trade.is_long]
        for trade in trades_long:
            if trade.tp is not None:
                min_profit_target_long = min(min_profit_target_long or trade.tp, trade.tp)
            if trades_long and min_profit_target_long  is None:
                min_profit_target_long = (float(strategy.parameters.profit_target_long_pct + 100) / 100) * last_price
                for trade in trades_long:
                    if trade.tp is None:
                        trade.tp = min_profit_target_long

Explanation

The existing function such as update_stop_losses and check_set_profit_targets_long_trades are utilized with default parameters - a 2.5 ATR multiplier for stop-losses and a 29.9% profit target - to manage trades effectively. During periods of elevated volatility (when true range delta exceeds 1.98), stop-losses are tightened to 1.1 ATR, reinforcing the firm's commitment to robust risk management practice that underpin its pursuit of consistent and sustainable performance.

