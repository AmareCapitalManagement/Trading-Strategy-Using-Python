# Trading-Strategy-Using-Python
Amare Capital Management (Pty) Ltd 

Amare Capital Management (Pty) Ltd is a systematic investment management firm specializing in trading and investing in stocks and stock derivatives. We are committed to developing and refining quantitative trading strategies through rigorous statistical analysis and robust backtesting. Our approach focuses on simplifying and enhancing strategy performance, leveraging a Python-based framework to systematically identify and capitalize on market opportunities.

   ![ACM w color](https://github.com/user-attachments/assets/85e2320e-5494-4713-8c31-b8aeece758b5)

# PYTHON BACKTESTING: TREND-ALIGNED REVERSAL STRATEGY WITH VOLATILITY AND RISK MANAGEMENT FILTERS  

This strategy identifies potential bullish reversals using the hammer candlestick pattern, filtered by the asset's position relative to its 200-day moving average and volatility conditions measured by the Average True Range. It aims to enter long positions when a hammer candle forms under specific conditions and manage risk with stop-losses, profit-targets, and specific situation handling. The approach is backtested across multiple tickers to ensure robustness, reflecting Amare Capital Management's commitment to rigorous statistical validation.

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

**STEP 5: SPECIAL SITUATIONS**

(Address market anomalies to avoid losses)

    from backtesting import Strategy 
    from constants2 import CLOSED_VOLATILITY_SPIKE, CLOSED_MAX_DURATION, SS_VOLATILITY_SPIKE, 
    SS_MAX_DURATION, SS_NO_TODAY
    from utils.strategy_exec.misc import add_tag_to_trades_and_close_position

    def process_volatility_spike(strategy: Strategy) -> bool:
        if strategy.data.tr_delta[-1] < 2.5:
            return False 
        add_tag_to_trades_and_close_position(strategy, CLOSED_VOLATILITY_SPIKE)
        return True 

    def process_max_duration(strategy: Strategy) -> bool:
        max_trade_duration_long = strategy.parameters.max_trade_duration_long
        if max_trade_duration_long is None or not strategy.trades:
            return False
        max_trade_duration = max((strategy.data.index[-1] - trade.entry_time).days for trade in strategy.trades)
        if strategy.trades[-1].is_long and max_trade_duration > max_trade_duration_long:
            add_tag_to_trades_and_close_position(strategy, CLOSED_MAX_DURATION)
            return True
        return False 

    def process_special_situations(strategy: Strategy) -> Tuple[bool, str]: 
        if process_max_duration(strategy):
            return True, SS_MAX_DURATION 
        if process_volatility_spike(strategy):
            return True, SS_VOLATILITY_SPIKE
        return False, SS_NO_TODAY

Explanation

The process_special_situation function is employed to automatically close positions during extreme volatility spikes (true range delta > 2.5) or when trades exceed a maximum holding period of 100 days. This proactive approach strengthens the resilience of the trading strategy, aligning with the firm's long-term vision of maintaining robust performance across varying market conditions.

**STEP 6: BACKTESTING AND OPTIMIZATION**

(Validate and refine the strategy)

    from typing import List 
    import pandas as pd
    import numpy as np
    from customizable.strategy_params import StrategyParams
    from utils.local_data import TickersData
    from strategy.run_backtest_for_ticker import run_backtest_for_ticker

    def run_all_tickers(tickers_data: TickersData, strategy_params: StrategyParams, tickers: list[str]) -> float:
        open("app_run.log", "w", encoding="UTF-8").close()
        performance_res = pd.Dataframe()
        all_trades = pd.DataFrame()
        for ticker in tickers:
            ticker_data = tickers_data.get_data(ticker)
            stat, trades_df, last_day_result = run_backtest_for_ticker(ticker, ticker_data, strategy_params)
            stat = stat.drop(["_strategy", "_equity_curve", "_trades"])
            stat["SQN_MODIFIED"] =stat["SQN"] / np.sqrt(stat["# Trades"])
            performance_res[ticker] = stat
            if strategy_params.save_all_trades_in_xlsx:
                trades_df["Ticker"] = ticker
                all_trades = pd.concat([all_trades, trades_df])
            if len(tickers) > 1:
                performance_res.to_excel("output.xlsx")
            if strategy_params.save_all_trades_in_xlsx:
                all_trades.to_excel("all_trades.xlsx", index=False)
            return performance_res.loc["SQN_modified", :].mean()

Explanation

Comprehensive backtests are conducted across all tickers in tickers_all using the run_all_tickers function, while key parameters _ such as atr_multiplier_threshold are fine tuned through run_strategy_main_optimize. This rigorous testing process validates the strategy's effectiveness and reflects the firm's commitment to a data-driven, evidence-based investment approach.

**STEP 7: EXECUTION AND MONITORING**

(Deploy the strategy and track performance)

    import logging 
    from dotenv import load_dotenv 
    from constants2 import LOG_FILE, tickers_all
    from customizable.strategy_params import StrategyParams 
    from f_v1_basic import add_features_v1_basic 
    from strategy.all_tickers import run_all_tickers 
    from utils.local_data import TickersData
    import warnings 

    logging.basicConfig(level=logging.DEBUG, format="%(message)s", filename=LOG_FILE, encoding="utf-8", filemode="a")

    if __name__ == "__main__":
        load_dotenv()
        open(LOG_FILE, "w", encoding="UTF-8").close()

        strategy_params = StrategyParams(
            max_trade_duration_long=100,
            max_trade_duration_short=100,
            profit_target_long_pct=29.9,
            profit_target_short_pct=29.9,
            stop_loss_default_atr_multiplier=2.5,
            save_all_trades_in_xlsx= True,
        )

        tickers_data = TickersData (
            add_feature_cols_func=add_features_v1_basic,
            tickers=tickers_all,
        )
  
        SQN_modified_mean = run_all_tickers(
            tickers_data=tickers_data,
            tickers=tickers_all,
            strategy_params=strategy_params,
        )
        logging.debug(f"SQN_modified_mean={SQN_modified_mean}")
        print(f"SQN_modified_mean={SQN_modified_mean}, see output.xlsx") 
    
    warnings.filterwarnings("ignore")

Explanation

The last_day_result function is used to monitor trading signals for real-time execution, with outcomes saved in output.xlsx for ongoing analysis. This continuos monitoring framework supports the firm's commitment to iterative strategy refinement and long-term performance improvement.

The System Quality Number (SQN) is a popular indicator of the trading system's quality. Its classic formula has a drawback: it tends to produce overly optimistic results when analyzing more than 100 trades, particularly when the number of trades exceeds 150-200.

SQN_modified is devoid of this drawback. It is simply the average of trade profits divided by the standard deviation of profits. A trading system is considered not bad if its SQN_modified has a positive value of at least 0.1. Systems whose value exceeds 0.2 are deemed decent or even good.

# ANCHORED VOLUME WEIGHTED AVERAGE PRICES (VWAPS)

This systematic trading strategy utilizes Anchored VWAPs to identify trends, support and resistance levels, and optimal entry and exit points. Primarily designed for daily (1d) swing trading, it can also be adapted for intraday timeframes like 15-minute or 5-minute charts. The strategy focuses on stocks and ETFs, incorporating key indicators such as Anchored VWAPs, the Average True Range (ATR), significant price levels, and a 5-day Simple Moving Average (SMA). Anchored VWAP, a powerful tool in technical analysis, calculates the volume-weighted average price of an asset from a specific anchor point, such as key highs, lows, or market events, offering dynamic support and resistance levels. It is computed as the cumulative sum of the Typical Price (Open + High + Low + Close)/4 multiplied by volume, divided by total volume. This approach helps confirm trends, identify support and resistance zones, and generate trade signals based on price interactions with VWAP levels. The provided code framework efficiently fetches OHLC data, computes Anchored VWAPs, detects significant price levels, and visualizes them, ensuring a systematic and repeatable trading process.

**STEP 1: DATA PREPARATION**

(Fetch Open, High, Low, Close, Volume (OHLCV) data and compute necessary indicators)

    import pandas as pd
    from import_ohlc.yahoo_finance import get_ohlc_from_yf
    from misc.atr import add_atr_col_to_df
    from misc.fill_min_max import fill_is_min_max
    from constants import ATR_SMOOTHING_N

    def prepare_data(ticker: str, period: str= "2y", interval: str = "1d") -> pd.DataFrame:
        try:
            df = get_ohlc_from_yf(ticker=ticker, period=period, interval=interval)
        except Exception as e:
            print(f"Failed to fetch data for {ticker} from Yahoo Finance: {e}")
            print("Alpha Vantage fetch not available as backup")
            return pd.DataFrame()
        
        df = add_atr_col_to_df(df, n=ATR_SMOOTHING_N, exponential=False)

        df = fill_is_min_max(df)

        print(f"Prepared data for {ticker}:")
        print(df[["Open", "High", "Low", "Close", "Volume", f"atr_{ATR_SMOOTHING_N}", "is_min", "is_max"]].tail())
        return df
    
    ticker = "AAPL"
    df = prepare_data(ticker)

    ticker_data = {}
    if not df.empty:
       ticker_data[ticker] = df 

Explanation

The code fetches two years of daily OHLCV data for the ticker using the get_ohlc_from_yf function, computes a 14-period ATR (shifted to the previous day's value), and identifies significant highs and lows where price movements exceed 2.5 times the ATR. The resulting data, including OHLCV, ATR, and marked min/max levels, is stored in a dictionary of DataFrames for further analysis. Key functions from the codebase, such as add_atr_col_to_df for calculating volatility and fill_is_min_max for detecting significant levels, are used. This data provides the necessary foundation for VWAP-based trading and risk management. The output is a directionary of DataFrames, printed for verification, containing the OHLCV, ATR, and key level indicators.

**STEP 2: ANCHOR POINTS**

(Set anchor dates for VWAP calculations, combining manual and automatic points)

    from constants import first_day_of_year

    def get_anchor_dates(df: pd.DataFrame, custom_dates: list[str] = None) -> list[str]:
        last_min_date = df[df["is_min"]].index.max()
        last_max_date = df[df["is_max"]].index.max()

        anchor_dates = [first_day_of_year]
    
        if pd.notna(last_min_date):
            anchor_dates.append(last_min_date.strftime('%Y-%m-%d %H:%M:%S'))
        if pd.notna(last_max_date):
            anchor_dates.append(last_max_date.strftime('%Y-%m-%d %H:%M:%S'))

        anchor_dates = [date for date in anchor_dates if pd.notna (date)]
        print(f"Anchor dates for {df.attrs.get('ticker', 'unknown')}: {anchor_dates}")
        return anchor_dates
        
    anchor_dates_dict = {}
    anchor_dates_dict[ticker] = get_anchor_dates(ticker_data[ticker])

Explanation

The code creates a list of anchor dates for a specific stock ticker, starting with the fixed baseline date, first_day_of_year. It then adds custom_dates (such as earning reports, price peaks, and market corrections), and incorporates the most recent min/max dates from the stock's price history defined in prepare data function. The anchored dates are stored in a dictionary, providing a set of key reference points for further analysis, like VWAP calculations or trading signals. Custom dates are chosen to reflect significant events, such as earning reports or major price movements, which help us traders make data-driven decisions. The code automates the process of adding these anchor points, ensuring they align with key market events, while offering flexibiity for our strategies.

**STEP 3: TRADING SIGNALS**

(Define systematic rule for trend identification, entries, exits, and risk management)

    import pandas as pd
    from vwaps_plot import vwaps_plot_build_save
    from misc.chart_annotation import get_chart_annotation_1d 
    from constants import ATR_SMOOTHING_N, first_day_of_year

    def analyze_ticker(df: pd.DataFrame, ticker: str, anchor_dates: list[str]):
        df.attrs["ticker"] = ticker
        df = df.copy()
    
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_convert(None)
    
        dates_only = df.index.normalize().date

        for i, anchor_date in enumerate(anchor_dates, 1):
            anchor_ts = pd.Timestamp(anchor_date).date()
            if anchor_ts in dates_only:
                anchor_idx = list(dates_only).index(anchor_ts)
            else:
                if anchor_ts < df.index[0].date():
                    print(f"Info: Anchor date {anchor_date} is before the start of data for {ticker}. Using first available date.")
                else:
                    #print(f"Warning: Anchor date {anchor_date} not found in {ticker} index. Using start of data.")
                    anchor_idx = 0

            df_from_anchor = df.iloc[anchor_idx:]
            typical_price = (df_from_anchor["Open"] + df_from_anchor["High"] +
                             df_from_anchor["Low"] + df_from_anchor["Close"]) / 4
            cumulative_typical_volume = (typical_price * df_from_anchor["Volume"]).cumsum()
            cumulative_volume = df_from_anchor["Volume"].cumsum()
            vwap = cumulative_typical_volume / cumulative_volume.replace(0, pd.NA)
            vwap = vwap.ffill()
            df.loc[df_from_anchor.index, f"A_VWAP_{i}"] = vwap

        vwaps_plot_build_save(
            input_df=df,
            anchor_dates=anchor_dates,
            chart_title=f"{ticker} Daily with Anchored VWAPs",
            chart_annotation_func=get_chart_annotation_1d,
            add_last_min_max=False,
            file_name=f"daily_{ticker}_annotated.png",
            print_df=False  
        )
   
        last_close = df["Close"].iloc[-1]
        vwap_year = df["A_VWAP_1"].iloc[-1]
        vwap_min = df["A_VWAP_2"].iloc[-1]
        vwap_max = df["A_VWAP_3"].iloc[-1]
        atr = df[f"atr_{ATR_SMOOTHING_N}"].iloc[-1]

        trend = "Neutral"
        if last_close > vwap_year and last_close > vwap_min:
            trend = "Bullish"
        elif last_close < vwap_year and last_close < vwap_max:
            trend = "Bearish"
 
        signal = None
        if trend == "Bullish" and last_close > vwap_min and abs(last_close - vwap_min) < atr * 0.5:
            signal = "Long"
        elif trend == "Bearish" and last_close < vwap_max and abs(vwap_max - last_close) < atr * 0.5:
            signal = "Short"

        if signal == "Long":
            entry_price = last_close
            stop_loss = vwap_min - atr
            take_profit = vwap_max
            risk = entry_price - stop_loss
        elif signal == "Short":
            entry_price = last_close
            stop_loss = vwap_max + atr
            take_profit = vwap_min
            risk = stop_loss - entry_price
        else:
            entry_price = stop_loss = take_profit = risk = None

        account_size = 10000
        risk_percent = 0.01
        position_size = (account_size * risk_percent) / risk if risk else 0

        return {
            "trend": trend,
            "signal": signal,
            "last_close": last_close,
            "vwap_year": vwap_year,
            "vwap_min": vwap_min,
            "vwap_max": vwap_max,
            "atr": atr,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "position_size": position_size
        }

    results = {}
    for ticker in ticker_data:
        results[ticker] = analyze_ticker(ticker_data[ticker], ticker, anchor_dates_dict[ticker])

    results_df = pd.DataFrame(results).T
    display(results_df)

Explanation

This script computes and plots Anchored VWAPs for each ticker, then applies a VWAP-based trading strategy. It calculates three anchored VWAP levels—year-start, recent minimum, and recent maximum—and uses them to determine market trend and generate trade signals. A bullish trend is identified when the current close is above both the year-start and last-min VWAPs, while a bearish trend occurs when the close is below both the year-start and last-max VWAPs. Entry signals are triggered when the price is near key VWAPs within 0.5× ATR, and trade exits are defined using ATR-based stop losses and VWAP-based take profits. It outputs a structured dictionary with trade details and saves annotated charts for each ticker, turning the VWAP strategy into a fully operational, risk-managed system.

**STEP 4: EXECUTION AND VISUALIZATION**

(Execute the strategy across the watchlist and visualize results for monitoring)

    from datetime import datetime

    print("\nTrade Summary:")
    for ticker, res in results.items():
        print(f"\nTicker: {ticker} on {datetime.now().date()}")
        print(f"Trend: {res['trend']}")
        print(f"Signal: {res['signal']}")
        print(f"Last Close: {res['last_close']:.2f}")
        print(f"VWAPs - Year: {res['vwap_year']:.2f}, Min: {res['vwap_min']:.2f}, Max: {res['vwap_max']:.2f}")
        print(f"ATR: {res['atr']:.2f}")
        if res["signal"]:
            print(f"Entry: {res['entry_price']:.2f}, Stop Loss: {res['stop_loss']:.2f}, "
                  f"Take Profit: {res['take_profit']:.2f}")
            print(f"Position Size: {res['position_size']:.2f} shares")
            print(f"Executing {res['signal']} trade for {ticker}")
        else:
            print("No trade signal generated.")

Explanation

This step runs the VWAP-based strategy and prints a clear, date-stamped trade summary for each. For every ticker, it displays the detected trend, signal (if any), last closing price, VWAP levels (year, min, and max), and ATR value. If a trade signal is generated, it also shows detailed trade parameters including entry price, stop loss, take profit, and calculated position size. The output provides a concise and actionable snapshot of trade setups, supporting informed execution and ongoing strategy monitoring.

**STEP 5: REFINEMENT**

(Enhance the strategy with intraday tools, relative-strength, and quick charting) 

    from draw_avg import draw_5_days_avg
    from price_volume import draw_profile_of_data
    from vwaps_plot import vwaps_plot_build_save
    from misc.chart_annotation import get_chart_annotation_1d
    from ratio import draw_ratio   
    import matplotlib.pyplot as plt

    for ticker in ticker_data:
        draw_5_days_avg(ticker=ticker, interval="15m")  
        print(f"{ticker}: 5-day SMA image generated")

        draw_profile_of_data(ohlc_df=ticker_data[ticker], ticker=ticker)
        print(f"{ticker}: Price and Volume profile image generated")
    
        intraday_df = get_ohlc_from_yf(ticker=ticker, period="5d", interval="1m")
        intraday_df = add_atr_col_to_df(intraday_df, n=ATR_SMOOTHING_N, exponential=False)
    
        vwaps_plot_build_save(
            input_df=intraday_df,
            anchor_dates=anchor_dates_dict[ticker],
            chart_title=f"{ticker} 1m with Anchored VWAPs",
            chart_annotation_func=get_chart_annotation_1d,
            add_last_min_max=False,
            file_name=f"intraday_{ticker}.png",
            hide_extended_hours=True,
            print_df=False
       )
       print(f"{ticker}: Intraday VWAP image generated")

       print(f"{ticker}: Close-to-Close ratio image generated")
       draw_ratio(ticker_1=ticker, ticker_2="MSFT", cutoff_date="2020-01-01")    
       plt.close()

Explanation

The reason for integrating intraday tools such as SMA, volume profiles, relative strength ratios, and quick charting features is to enrich the daily trading strategy with more granular, real-time insights into market conditions. By incorporating these elements, traders can monitor price movements and trends more effectively within the day, allowing for better short-term decision-making. The use of intraday VWAP and other indicators helps identify key price levels and trends during trading hours, enhancing the strategy’s accuracy.

# CONCLUSION

In conclusion, Amare Capital Managememt (Pty) Ltd has developed a robust and systematic trading framework that leverages Python to implement two distinct yet complementary strategies: the Hammer Reversal with Volatility Filter and the Anchored VWAP approach. The Hammer Reversal strategy effectively identifies bullish reversal opportunities through rigorous analysis, incorporating hammer candlestick patterns, volatility filters, and risk management protocols, validated via comprehensive backtesting across multiple tickers. Meanwhile, the Anchored VWAP strategy enhances trend identification and trade execution by utilizing dynamic support and resistance levels, enriched with intraday tools and relative strength analysis for real-time adaptability. Together, these strategies exemplify Amare Capital Management's commitment to data-driven decision-making, disciplined risk management, and continous refinement, positioning the firm to achieve consistent and sustainable performance in diverse market conditions.

