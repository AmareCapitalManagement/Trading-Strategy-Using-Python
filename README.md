# QUANTITATIVE VALUE AND MOMENTUM FACTOR STRATEGY

Value factor identifies stocks trading below their intrinsic value, using metrics like price-to-earnings (P/E), price-to-book (P/B), price-to-sales (P/S), enterprise value to EBITDA (EV/EBITDA), and enterprise value to gross profit (EV/GP). These metrics evaluate a stock's price relative to earnings, assets, sales, or profitability. Momentum factor focuses on stocks with the highest recent price momentum, assuming that stocks that have been performing well recently will continue to perform well in the short-term. This strategy targets stocks with the lowest valuation metrics, assuming they are undervalued and likely to outperform and exhibit strong, high-quality momentum across varios timeframes (1-month, 3-months, 6-months, and 1-year)

**Step 1: Setting Up the Environment and Importing Stocks**

(Efficient data processing requires libraries for numerical computations, data manipulation, and financial data retrieval)

    import numpy as np
    import pandas as pd
    import yfinance as yf
    import math
    from scipy import stats 
    import warnings
    from statistics import mean 
    from datetime import datetime, timedelta
    import time

    warnings.filterwarnings("ignore")

    stocks = ["ABG.JO", "AEL.JO", "AFT.JO","AGL.JO", "ANG.JO", "APN.JO", "ATT.JO", "BID.JO", "BTI.JO", "BVT.JO", "CFR.JO", "CLS.JO", "CPI.JO", "DSY.JO", "FSR.JO",                   "GRT.JO", "INL.JO", "INP.JO", "ITE.JO", "LBR.JO", "LHC.JO", "MNP.JO", "MRP.JO", "MTN.JO","NED.JO", "NPN.JO", "NTC.JO", "OMU.JO", "PPH.JO", "RDF.JO",                   "REM.JO", "RMH.JO", "RNI.JO", "SAP.JO", "SBK.JO", "SHP.JO", "SLM.JO", "SOL.JO", "SPP.JO", "TBS.JO", "TFG.JO", "TRU.JO", "VOD.JO", "WHL.JO"]
 
**Explanation**

We use numpy for calculations, pandas for data handling, yfinance for Yahoo Finance data, math for share calculations, scipy.stats for percentiles, and xlswriter for Excel output. The warnings library supresses yfinance depreciation warnings.

**Step 2: Fetching Stock Data and Building the DataFrame**

(Valueation metrics (P/E, P/B, P/S, EV/EBIT, EV/EBITDA, EV/GP) are critical for identifying value stocks, and momentum metrics (1-M, 3-M, 6-M, 12-M) for identifying trending stocks. A DataFrame organizes these metrics for analysis, enabling ranking and filtering)

    from IPython.display import display 

    def get_valuation_ratios(ticker):
        stock = yf.Ticker(ticker)
        info = stock.info

        try:
            name = info.get('shortName', 'N/A')
            sector = info.get('sector', 'N/A')
            price_cents = info.get('currentPrice', np.nan)
            price = price_cents / 100 if not np.isnan(price_cents) else np.nan
            price_str = f"R{price:,.2f}" if not np.isnan(price) else "N/A"
        
            pe_ratio = info.get('trailingPE', np.nan)
            pb_ratio = info.get('priceToBook', np.nan)
            if np.isnan(pb_ratio):
                print(f"P/B is missing for {ticker}")
            ps_ratio = info.get('priceToSalesTrailing12Months', np.nan)
            ev = info.get('enterpriseValue', np.nan)
            ebitda = info.get('ebitda', np.nan)
            gross_profit = info.get('grossProfits', np.nan)
            ev_to_ebitda = ev / ebitda if ev and ebitda else np.nan
            ev_to_gp = ev / gross_profit if ev and gross_profit else np.nan

            return {
                'Ticker': ticker,
                'Name': name,
                'Sector': sector,
                'Price': price_str,
                'P/E': pe_ratio,
                'P/B': pb_ratio,
                'P/S': ps_ratio,
                'EV/EBITDA': ev_to_ebitda,
                'EV/GP': ev_to_gp
            }

        except Exception as e:
            print(f"Error fetching ratios for {ticker}: {e}")
            return {
                'Ticker': ticker,
                'Name': 'N/A',
                'Sector':'N/A',
                'Price': 'N/A',
                'P/E': np.nan,
                'P/B': np.nan,
                'P/S': np.nan,
                'EV/EBITDA': np.nan,
                'EV/GP': np.nan
            }

    value_data = [get_valuation_ratios(ticker) for ticker in stocks]
    value_df = pd.DataFrame(value_data)

    end_date = datetime.today()
    start_date = end_date - timedelta(days=730)

    price_data = yf.download(stocks, start=start_date, end=end_date)['Close']
    momentum_df = pd.DataFrame(index=stocks)

    lookback_periods = {
        '1M': 21,
        '3M': 63,
        '6M': 126,
        '1Y': 252,
    }

    for label, days in lookback_periods.items():
        returns = price_data.pct_change(periods=days).iloc[-1]
        momentum_df[f"{label} Return"] = returns

    for label in lookback_periods.keys():
        momentum_df[f"{label} Return"] = momentum_df[f"{label} Return"].apply(lambda x: f"{x:.2%}")

    combined_df = pd.merge(value_df, momentum_df, left_on='Ticker', right_index=True)
    pd.set_option('display.width', None)
    pd.set_option('display.max_columns', None)
    display(combined_df)

    combined_df.to_excel("stock_valuation_momentum.xlsx", index=False)

**Explanation**

We use yfinance to fetch stock dat (price, trailing P/E, P/B, P/S, enterprise value, EBIT, EBITDA, gross profit) and calculate EV/EBIT, EV/EBITDA, and EV/GP. A DataFrame is initialized with columns for tickers, prices, shares to buy, metrics, percentiles, and a robust value (RV) score. Error handling ensures robustness if data is missing. Combining data fetching and DataFrame creation streamlines the data collection process. Also, the historical price data will allow us to calculate momentum returns over different time periods.

**Step 3: Handling Missing Data**

(Financial datasets often have missing values due to unavailable metrics. Addressing these ensures a complete dataset for analysis)

    valuation_columns=['P/E', 'P/B', 'P/S','EV/EBITDA', 'EV/GP']

    momentum_columns = ['1M Return','3M Return','6M Return','1Y Return']

    for column in valuation_columns:
        combined_df[column].fillna(combined_df[column].mean(), inplace=True)

    for column in momentum_columns:
        combined_df[column] = combined_df[column].str.rstrip('%').astype(float) / 100
        combined_df[column].fillna(combined_df[column].mean(), inplace=True)
    
    print("Missing values per column:")
    print(combined_df.isnull().sum())

**Explanation**

We replace missing values with the mean of non-misssing values for each metric. This preserves the dataset's size, assuming missing data is not systematically biased. This step remains standalone due to its distinct focus on data cleaning.

**Step 4: Calculating Percentile and RV Score**

(Percentiles normalize valuation metrics and momentum metrics to a 0-1 scale for comparison across stocks. A composite Score, the mean of these percentiles, robustly measures value. We select the stocks by a Score, and reset the index)

    value_metrics = {
        'P/E': 'PE Percentile',
        'P/B': 'PB Percentile',
        'P/S': 'PS Percentile',
        'EV/EBITDA': 'EV/EBITDA Percentile',
        'EV/GP': 'EV/GP Percentile'
    }

    momentum_metrics = {
        '1M Return': '1M Percentile',
        '3M Return': '3M Percentile',
        '6M Return': '6M Percentile',
        '1Y Return': '1Y Percentile'
    }

    for row in combined_df.index:
        for metric, pct_col in value_metrics.items():
            combined_df.loc[row, pct_col] = stats.percentileofscore(
                combined_df[metric], combined_df.loc[row, metric]
            ) / 100

    for row in combined_df.index:
        for metric, pct_col in momentum_metrics.items():
            combined_df.loc[row, pct_col] = stats.percentileofscore(
                combined_df[metric], combined_df.loc[row, metric]
            ) / 100

    combined_df['Value Score'] = combined_df[list(value_metrics.values())].mean(axis=1)
    combined_df['Momentum Score'] = combined_df[list(momentum_metrics.values())].mean(axis=1)

    value_buy_thresh = combined_df['Value Score'].quantile(0.2)
    value_sell_thresh = combined_df['Value Score'].quantile(0.8)
    momentum_buy_thresh = combined_df['Momentum Score'].quantile(0.8)
    momentum_sell_thresh = combined_df['Momentum Score'].quantile(0.2)

    combined_df['Value Signal'] = combined_df['Value Score'].apply(
        lambda x: 'BUY' if x <= value_buy_thresh else ('SELL' if x >= value_sell_thresh else 'HOLD')
    )

    combined_df['Momentum Signal'] = combined_df['Momentum Score'].apply(
        lambda x: 'BUY' if x >= momentum_buy_thresh else ('SELL' if x <= momentum_sell_thresh else 'HOLD')
    )

    final_df = combined_df[['Ticker', 'Value Score', 'Value Signal', 'Momentum Score', 'Momentum Signal']]
    final_df.sort_values(by='Ticker', inplace=True)
    final_df.reset_index(drop=True, inplace=True)

    pd.set_option('display.max_rows', None)
    print(final_df)                                                      
                                                                                        
**Explanation**

We compute percentile ranks for each metric using scipy.stats.percentileofscore. The RV Score is the average of the valuation metrics, with lower scores indicating better value, and the average of the momentum metrics, with higher scores indicating higher quality momentum. We sort by RV Score, select the top stocks, and reset the index. Combining these steps aligns the analytical focus on ranking and filtering.

The quantitative value and momentum factor strategy is designed to identify and capitalize on two distinct market inefficiencies: undervaluation (via the value factor) and price trends (via the momentum factor). The screener aims to buy low (cheap) value stocks, sell high (expensive) value stocks, buy high positive momentum stocks, and sell low negative momentum stocks. 

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

**Step 8: Bearish Signal Detection Using Shooting Star and 200-Day Moving Average**

(We seek to identify potential bearish setups by detecting the reversal Shooting Star candlestick pattern combined with the price trading above the 200-day moving average (MA200). The purpose of this step is not to generate short-selling trades but rather flag stocks that should be avoided for buying after identifying bullish hammer setups.)

    import pandas as pd
    import yfinance as yf
    from typing import List
    from datetime import datetime
    import logging
    from dotenv import load_dotenv
    from contextlib import redirect_stdout
    import os

    from f_v1_basic import add_features_v1_basic
    from derivative_columns.shooting_star import add_col_is_shooting_star
    from derivative_columns.atr import add_tr_delta_col_to_ohlc

    LOG_FILE = "app_run.log"
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(message)s",
        filename=LOG_FILE,
        encoding="utf-8",
        filemode="a",
     )

    def fetch_ohlc_yfinance(ticker: str, start_date: str = "2020-01-01", end_date: str = "2025-04-06") -> pd.DataFrame:

        try:
            df = yf.Ticker(ticker).history(start=start_date, end=end_date, interval="1d")
            if df.empty:
                logging.error(f"No data fetched for {ticker}")
                return pd.DataFrame()
        
            df = df[["Open", "High", "Low", "Close", "Volume"]]
            df.index = pd.to_datetime(df.index).tz_localize(None)  # Remove timezone
            logging.debug(f"Fetched {len(df)} rows for {ticker}")
            return df
        except Exception as e:
            logging.error(f"Error fetching data for {ticker}: {str(e)}")
            return pd.DataFrame()

    def generate_bearish_signals(tickers: List[str], start_date: str = "2020-01-01", end_date: str = "2025-04-06") -> pd.DataFrame:
   
        results = []
    
        with open(os.devnull, 'w') as devnull:
            with redirect_stdout(devnull):
                for ticker in tickers:
                    logging.debug(f"Processing ticker: {ticker}")
           
                    df = fetch_ohlc_yfinance(ticker, start_date, end_date)
                    if df.empty:
                        logging.warning(f"Skipping {ticker} due to empty data")
                        continue
                
                    df = add_features_v1_basic(df)
                
                    df = add_col_is_shooting_star(df)
                    df = add_tr_delta_col_to_ohlc(df)
                
                    logging.debug(f"Data shape for {ticker}: {df.shape}")
                    logging.debug(f"Columns for {ticker}: {list(df.columns)}")
                
                    nan_counts = df[['Close', 'ma_200', 'atr_14', 'tr_delta', 'is_shooting_star']].isna().sum()
                    logging.debug(f"NaN counts for {ticker}:\n{nan_counts}")
                
                    df["Bearish_Signal"] = (
                        (df["is_shooting_star"] == True) &  
                        (df["Close"] > df["ma_200"]) &    
                        (df["tr_delta"] < 3.0)             
                    )
                
                    shooting_star_count = df["is_shooting_star"].sum()
                    uptrend_count = (df["Close"] > df["ma_200"]).sum()
                    volatility_count = (df["tr_delta"] < 3.0).sum()
                    signal_count = df["Bearish_Signal"].sum()
                    logging.debug(f"Shooting star count for {ticker}: {shooting_star_count}")
                    logging.debug(f"Uptrend count (Close > ma_200) for {ticker}: {uptrend_count}")
                    logging.debug(f"Volatility count (tr_delta < 3.0) for {ticker}: {volatility_count}")
                    logging.debug(f"Bearish signal count for {ticker}: {signal_count}")
                
                    df.to_excel(f"debug_{ticker}_full_data.xlsx")
                    logging.debug(f"Saved full data for {ticker} to debug_{ticker}_full_data.xlsx")
                
                    df_output = df[["Close", "ma_200", "atr_14", "tr_delta", "is_shooting_star", Bearish_Signal"]].copy()
                    df_output["Ticker"] = ticker
                    df_output["Date"] = df_output.index
                    df_output["Distance_to_MA200"] = ((df["Close"] - df["ma_200"]) / df["atr_14"]).round(2)
                    df_output = df_output[["Ticker", "Date", "Close", "ma_200", "atr_14", "tr_delta", "Distance_to_MA200", "is_shooting_star", "Bearish_Signal"]]
                
                    results.append(df_output[df_output["Bearish_Signal"] == True])
    
        result_df = pd.concat(results) if results else pd.DataFrame(
            columns=["Ticker", "Date", "Close", "ma_200", "atr_14", "tr_delta", "Distance_to_MA200", "is_shooting_star", "Bearish_Signal"]
        )
    
        result_df[["Close", "ma_200", "atr_14", "tr_delta"]] = result_df[["Close", "ma_200", "atr_14", "tr_delta"]].round(2)

        output_file = "bearish_signals.xlsx"
        result_df.to_excel(output_file, index=False)
        logging.debug(f"Bearish signals saved to {output_file}")
    
        return result_df

    if __name__ == "__main__":
        load_dotenv()
    
        open(LOG_FILE, "w", encoding="utf-8").close()
    
        custom_tickers = ["SPY", "QQQ", "AAPL", "TSLA"]
    
        bearish_signals_df = generate_bearish_signals(tickers=custom_tickers)

        pd.set_option('display.width', 1000)
        pd.set_option('display.max_columns', None)

        print(f"Generated bearish signals for {len(custom_tickers)} tickers.")
        print(f"Total signals: {len(bearish_signals_df)}")
        print(f"Results saved to bearish_signals.xlsx")
        if not bearish_signals_df.empty:
            print("\nSample of bearish signals:")
            print(bearish_signals_df.head())


Explanation

The objective of this step is to generate bearish signals by detecting potential reversal points using the shooting star candlestick pattern in an uptrend, filtered by trend and volatility conditions. The shooting star pattern is identified when is_shooting_star == True, with an uptrend filter where the price is above the 200-day moving average (Close > ma_200). The volatility filter ensures that the True Range delta (tr_delta) is below 2.5 for moderate volatility. Additionally, an optional condition checks if the price is significantly above the 200-day MA ((Close - ma_200) >= (atr_14 * 6)). The output is saved in a DataFrame (bearish_signals.xlsx) with columns for Ticker, Date, Close, MA200, ATR_14, TR_Delta, Is_Shooting_Star, and Bearish_Signal, helping traders identify stocks to avoid buying at potential peaks or consider for short positions. This approach focuses on real-time monitoring without backtesting.

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
            return pd.DataFrame()
        
        df = add_atr_col_to_df(df, n=ATR_SMOOTHING_N, exponential=False)

        df = fill_is_min_max(df)

        print(f"Prepared data for {ticker}:")
        print(df[["Open", "High", "Low", "Close", "Volume", f"atr_{ATR_SMOOTHING_N}", "is_min", "is_max"]].tail())
        return df
    
    ticker_data = {}
    tickers = ["ABG.JO", "AEL.JO", "AFT.JO","AGL.JO", "ANG.JO", "APN.JO", "ATT.JO", "BID.JO", "BTI.JO", "BVT.JO", "CFR.JO", "CLS.JO", "CPI.JO", "DSY.JO", 
               "FSR.JO", "GRT.JO", "INL.JO", "INP.JO", "ITE.JO", "LBR.JO", "LHC.JO", "MNP.JO", "MRP.JO", "MTN.JO","NED.JO", "NPN.JO", "NTC.JO", "OMU.JO", "PPH.JO", 
               "RDF.JO", "REM.JO", "RMH.JO", "RNI.JO", "SAP.JO", "SBK.JO", "SHP.JO", "SLM.JO", "SOL.JO", "SPP.JO", "TBS.JO", "TFG.JO", "TRU.JO", "VOD.JO", "WHL.JO"]

    for ticker in tickers: 
        df = prepare_data(ticker)
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

    anchor_dates_dict = {
        ticker: get_anchor_dates(df.assign(attrs={"ticker": ticker}))
        for ticker, df in ticker_data.items()
    }

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
    import warnings 

    for ticker in ticker_data:
        draw_5_days_avg(ticker=ticker, interval="15m")  
        print(f"{ticker}: 5-day SMA image generated")

        draw_profile_of_data(ohlc_df=ticker_data[ticker], ticker=ticker)
        print(f"{ticker}: Price and Volume profile image generated")

    GENERATE_INTRADAY_VWAP = False # We can toggle this to true if we want to re-enable it 
    
    intraday_df = get_ohlc_from_yf(ticker=ticker, period="5d", interval="1m")
    intraday_df = add_atr_col_to_df(intraday_df, n=ATR_SMOOTHING_N, exponential=False)

    if GENERATE_INTRADAY_VWAP:
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
        plt.close()
    
Explanation

The reason for integrating intraday tools such as SMA, volume profiles, relative strength ratios, and quick charting features is to enrich the daily trading strategy with more granular, real-time insights into market conditions. By incorporating these elements, traders can monitor price movements and trends more effectively within the day, allowing for better short-term decision-making. The use of intraday VWAP and other indicators helps identify key price levels and trends during trading hours, enhancing the strategy’s accuracy.

# EWMAC TREND-FOLLOWING STRATEGY 

The EWMAC (Exponentially Weighted Moving Average Crossover) strategy is a trend-following trading rule that captures momentum in asset prices using only price data. It compares fast and slow EWMA of the price to detect trends: when the fast EWMA is above the slow EWMA, it signals an uptrend (go long), and when the fast EWMA is below the slow EWMA, it signals a downtrend (go short). The raw signal is adjusted for volatility and scaled to reflect forecast strength, then capped to limit extremes. This simple systematic approach is backed by both empirical performance and behavioral finance theory, making it a robust and explainable trading strategy.

    import yfinance as yf
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from datetime import datetime

    tickers = ["ABG.JO", "AEL.JO", "AFT.JO", "AGL.JO", "ANG.JO", "APN.JO", "ATT.JO", "BID.JO", "BTI.JO", "BVT.JO", "CFR.JO", "CLS.JO", "CPI.JO", "DSY.JO", "FSR.JO", "GRT.JO",
               "INL.JO", "INP.JO", "ITE.JO", "LBR.JO", "LHC.JO", "MNP.JO", "MRP.JO", "MTN.JO", "NED.JO", "NPN.JO", "NTC.JO", "OMU.JO", "PPH.JO", "RDF.JO", "REM.JO", "RMH.JO",
               "RNI.JO", "SAP.JO", "SBK.JO", "SHP.JO", "SLM.JO", "SOL.JO", "SPP.JO", "TBS.JO","TFG.JO", "TRU.JO", "VOD.JO", "WHL.JO"]

    start_date = "2024-01-01"
    end_date = "2025-06-11"
    Lfast = 16
    Lslow = 4 * Lfast
    vol_lookback = 25
    capmin = -20
    capmax = 20

    def ewmac_forecast_scalar(Lfast, Lslow):
        return 10 / np.sqrt(Lfast)

    f_scalar = ewmac_forecast_scalar(Lfast, Lslow)

    data = yf.download(tickers, start=start_date, end=end_date)

    for ticker in tickers:
        try:
            price = data["Close"][ticker].dropna()

            if price.empty:
               print(f"No data for {ticker}. Skipping...")
               continue
            
            fast_ewma = price.ewm(span=Lfast).mean()
            slow_ewma = price.ewm(span=Lslow).mean()
            raw_ewmac = fast_ewma - slow_ewma

            returns = price.pct_change()
            vol = returns.ewm(span=vol_lookback).std()
            vol_adj_ewmac = raw_ewmac / vol

            forecast = vol_adj_ewmac * f_scalar
            cap_forecast = forecast.clip(lower=capmin, upper=capmax)

            fig, axs = plt.subplots(1, 2, figsize=(18, 6))

            axs[0].plot(price, label='Price', color='black')
            axs[0].plot(fast_ewma, label=f'Fast EWMA ({Lfast})', linestyle='--')
            axs[0].plot(slow_ewma, label=f'Slow EWMA ({Lslow})', linestyle='--')
            axs[0].set_title(f"EWMAC Crossover\n{ticker}")
            axs[0].set_xlabel("Date")
            axs[0].set_ylabel("Price")
            axs[0].legend()
            axs[0].grid(True)
        
            axs[1].plot(cap_forecast, label='Capped Forecast Signal', color='blue')
            axs[1].axhline(10, color='green', linestyle='--', label='Buy Threshold')
            axs[1].axhline(-10, color='red', linestyle='--', label='Sell Threshold')
            axs[1].set_title("Capped EWMAC Forecast Signal")
            axs[1].set_xlabel("Date")
            axs[1].set_ylabel("Forecast Value")
            axs[1].legend()
            axs[1].grid(True)

            plt.tight_layout()
            plt.savefig(f"{ticker}_ewmac_combined.png")
            plt.close()

            print(f" Saved: {ticker}_ewmac_combined.png")

        except Exception as e:
            print(f" Error with {ticker}: {e}")
   
**Explanation**

The Exponentially Weighted Moving Average Crossover (EWMAC) strategy is a robust and intuitive trend-following trading rule that captures medium- to long-term momentum in asset prices. By comparing a fast-moving average to a slow-moving average, the strategy identifies directional trends: it generates buy signals when prices are trending upward (fast MA > slow MA) and sell signals during downtrends (fast MA < slow MA).

The result is a dynamic signal that is responsive to trends, adaptive to volatility, and simple to implement, making it an ideal component of a systematic trading strategy. Its strength lies not only in its performance but also in its behavioral justification, simplicity, and positive skewness — offering large potential gains during strong market trends while limiting losses in range-bound periods.

# DISCOUNTED CASH FLOW (DCF) MODEL

The DCF model estimates a company's intrinsic value by projecting its future cash flows and discounting them to the present value using the Weighted Average Cost of Capital (WACC). The model assumes that a company's value is the sum of its future Free Cash Flow to Firm (FCFF), adjusted for the time value of money and risk.

To estimate the intrinsic value of a company, several componets and financial metrics are required. FCFF represents the cash flow available to all capital providers - both equity and debt - after accounting for operating expenses, taxes, capital expenditures, and changes in working capital. The Weighted Average Cost of Capital (WACC) is used as the discount rate in valuation, incorporating the cost of equity and cost of debt, each weighted according to the company's capital structure, to reflect its risk profile.

Another critical metric is Return on Invested Capital (ROIC), which measures how efficiently a company generayes returns on the capital invested in its business determining its quality. Comparing ROIC to WACC helps assess whether the firm is creating or destroying value. The intrinsic value is determined by discounted projected FCFFs and a terminal value (which captures the value beyond the explicit forecast period using a perpertual growth rate), then subtracting net debt and dividing the results by the number of shares outstanding to obtaing a fair value per share.

The necessary data to perform these calculations is sourced from Yahoo Finance and includes financial line items such as operating income (EBIT), taxes payable, depreciation and amortization, capital expenditures, changes in non-cash working capital, interest expense, total debt, income before tax, market capitalization, number of shares outstanding, cash balances, and beta. In addition to these, certain assumptions must be made - such as the risk-free rate, expected market return, short-term FCCF growth rate, and perpetual growth rate - as they are not directly available from financial databases.

With this data, several calculations are performed: determining the cost of debt and cost of equity (using the Capital Asset Pricing Model or CAPM), computing WACC, calculating ROIC, forecasting future FCFFs, and estimating the terminal value. These inputs are then used to arrive at the fair value per share, helping us assess whether a stock is undervalued or overvalued.

    import pandas as pd
    import numpy as np
    import yfinance as yf
    import matplotlib.pyplot as plt
    import seaborn as sns 
 
    def calculate_dcf(ticker, growth_rate=0 perpetual_growth_rate=0.02, risk_free_rate=0.04, market_return=0.10, forecast_years=4):
        """
        Calculate the intrinsic value per share using a DCF model. 

        Parameters:
            ticker (str): Company ticker symbol 
            growth_rate (float): Annual growth rate for FCFF projections
            perpetual_growth_rate (float): Growth rate for terminal value
            risk_free_rate (float): Risk-free rate
            market_return (float): Expected market return
            forecast_years (int): Number of years for explicit forecast

        Returns:
            dict: Results including FCFF, WACC, ROIC, and fair value per share.

        """

        try:
            company = yf.Ticker(ticker)
            financials = company.financials
            balance_sheet = company.balance_sheet
            cashflow = company.cashflow
            info = company.info

            ebit = financials.loc['EBIT'].iloc[0] if 'EBIT' in financials.index else 0
            interest_expense = financials.loc['Interest Expense'].iloc[0] if 'Interest Expense' in financials.index else 0
            income_before_tax = financials.loc['Pretax Income'].iloc[0] if 'Pretax Income' in financials.index else 0
            taxes = financials.loc['Tax Provision'].iloc[0] if 'Tax Provision' in financials.index else 0

            total_debt = balance_sheet.loc['Total Debt'].iloc[0] if 'Total Debt' in balance_sheet.index else 0
            cash_equivalents = balance_sheet.loc['Cash And Cash Equivalents'].iloc[0] if 'Cash And Cash Equivalents' in balance_sheet.index else 0
            current_assets = balance_sheet.loc['Current Assets'].iloc[0] if 'Current Assets' in balance_sheet.index else 0
            current_liabilities = balance_sheet.loc['Current Liabilities'].iloc[0] if 'Current Liabilities' in balance_sheetindex else 0
            net_ppe = balance_sheet.loc['Net PPE'].iloc[0] if 'Net PPE' in balance_sheet.index else 0

            depreciation = cashflow.loc['Depreciation And Amortization'].iloc[0] if 'Depreciation And Amortization' in cashflow.index else 0
            capex = cashflow.loc['Capital Expenditure'].iloc[0] if 'Capital Expenditure' in cashflow.index else 0
            working_capital_change = cashflow.loc['Change In Working Capital'].iloc[0] if 'Change In Working Capital' in cashflow.index else 0

            beta = info.get('beta', 1.0)
            market_cap = info.get('marketCap', 0)
            shares_outstanding = info.get('sharesOutstanding', 1)

            fcff = ebit - taxes + depreciation - capex - working_capital_change

            effective_tax_rate = taxes / income_before_tax if income_before_tax != 0 else 0.25 
            cost_of_debt = (interest_expense / total_debt) * (1- effective_tax_rate) if total_debt != 0 else 0

            cost_of_equity = risk_free_rate + beta * (market_return - risk_free_rate)

            total_weight = total_debt + market_cap 
            weight_debt = total_debt / total_weight if total_weight != 0 else 0
            weight_equity = market_cap / total_weight if total_weight != 0 else 1
            wacc = (weight_equity * cost_of_equity) + (weight_debt * cost_of_debt)

            invested_capital = current_assets - current_liabilities + net_ppe 
            roic = (ebit * (1 - effective_tax_rate)) / invested_capital if invested_capital != 0 else 0 

            future_fcff = [fcff * (1 + growth_rate) ** t for t in range(1, forecast_years + 1)]

            last_fcff = future_fcff[-1] if future_fcff else fcff
            terminal_value = (last_fcff * (1 + perpetual_growth_rate)) / (wacc - perpetual_growth_rate) if wacc > perpetual_growth_rate else 0

            pv_fcff = [fcff / (1 + wacc) ** t for t, fcff in enumerate(future_fcff, 1)]
            pv_terminal = terminal_value / (1 + wacc) ** forecast_years if terminal_value != 0 else 0

            total_pv = sum(pv_fcff) + pv_terminal
            market_equity_value = total_pv + cash_equivalents - total_debt
            fair_value_per_share = market_equity_value / shares_outstanding if shares_outstanding != 0 else 0

            excess_returns = roic - wacc if roic != 0 and wacc != 0 else 0 

            return {
                'Ticker': ticker,
                'FCFF': fcff,
                'WACC': wacc,
                'ROIC': roic,
                'Excess Returns': excess_returns,
                'Future FCFF': future_fcff,
                'PV of FCFF': pv_fcff,
                'Terminal Value': terminal_value,
                'PV of Terminal Value': pv_terminal,
                'Market Equity Value': market_equity_value,
                'Fair Value Per Share': fair_value_per_share,
                'Invested Capital': invested_capital
            }

        except Exception as e:
            print(f"Error fetching data or calculating DCF for {ticker}: {e}")
            return None 

    def plot_dcf_charts(results, ticker):
        if not results:
            print("No results to plot.")
            return

        fig, axs = plt.subplots(2, 2, figsize=(14, 10))  
    
        years = ['TTM'] + [f'FY{2025 + i}' for i in range(1, 5)]
        fcff_values = [results['FCFF']] + results['Future FCFF']
        axs[0, 0].plot(years, fcff_values, marker='o', color='blue')
        axs[0, 0].set_title(f'{ticker} FCFF Projections')
        axs[0, 0].set_xlabel('Year')
        axs[0, 0].set_ylabel('FCFF (R)')
        axs[0, 0].grid(True)

        pv_fcff_sum = sum(results['PV of FCFF'])
        pv_terminal = results['PV of Terminal Value']
        axs[0, 1].pie([pv_fcff_sum, pv_terminal],
                  labels=['PV of FCFF', 'PV of Terminal Value'],
                  autopct='%1.1f%%',
                  colors=['lightgreen', 'lightgray'])
        axs[0, 1].set_title(f'{ticker} Intrinsic Value Components')

        axs[1, 0].bar(['WACC', 'ROIC'],
                  [results['WACC'] * 100, results['ROIC'] * 100],
                  color=['orange', 'purple'])
        axs[1, 0].set_title(f'{ticker} WACC vs ROIC')
        axs[1, 0].set_ylabel('Percentage (%)')
        axs[1, 0].grid(True, axis='y')

        growth_rates = np.linspace(max(0.05, results['WACC'] - 0.02), 0.15, 5)
        wacc_rates = np.linspace(max(0.03, results['WACC'] - 0.02), results['WACC'] + 0.02, 5)
        fair_values = np.zeros((len(growth_rates), len(wacc_rates)))
        for i, g in enumerate(growth_rates):
            for j, w in enumerate(wacc_rates):
                temp_results = calculate_dcf(ticker,
                                             growth_rate=g,
                                             perpetual_growth_rate=min(g, w - 0.01),
                                             risk_free_rate=0.04,
                                            market_return=0.10)
                fair_values[i, j] = temp_results['Fair Value Per Share'] if temp_results else 0

        sns.heatmap(fair_values,
                   xticklabels=[f"{x*100:.1f}%" for x in wacc_rates],
                   yticklabels=[f"{x*100:.1f}%" for x in growth_rates],
                annot=True,
                fmt=".2f",
                cmap="YlGnBu",
                ax=axs[1, 1])
        axs[1, 1].set_title(f'{ticker} Sensitivity: Fair Value (R)')
        axs[1, 1].set_xlabel('WACC (%)')
        axs[1, 1].set_ylabel('Growth Rate (%)')

        plt.tight_layout()
        plt.show()
    
    ticker = "MSFT"
    results = calculate_dcf(ticker)

    if results:
        print(f"DCF Analysis for {results['Ticker']}:")
        print(f"FCFF: R{results['FCFF']:,.2f}")
        print(f"WACC: {results['WACC']*100:.2f}%")
        print(f"ROIC: {results['ROIC']*100:.2f}%")
        print(f"Excess Returns: {results['Excess Returns']*100:.2f}%")
        print(f"Future FCFF (2026-2030): {[f'T{x:,.2f}' for x in results['Future FCFF']]}")
        print(f"PV of FCFF: {[f'R{x:,.2f}' for x in results['PV of FCFF']]}")
        print(f"Terminal Value: R{results['Terminal Value']:,.2f}")
        print(f"PV of Terminal Value: R{results['PV of Terminal Value']:,.2f}")
        print(f"Market Equity Value: R{results['Market Equity Value']:,.2f}")
        print(f"Fair Value Per Share: R{results['Fair Value Per Share']:,.2f}")

        plot_dcf_charts(results, ticker)
    else:
        print("Failed to compute DCF. Check ticker or data availability")

**Explanation**

This value momentum investing is an innovative investment strategy that merges the foundational principles of value and investing with the dynamics aspects of momentum investing. At its core, value investing seeks to identify stocks that are undervalued relative to their intrinsic worth. Momentum investing, on the other hand, leverages the tendency of stocks to continue moving in the same direction-upward or downward based on recent price movements. This hybrid approach aims to invest in undervalued stocks that are currently experiencing increasing demand, thereby maximizing potential returns while minimizing risk. By harnessing the strengths of both strategies, Amare Capital Management (Pty) Ltd can capitalize on market inefficiencies and identify opportunities that others may overlook.

<p align="center">
  <img src="https://github.com/user-attachments/assets/10d1c94c-d6f6-43ca-8d8b-e8dfb985d25f" alt="ACM w colour" width="780" height="253">
</p>

