from typing import Callable, List 

from import_ohlc.yahoo_finance import get_ohlc_from_yf
from misc.chart_annotation import get_chart_annotation_1d
from vwaps_plot import vwaps_plot_build_save

def draw_daily_chart_ticker(
    ticker: str,
    anchor_dates: list[str],
    get_ohlc_func: Callable = get_ohlc_from_yf,
    chart_annotation_func: Callable = get_chart_annotation_1d,
):
    interval = "1d"
    hist = get_ohlc_func(ticker=ticker, period="max", interval=interval)
    chart_title = {"ticker": ticker, "interval": interval}
    vwaps_plot_build_save(
        input_df=hist,
        anchor_dates=anchor_dates,
        chart_title=str(chart_title),
        chart_annotation_func=chart_annotation_func,
        add_last_min_max=True,
        file_name=f"daily_{ticker}.png",
    )
