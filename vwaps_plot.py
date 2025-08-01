import datetime 
from typing import Callable, List, Optional, Set, Tuple

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from constants import ATR_SMOOTHING_N, DEFAULT_RESULTS_FILE 
from misc.fill_min_max import fill_is_min_max
from misc.chart_annotation import get_chart_annotation_1d 

def _add_last_min_max_dates(
    input_df: pd.DataFrame, anchor_dates: Set[pd.Timestamp]
) -> Tuple[pd.DataFrame, Set[pd.Timestamp]]:
    """
    Add dates of last min and max to the set of dates.
    """
    df = input_df.copy()
    if (
        f"atr_{ATR_SMOOTHING_N}" not in df.columns
        or "is_min" not in df.columns
        or "is_max" not in df.columns
    ):
        df = fill_is_min_max(df=df)
    last_min_date = df[df["is_min"] == True].index.max() 
    last_max_date = df[df["is_max"] == True].index.max()  
    anchor_dates.update({last_min_date, last_max_date})

    # NOTE return not only anchor_dates but also pd.DataFrame,
    # because you may want later to use somewhere
    # the new columns is_min, is_max, atr_{ATR_SMOOTHING_N}
    return df, anchor_dates

def _preprocess_anchor_dates(
    anchor_dates: List[str],
) -> Tuple[Set[pd.Timestamp], Optional[pd.Timestamp]]:
    """
    1. Convert a list of strings to a list of Timestamps.
    2. Search for the x-marked minimum threshold date.
    """

    # NOTE If an element with anchor_date[0] == "x" is found
    # in the anchor_dates list, it is assigned to min_anchor_date.
    # Otherwise, min_anchor_date remains None.
    # There is benefits to this feature.

    min_anchor_date = None
    for anchor_date in anchor_dates:
        if isinstance(anchor_date, datetime.datetime):
            continue
        if anchor_date[0] == "x":
            min_anchor_date = pd.to_datetime(anchor_date[1:])
    anchor_points = [
        (
            anchor_date[1:]
            if not isinstance(anchor_date, datetime.datetime) and anchor_date[0] == "x"
            else anchor_date
        )
        for anchor_date in anchor_dates
    ]
    anchor_points_ts: List[pd.Timestamp] = [
        (
            pd.to_datetime(anchor_date)
            if not isinstance(anchor_date, datetime.datetime)
            else anchor_date
        )
        for anchor_date in anchor_points
    ]
    return set(anchor_points_ts), min_anchor_date


def vwaps_plot_build_save(
    input_df: pd.DataFrame,
    anchor_dates: list[str],
    chart_title: str = "",
    chart_annotation_func: Callable = get_chart_annotation_1d,
    add_last_min_max: bool = False,
    file_name: str = DEFAULT_RESULTS_FILE,
    print_df: bool = True,
    hide_extended_hours: bool = False,
) -> None:
    """
    1. Transform every element of anchor_dates to pd.Timestamp.
    2. Add a new column with a typical price.
    3. For each anchor date, create a column with Anchored VWAP.
    4. Build a candlestick chart with all Anchored VWAPs and save it.

    Add x before the desired date to make the chart start from that date.
    """

    df = input_df.copy()

    # Otherwise, TypeError: Invalid comparison between
    # dtype=datetime64[ns, America/New_York] and Timestamp
    
    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)

    anchor_points, min_threshold_point = _preprocess_anchor_dates(
        anchor_dates=anchor_dates
    )
    if add_last_min_max:
        df, anchor_points = _add_last_min_max_dates(
            input_df=df, anchor_dates=anchor_points
        )
    if min_threshold_point is None:
        min_threshold_point = min(anchor_points)

    if "Typical" not in df.columns:
        df["Typical"] = (df["Open"] + df["High"] + df["Low"] + df["Close"]) / 4
    if "TypicalMultiplyVolume" not in df.columns:
        df["TypicalMultiplyVolume"] = df["Typical"] * df["Volume"]

    # Add anchored VWAP column for every date passed in anchor_points
    counter = 0
    for anchor_dt in anchor_points:
        counter = counter + 1
        df[f"A_VWAP_{counter}"] = (
            df["TypicalMultiplyVolume"]
            .where(df.index >= anchor_dt)
            .groupby(df.index >= anchor_dt)
            .cumsum()
            / df["Volume"]
            .where(df.index >= anchor_dt)
            .groupby(df.index >= anchor_dt)
            .cumsum()
        )

    df = df[df.index >= min_threshold_point]

    if print_df:
        print(df[["Open", "High", "Low", "Close", "Volume", f"atr_{ATR_SMOOTHING_N}"]])
    # df.to_excel("DF_before_plot_VWAP.xlsx")
    del df["TypicalMultiplyVolume"]
    del df["Typical"]

    plot_data = [
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            line=dict(width=1),
        )
    ]
    for counter in range(1, len(anchor_points) + 1):
        plot_data.append(
            go.Scatter(
                x=df.index,
                y=df[f"A_VWAP_{counter}"],
                mode="lines",
            ),
        )
    fig = go.Figure(data=plot_data)
    # fig.update_layout(
    #     margin=dict(l=10, r=10, t=10, b=10),
    # )

    # Add title to the chart if data is available
    # and increase the top margin to fit the title
    fig.update_layout(
        title=chart_title,
        title_x=0.5,
        title_y=0.99,
        margin=dict(l=10, r=10, t=20, b=10),
    )
    fig.add_annotation(
        xref="x domain",
        yref="y domain",
        x=0.01,
        y=0.99,
        text=chart_annotation_func(df=df),
        showarrow=False,
        # row=1,
        # col=1,
    )

    fig.update_xaxes(
        rangeslider_visible=False,
        rangebreaks=[
            dict(bounds=["sat", "mon"]),  # hide weekends, Saturday to before Monday
        ],
    )
    if hide_extended_hours and (input_df.attrs["interval"] != "1d"):
        fig.update_xaxes(
            rangebreaks=[
                dict(
                    # NOTE You may have to adjust these bounds for hours
                    bounds=[21, 13.5],
                    pattern="hour",
                ),  # hide hours outside of trading hours, in my case 21:00-13:30
            ],
        )

    fig.update_layout(showlegend=False)

    # NOTE it requires kaleido package
    
    fig.write_image(file_name)
