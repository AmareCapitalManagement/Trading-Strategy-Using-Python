
# Module
import inspect
from typing import Callable

import pandas as pd


def ensure_df_has_all_required_columns(
    df: pd.DataFrame, volume_col_required: bool = False
) -> None:
    if volume_col_required:
        required_columns = ["Open", "High", "Low", "Close", "Volume"]
    else:
        required_columns = ["Open", "High", "Low", "Close"]
    df_columns = df.columns
    for col_name in required_columns:
        if col_name not in df_columns:
            caller = inspect.stack()[1].function
            raise ValueError(
                f"OHLC DataFrame must contain columns: {required_columns}, absent column {col_name}, checked by {caller} functin"
            )


def add_z_score_col_to_df(
    df: pd.DataFrame, col_name: str, window: int = 100
) -> pd.DataFrame:
    df_internal = df.copy(deep=True)
    col_mean = df[col_name].rolling(window=window, min_periods=window).mean()
    col_std = df[col_name].rolling(window=window, min_periods=window).std()
    df_internal[f"{col_name}_z_sc"] = (df[col_name] - col_mean) / colstd
    return df_internal


def add_feature_group_col_to_df(
    df: pd.DataFrame,
    continuous_feature_col_name: str,
    new_col_name: str,
    get_label_for_group: Callable,
) -> pd.DataFrame:
    """
    Break a conti.nuous feature into discretionary groups
    in order to then conduct an analysis of these groups.
    Input: DataFrame with some feature in column continuous_feature_col_name.
    For e eature value, its label is determined
    using the  's custom function get_group_label_XXX
    and saved in a new column new_col_name.
    """

    # NOTE get_label_for_group example - see functions
    # get_group_label_forecast_bb and get_group_label_tr_delta

    res = df.copy()
    res[new_col_name] = res[continuous_feature_ol_name].apply(get_label_for_group)
    return res


def get_forecast_bb(df: pd.DataFrame) -> pd.Series:
    return df["forecast_bb"]
