import pandas as pd

from constants2 import FEATURE_COL_NAME_ADVANCED, FEATURE_COL_NAME_BASIC
from derivative_columns.atr import add_atr_col_to_df
from derivative_columns.ma import add_moving_average

MOVING_AVERAGE_N = 200
REQUIRED_DERIVATIVE_COLUMNS_F_V1_BASIC = {"atr_14", f"ma_{MOVING_AVERAGE_N}"}

def add_required_cols_for_f_v1_basic(df: pd.DataFrame) -> pd.DataFrame:
    df_columns = df.columns # Add this line to get the columns from df
    internal_df = df.copy() #Add this line to create the internal_df used later

    if f"ma_{MOVING_AVERAGE_N}" not in df_columns: 
        if len(internal_df) >= MOVING_AVERAGE_N:
            internal_df = add_moving_average(df=internal_df, n=MOVING_AVERAGE_N)
        else:
            print(f"Not enough data for {MOVING_AVERAGE_N}-day MA (len={len(internal_df)}), adding empty ma_200 column")
            internal_df[f"ma_{MOVING_AVERAGE_N}"] = pd.Series(index=internal_df.index, dtype=float)

    # Add ATR (atr_14)
    if "atr_14" not in df_columns:
        if "tr" in df_columns:
            if len(internal_df) >= 14:
                internal_df["atr_14"] = internal_df["tr"].rolling(14).mean()
            else:
                print("Not enough data for 14-period ATR, adding empty atr_14 column")
                internal_df["atr_14"] = pd.Series(index=internal_df.index, dtype=float)
        else:
            internal_df = add_atr_col_to_df(df=internal_df, n=14, exponential=False)

    print(f"Shape after adding required columns: {internal_df.shape}")
    print(f"First few rows after adding required columns:\n{internal_df.head() if not internal_df.empty else 'DataFrame is empty'}")
    return internal_df

def add_features_v1_basic(
    df: pd.DataFrame, atr_multiplier_threshold: int = 6
) -> pd.DataFrame:
    """
    First make sure that all necessary derived columns are present.
    After that, add features_v1_basic columns.
    """
    print(f"Initial shape of df in add_features_v1_basic: {df.shape}")
    print(f"First few rows of input df:\n{df.head() if not df.empty else 'DataFrame is empty'}")

    if df.empty:
        raise ValueError("Input DataFrame to add_features_v1_basic is empty")

    res = df.copy()

    for col in REQUIRED_DERIVATIVE_COLUMNS_F_V1_BASIC:
        if col not in res.columns:
            res = add_required_cols_for_f_v1_basic(df=res)

    # Customize below
    # Handle cases where ma_200 or atr_14 might be NaN
    res[FEATURE_COL_NAME_BASIC] = res["Close"] < res[f"ma_{MOVING_AVERAGE_N}"]
    # Replace NaN with False to avoid losing rows
    res[FEATURE_COL_NAME_BASIC] = res[FEATURE_COL_NAME_BASIC].fillna(False)

    res[FEATURE_COL_NAME_ADVANCED] = (res["ma_200"] - res["Close"]) >= (
        res["atr_14"] * atr_multiplier_threshold
    )
    # Replace NaN with False to avoid losing rows
    res[FEATURE_COL_NAME_ADVANCED] = res[FEATURE_COL_NAME_ADVANCED].fillna(False)

    print(f"Shape after adding features: {res.shape}")
    print(f"First few rows after adding features:\n{res.head() if not res.empty else 'DataFrame is empty'}")

    return res
