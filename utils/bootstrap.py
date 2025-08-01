import sys
import numpy as np
import pandas as pd
from scipy.stats import bootstrap

from constants2 import DEFAULT_BOOTSTRAP_CONFIDENCE_LEVEL


def get_bootstrapped_mean_ci(
    data: np.typing.NDArray[np.float64],
    conf_level: float = DEFAULT_BOOTSTRAP_CONFIDENCE_LEVEL,
    calculate_positive_pct: bool = True,
) -> dict:
    """
    Calculate the mean value and determine the left and right boundaries
    of the confidence interval using the bootstrap method.
    """
    data = data[~np.isnan(data)]
    if len(data) <= 3:
        return {
            f"ci_left_{conf_level}": np.nan,
            "mean_val": np.nan,
            f"ci_right_{conf_level}": np.nan,
            "count": data.size,
        }
    mean_ci_left, mean_ci_right = bootstrap(
        (data,),
        np.mean,
        confidence_level=conf_level,
        n_resamples=1000,
        random_state=1,
        method="percentile",
    ).confidence_interval
    if isinstance(mean_ci_left, np.floating):
        mean_ci_left = mean_ci_left.item()
    if isinstance(mean_ci_right, np.floating):
        mean_ci_right = mean_ci_right.item()
    res = {
        f"ci_left_{conf_level}": mean_ci_left,
        "mean_val": np.mean(data),
        f"ci_right_{conf_level}": mean_ci_right,
        "count": data.size,
    }
    if isinstance(res["mean_val"], np.floating):
        res["mean_val"] = res["mean_val"].item()
    if calculate_positive_pct:
        res["positive_pct"] = float(data[np.where(data > 0)].size) / data.size
    for key, value in res.items():
        res[key] = round(value, 3)
    return res


def analyze_values_by_group(
    df: pd.DataFrame,
    group_col_name: str,
    values_col_name: str,
    group_order_map: dict,
    excel_file_name: str = "analyze_values_by_group.xlsx",
) -> None:
    """
    Input: DataFrame containing some values classified by groups.
    For every group, get its values_col_name mean value
    and bootstrapped confidence interval.
    Save all results in Excel file.
    """
    res = dict()
    group_names = df.dropna()[group_col_name].unique()
    print()
    groups_total_count = len(group_names)
    counter = 0
    for group_name in group_names:
        counter = counter + 1
        print(
            f"analyze_values_by_group: running {group_name=}, {counter} of {groups_total_count}...",
            file=sys.stderr,
        )
        res[group_name] = get_bootstrapped_mean_ci(
            data=df[df[group_col_name] == group_name][values_col_name].dropna().values
        )
    print("analyze_values_by_group: running final all_data...", file=sys.stderr)
    res["all_data"] = get_bootstrapped_mean_ci(data=df[values_col_name].dropna().values)
    df = pd.DataFrame(res).T

    # now sort DF rows according to the group_order_map
    for i, _ in df.iterrows():
        df.loc[i, "order"] = group_order_map[df.loc[i].name]
    df = df.sort_values("order")
    del df["order"]

    df.to_excel(excel_file_name)
