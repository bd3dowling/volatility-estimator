import numpy as np
import pandas as pd

from volatility_estimator.config import END_TIME, START_TIME


def clean_price_frame(frame: pd.DataFrame, splits: dict[str, float]) -> pd.DataFrame:
    return (
        frame.pipe(_filter_non_trading_hours)
        .pipe(_filter_zero_prices)
        .pipe(_combine_identical_timestamps)
        .pipe(_remove_outliers)
        .pipe(_adjust_for_all_splits, splits=splits)
        .pipe(_add_date_column)
    )


def adjust_for_split(frame: pd.DataFrame, split: float) -> pd.DataFrame:
    # NOTE: copy to avoid side-effecting
    frame = frame.copy()
    frame["price"] = frame["price"] / split

    return frame


def _adjust_for_all_splits(frame: pd.DataFrame, splits: dict[str, float]) -> pd.DataFrame:
    # NOTE: copy to avoid side-effecting
    frame = frame.copy()

    for split_date_str, split_ratio in splits.items():
        split_date = pd.to_datetime(split_date_str, format="%Y-%m-%d")
        frame.loc[frame["ts"] < split_date, "price"] /= split_ratio

    return frame


def _filter_non_trading_hours(frame: pd.DataFrame) -> pd.DataFrame:
    # P1 of Barndorff-Nielsen (2008)
    return frame.loc[lambda row: row["ts"].dt.time >= START_TIME].loc[
        lambda row: row["ts"].dt.time <= END_TIME
    ]


def _filter_zero_prices(frame: pd.DataFrame) -> pd.DataFrame:
    # P2 of Barndorff-Nielsen (2008)
    return frame.loc[lambda row: row["price"] > 0]


def _combine_identical_timestamps(frame: pd.DataFrame) -> pd.DataFrame:
    # T3 of Barndorff-Nielsen (2008)
    return frame.groupby("ts", as_index=False).median()


def _remove_outliers(
    frame: pd.DataFrame, window_size: int = 50, threshold: float = 10.0
) -> pd.DataFrame:
    # Q4 of Barndorff-Nielsen (2008)
    prices = frame["price"]

    if window_size % 2 != 0:
        raise ValueError("window_size must be even")

    half_window = window_size // 2

    # Define the median function excluding the middle value
    def _median_exclude_middle(x):
        without_middle = np.concatenate([x[:half_window], x[half_window + 1 :]])
        return np.median(without_middle)

    # Define the MAD function excluding the middle value
    def _mad_exclude_middle(x):
        without_middle = np.concatenate([x[:half_window], x[half_window + 1 :]])
        median = np.median(without_middle)
        return np.mean(np.abs(without_middle - median))

    # Calculate the rolling centered median excluding the current observation
    medians = prices.rolling(window=window_size, center=True, min_periods=1).apply(
        _median_exclude_middle, raw=True
    )

    # Calculate the rolling MAD excluding the current observation
    mads = prices.rolling(window=window_size, center=True, min_periods=1).apply(
        _mad_exclude_middle, raw=True
    )

    # Calculate the deviation from the rolling median
    deviations = np.abs(prices - medians)

    # Identify the outliers
    outliers = deviations > (threshold * mads)

    # Remove the outliers
    cleaned_frame = frame[~outliers]

    return cleaned_frame


def _add_date_column(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["date"] = frame["ts"].dt.date

    return frame
