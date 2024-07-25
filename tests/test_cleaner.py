import numpy as np
import pandas as pd
import pytest

from volatility_estimator.cleaner import adjust_for_split


@pytest.fixture
def dummy_df():
    # Define the start and end time for a single day
    start_time = pd.Timestamp("2024-07-24 00:00:00")
    end_time = pd.Timestamp("2024-07-24 23:59:59")

    # Create a range of timestamps for the day at 1-minute intervals
    timestamps = pd.date_range(start=start_time, end=end_time, freq="min")

    # Assign all ones as the price
    prices = np.ones(len(timestamps))

    return pd.DataFrame({"ts": timestamps, "price": prices})


def test_adjust_for_split_half(dummy_df: pd.DataFrame):
    actual = adjust_for_split(dummy_df, split=2)

    expected = pd.DataFrame(
        {
            "ts": dummy_df["ts"],
            "price": 0.5 * np.ones(len(dummy_df)),
        }
    )

    assert actual.equals(expected)
