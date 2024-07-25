from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from volatility_estimator.cleaner import clean_price_frame
from volatility_estimator.config import CLEAN_PRICE_PATH, HIST_VOL_PATH
from volatility_estimator.estimator import VolatilityEstimatorName, get_estimator

# pd.bdate_range(end=pd.to_datetime("2017-05-30", format="%Y-%m-%d"), periods=30).date


def load_price_frame(file_path: Path) -> pd.DataFrame:
    return pd.read_csv(
        file_path,
        dtype={"price": "float64"},
        parse_dates=["ts"],
    )


def load_clean_comine_price_frames(
    stock_file_paths: Iterator[Path],
    stock_splits: dict[str, float],
) -> pd.DataFrame:
    price_frames: list[pd.DataFrame] = []

    for file_path in stock_file_paths:
        price_frame = load_price_frame(file_path)

        if price_frame.empty:
            print(file_path)
            continue

        cleaned_price_frame = clean_price_frame(price_frame, splits=stock_splits)

        price_frames.append(cleaned_price_frame)

    return pd.concat(price_frames, ignore_index=True)


def base_process_prices(
    stock: str,
    stock_file_paths: Iterator[Path],
    stock_splits: dict[str, float],
) -> None:
    # Base processing and cleaning of stock files
    stock_frame = load_clean_comine_price_frames(stock_file_paths, stock_splits)

    # Store as partitioned parquets
    stock_frame.to_parquet(
        f"{CLEAN_PRICE_PATH / stock}.parquet", index=False, partition_cols=["date"]
    )


def incremental_process_prices(
    stock: str,
    date: str,
    file_path: Path,
    stock_splits: dict[str, float],
) -> None:
    price_frame = load_price_frame(file_path)

    if price_frame.empty:
        print(file_path)
        return

    cleaned_price_frame = clean_price_frame(price_frame, stock_splits)

    if not stock_splits.get(date):
        # Store as partitioned parquets
        cleaned_price_frame.to_parquet(
            f"{CLEAN_PRICE_PATH / stock}.parquet", index=False, partition_cols=["date"]
        )

        return

    # need to reprocess old price data...


def base_compute_volatility(
    stock: str,
    estimator_method: VolatilityEstimatorName,
    lookback_window: int,
    num_trading_days: int,
    other_estimator_kwargs: Any,
) -> None:
    # Instantiate estimator
    estimator = get_estimator(
        estimator_method,
        lookback_window=lookback_window,
        num_trading_days=num_trading_days,
        **other_estimator_kwargs,
    )

    # Load stock frame
    stock_frame = pd.read_parquet(f"{CLEAN_PRICE_PATH / stock}.parquet")

    # Compute historical volatility
    hist_vol_frame = estimator.estimate_volatility(stock_frame)

    # Store
    hist_vol_frame.to_parquet(
        f"{HIST_VOL_PATH / estimator_method / stock}.parquet",
        index=False,
    )
