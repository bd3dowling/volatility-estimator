from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from volatility_estimator.cleaner import adjust_for_split, clean_price_frame
from volatility_estimator.config import CLEAN_PRICE_PATH, HIST_VOL_PATH
from volatility_estimator.estimator import VolatilityEstimatorName, get_estimator
from volatility_estimator.logger import get_logger

# pd.bdate_range(end=pd.to_datetime("2017-05-30", format="%Y-%m-%d"), periods=30).date

logger = get_logger()


def base_process_prices(
    stock: str,
    stock_file_paths: Iterator[Path],
    stock_splits: dict[str, float],
) -> None:
    logger.info(f"Processing files for stock {stock}")

    # Base processing and cleaning of stock files
    stock_frame = _load_clean_comine_price_frames(stock_file_paths, stock_splits)

    # Store as partitioned parquets
    logger.info("Saving cleaned price data")
    stock_frame.to_parquet(
        f"{CLEAN_PRICE_PATH / stock}.parquet", index=False, partition_cols=["date"]
    )


def incremental_process_prices(stock: str, file_path: Path, split_ratio: float = 1) -> None:
    price_frame = _load_price_frame(file_path)

    if price_frame.empty:
        logger.warning(f"File {file_path.name} has zero rows")
        return

    # NOTE: if there is a stock split, handle it separately with full history
    cleaned_price_frame = clean_price_frame(price_frame, splits={})

    # Store as new parquet shard corresponding to the date
    cleaned_price_frame.to_parquet(
        f"{CLEAN_PRICE_PATH / stock}.parquet", index=False, partition_cols=["date"]
    )

    # Need to reprocess old price data to align with future...
    if split_ratio != 1:
        return

    logger.info("Stock has split; rebasing old prices...")
    full_stock_frame = pd.read_parquet(f"{CLEAN_PRICE_PATH / stock}.parquet")
    updated_stock_frame = adjust_for_split(full_stock_frame, split_ratio)

    # Re-store full history
    logger.info("Re-saving full stock price history...")
    updated_stock_frame.to_parquet(
        f"{CLEAN_PRICE_PATH / stock}.parquet", index=False, partition_cols=["date"]
    )


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


def _load_price_frame(file_path: Path) -> pd.DataFrame:
    logger.info(f"Loading file {file_path.name}")
    return pd.read_csv(
        file_path,
        dtype={"price": "float64"},
        parse_dates=["ts"],
    )


def _load_clean_comine_price_frames(
    stock_file_paths: Iterator[Path],
    stock_splits: dict[str, float],
) -> pd.DataFrame:
    price_frames: list[pd.DataFrame] = []

    for file_path in stock_file_paths:
        price_frame = _load_price_frame(file_path)

        if price_frame.empty:
            logger.warning(f"File {file_path.name} has zero rows")
            continue

        logger.info(f"Cleaning frame from {file_path.name}")
        cleaned_price_frame = clean_price_frame(price_frame, splits=stock_splits)

        price_frames.append(cleaned_price_frame)

    logger.info("Combining daily price frames")
    return pd.concat(price_frames, ignore_index=True)
