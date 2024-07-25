import shutil
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from volatility_estimator.cleaner import adjust_for_split, clean_price_frame
from volatility_estimator.config import CLEAN_PRICE_PATH, HIST_VOL_PATH
from volatility_estimator.estimator import VolatilityEstimatorName, get_estimator
from volatility_estimator.logger import get_logger

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
    output_path = CLEAN_PRICE_PATH / f"{stock}.parquet"

    # Check doesn't exist already; delete if does
    if output_path.exists():
        logger.warning(f"Found existing data at {output_path}; deleting...")

        # NOTE: might point to directory (since parquet sharded)
        if output_path.is_file():
            output_path.unlink()
        else:
            shutil.rmtree(output_path)

    logger.info("Saving cleaned price data")
    stock_frame.to_parquet(output_path, index=False, partition_cols=["date"])


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
    if split_ratio == 1:
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
    **other_estimator_kwargs: Any,
) -> None:
    # Instantiate estimator
    estimator = get_estimator(
        estimator_method,
        lookback_window=lookback_window,
        **other_estimator_kwargs,
    )

    # Load stock frame
    stock_frame = pd.read_parquet(f"{CLEAN_PRICE_PATH / stock}.parquet")

    # Compute historical volatility
    logger.info(
        f"Computing historical volatility using {estimator_method=} with {lookback_window=}"
    )
    hist_vol_frame = estimator.estimate_volatility(stock_frame)

    # Store volatility calculation
    output_path = HIST_VOL_PATH / stock / f"{estimator_method}_{lookback_window}.parquet"

    # Make directory if doesn't exist already
    output_path.parent.mkdir(exist_ok=True)

    logger.info(f"Storing result at {output_path}")
    hist_vol_frame.to_parquet(output_path, index=False)


def incremental_compute_volatility(
    stock: str,
    date: str,
    estimator_method: VolatilityEstimatorName,
    lookback_window: int,
    **other_estimator_kwargs: Any,
) -> None:
    # Instantiate estimator
    estimator = get_estimator(
        estimator_method,
        lookback_window=lookback_window,
        **other_estimator_kwargs,
    )

    # Get previous lookback trading days
    # TODO: use better lookup
    previous_days = pd.bdate_range(end=pd.to_datetime(date, format="%Y-%m-%d"), periods=lookback_window).date
    logger.info(f"Only considering dates {previous_days}")

    # NOTE: str(datetime.date) -> "YYYY-MM-DD"
    parquet_file_paths_and_dates = [
        (prev_date, CLEAN_PRICE_PATH / f"{stock}.parquet" / f"date={prev_date}")
        for prev_date in previous_days
    ]

    logger.info(f"Loading relevant parquet shards at {CLEAN_PRICE_PATH / stock}")
    subset_stock_frames: list[pd.DataFrame] = []
    for prev_date, file_path in parquet_file_paths_and_dates:
        if not file_path.exists():
            logger.critical(f"Did not find data for {file_path.name}")
            return

        date_price_frame = pd.read_parquet(file_path).assign(date=prev_date)
        subset_stock_frames.append(date_price_frame)

    full_subset_stock_frame = pd.concat(subset_stock_frames, ignore_index=True)

    # Compute historical volatility
    logger.info(
        f"Computing historical volatility using {estimator_method=} with {lookback_window=}"
    )

    hist_vol_frame = estimator.estimate_volatility(full_subset_stock_frame)
    new_vol_estimate = hist_vol_frame.tail(1)

    # Load previous volatility frame
    output_path = HIST_VOL_PATH / stock / f"{estimator_method}_{lookback_window}.parquet"

    logger.info("Loading previous volatility calc frame")
    prev_vol = pd.read_parquet(output_path)

    # Add new row
    logger.info("Adding new row to volatility calc frame")
    new_vol = pd.concat([prev_vol, new_vol_estimate], ignore_index=True)

    # Re-save
    logger.info(f"Re-storing result at {output_path}")
    new_vol.to_parquet(output_path, index=False)


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
