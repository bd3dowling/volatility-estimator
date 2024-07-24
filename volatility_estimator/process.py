from pathlib import Path
from typing import Iterator

import pandas as pd

from volatility_estimator.cleaner import clean_price_frame


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


def incremental_process(file_path: Path) -> pd.DataFrame:
    pass


def base_process(
    stock: str, stock_file_paths: Iterator[Path], stock_splits: dict[str, float]
) -> None:
    # Base processing and cleaning of stock files
    stock_frame = load_clean_comine_price_frames(stock_file_paths, stock_splits)

    # Store as partitioned parquets
    stock_frame.to_parquet(
        f"{CLEAN_DATA_PATH / 'prices' / stock}.parquet", index=False, partition_cols=["date"]
    )

    # TODO: move from raw to processed

    # Compute historical volatility
    hist_vol_frame = hist_vol_comp(stock_frame)

    # TODO: Store historical volatility
