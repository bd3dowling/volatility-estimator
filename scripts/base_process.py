from itertools import groupby
from pathlib import Path

import pandas as pd

from volatility_estimator.process import base_process

DATA_PATH = Path() / ".." / "data"
RAW_DATA_PATH = DATA_PATH / "raw"
CLEAN_DATA_PATH = DATA_PATH / "clean"

TICKERS = ("a", "b", "c", "d")
FILE_NAME_PATTERN = "prices_*_*.csv"

START_TIME = pd.Timestamp("8:00").time()
END_TIME = pd.Timestamp("16:30").time()
LAST_DATE = "20170818"

STOCK_SPLITS: dict[str, dict[str, float]] = {"d": {"2017-05-22": 10}}

raw_file_paths = sorted(RAW_DATA_PATH.glob(FILE_NAME_PATTERN))

raw_file_paths_ex_last = [
    file_path for file_path in raw_file_paths if LAST_DATE not in file_path.name
]

for stock, stock_file_paths in groupby(
    raw_file_paths_ex_last,
    key=lambda file_path: file_path.name.split("_")[1],
):
    base_process(stock, stock_file_paths, STOCK_SPLITS.get(stock, {}))
