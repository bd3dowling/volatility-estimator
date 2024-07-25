from itertools import groupby

from volatility_estimator.config import RAW_DATA_PATH, RAW_FILE_NAME_PATTERN
from volatility_estimator.process import base_process_prices

LAST_DATE = "20170818"
STOCK_SPLITS: dict[str, dict[str, float]] = {"d": {"2017-05-22": 10}}

raw_file_paths = sorted(RAW_DATA_PATH.glob(RAW_FILE_NAME_PATTERN))

raw_file_paths_ex_last = [
    file_path for file_path in raw_file_paths if LAST_DATE not in file_path.name
]

for stock, stock_file_paths in groupby(
    raw_file_paths_ex_last,
    key=lambda file_path: file_path.stem.split("_")[1],
):
    base_process_prices(
        stock,
        stock_file_paths,
        STOCK_SPLITS.get(stock, {}),
    )

print("done")
