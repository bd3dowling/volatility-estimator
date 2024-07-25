from volatility_estimator.config import RAW_DATA_PATH, RAW_FILE_NAME_PATTERN
from volatility_estimator.process import incremental_process_prices

LAST_DATE = "20170818"
STOCK_SPLITS: dict[str, dict[str, float]] = {"d": {"2017-05-22": 10}}

raw_file_paths = sorted(RAW_DATA_PATH.glob(RAW_FILE_NAME_PATTERN))

raw_file_paths_last = [file_path for file_path in raw_file_paths if LAST_DATE in file_path.name]

for file_path in raw_file_paths_last:
    _, stock, date = file_path.stem.split("_")

    incremental_process_prices(
        stock,
        date,
        file_path,
        STOCK_SPLITS.get(stock, {}),
    )

print("done")
