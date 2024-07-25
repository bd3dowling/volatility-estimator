from itertools import groupby

from volatility_estimator.config import DATA_PATH, RAW_FILE_NAME_PATTERN
from volatility_estimator.logger import get_logger
from volatility_estimator.process import base_process_prices

LAST_DATE = "20170818"
STOCK_SPLITS: dict[str, dict[str, float]] = {"d": {"2017-05-22": 10}}

if __name__ == "__main__":
    logger = get_logger()
    logger.info("Starting batch processing of raw price data...")

    batch_data = DATA_PATH / "raw" / "batch"
    logger.info(f"Loading data from {batch_data.absolute()}")

    raw_file_paths = sorted(batch_data.glob(RAW_FILE_NAME_PATTERN))

    if not len(raw_file_paths):
        logger.error("No data found...")

    for stock, stock_file_paths in groupby(
        raw_file_paths,
        key=lambda file_path: file_path.stem.split("_")[1],
    ):
        base_process_prices(stock, stock_file_paths, STOCK_SPLITS.get(stock, {}))

    logger.info("Finishing processing!")
