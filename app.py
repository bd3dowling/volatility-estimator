from logging import Logger
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from scripts.base_compute_volatility import LOOKBACK_WINDOW
from volatility_estimator.config import LOAD_DATA_PATH, RAW_FILE_NAME_PATTERN
from volatility_estimator.estimator import VolatilityEstimatorName
from volatility_estimator.logger import get_logger
from volatility_estimator.process import incremental_compute_volatility, incremental_process_prices

STOCK_SPLITS: dict[str, dict[str, float]] = {"d": {"2017-05-22": 10}}
ESTIMATOR_METHODS = [
    VolatilityEstimatorName.TICK_AVERAGE_REALISED_VARIANCE,
    VolatilityEstimatorName.CLOSE_TO_CLOSE_STD_DEVIATION,
    VolatilityEstimatorName.YANG_ZHANG,
]


class Handler(FileSystemEventHandler):
    def __init__(self, logger: Logger):
        self.logger = logger

    def on_created(self, event: FileSystemEvent):
        logger.info(f"Received created event - {event.src_path}")

        file_path = Path(event.src_path)

        if not file_path.match(RAW_FILE_NAME_PATTERN):
            logger.warning(
                f"Found file {file_path.name} which does not match required pattern "
                "({RAW_FILE_NAME_PATTERN}); leaving file in load directory..."
            )
            return

        _, stock, date = file_path.stem.split("_")
        formatted_date = f"{date[:4]}-{date[4:6]}-{date[6:]}"

        logger.info(f"Processing file {file_path.name}")
        incremental_process_prices(
            stock, file_path, STOCK_SPLITS.get(stock, {}).get(formatted_date, 1)
        )

        logger.info(f"Deleting file {file_path.name}")
        file_path.unlink()

        logger.info(f"Computing historical volatility for {stock}...")
        for estimator_method in ESTIMATOR_METHODS:
            incremental_compute_volatility(
                stock=stock,
                date=formatted_date,
                estimator_method=estimator_method,
                lookback_window=LOOKBACK_WINDOW,
            )

        logger.info("Finished handling event...")


if __name__ == "__main__":
    logger = get_logger()

    event_handler = Handler(logger)
    observer = Observer()
    observer.schedule(event_handler, LOAD_DATA_PATH)
    observer.start()

    logger.info(f"Observer started, watching {LOAD_DATA_PATH.absolute()}")

    try:
        while observer.is_alive():
            observer.join(1)
    finally:
        observer.stop()
        observer.join()
        logger.warning("Observer stopped...")
