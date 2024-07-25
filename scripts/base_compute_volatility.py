from volatility_estimator.config import CLEAN_PRICE_PATH
from volatility_estimator.estimator import VolatilityEstimatorName
from volatility_estimator.logger import get_logger
from volatility_estimator.process import base_compute_volatility

STOCK_SPLITS: dict[str, dict[str, float]] = {"d": {"2017-05-22": 10}}

LOOKBACK_WINDOW = 30
ESTIMATOR_METHODS = [
    VolatilityEstimatorName.TICK_AVERAGE_REALISED_VARIANCE,
    VolatilityEstimatorName.CLOSE_TO_CLOSE_STD_DEVIATION,
    VolatilityEstimatorName.YANG_ZHANG,
]

if __name__ == "__main__":
    logger = get_logger()
    logger.info("Starting batch computation of historical volatility...")

    logger.info(f"Loading data from {CLEAN_PRICE_PATH.absolute()}")

    # NOTE: will match directories (i.e. handles sharded parquets)
    for stock_parquet_path in CLEAN_PRICE_PATH.glob("*.parquet"):
        stock = stock_parquet_path.stem
        logger.info(f"Computing historical volatility for {stock}...")

        for estimator_method in ESTIMATOR_METHODS:
            base_compute_volatility(
                stock=stock,
                estimator_method=estimator_method,
                lookback_window=LOOKBACK_WINDOW,
            )

    logger.info("Finished computation!")
