from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum, auto

import numpy as np
import pandas as pd

from volatility_estimator.config import NUM_TRADING_DAYS

__SAMPLER__: dict["VolatilityEstimatorName", type["VolatilityEstimator"]] = {}


def register_estimator(name: "VolatilityEstimatorName"):
    def wrapper(cls):
        if __SAMPLER__.get(name):
            raise NameError(f"Name {name} is already registered!")
        __SAMPLER__[name] = cls
        return cls

    return wrapper


def get_estimator(
    name: "VolatilityEstimatorName",
    lookback_window: int,
    **kwargs,
) -> "VolatilityEstimator":
    if __SAMPLER__.get(name) is None:
        raise NameError(f"Name {name} is not defined!")
    return __SAMPLER__[name](lookback_window=lookback_window, **kwargs)


class VolatilityEstimatorName(StrEnum):
    TICK_AVERAGE_REALISED_VARIANCE = auto()
    CLOSE_TO_CLOSE_STD_DEVIATION = auto()
    YANG_ZHANG = auto()


@dataclass
class VolatilityEstimator(ABC):
    lookback_window: int

    @abstractmethod
    def estimate_volatility(self, price_frame: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


@register_estimator(VolatilityEstimatorName.TICK_AVERAGE_REALISED_VARIANCE)
class TickAverageRealisedVariance(VolatilityEstimator):
    """Trade-to-Trade average realised variance historical volatility estimator."""

    def estimate_volatility(self, price_frame: pd.DataFrame) -> pd.DataFrame:
        price_frame = price_frame.copy()
        prices = price_frame["price"]

        # Compute trade-to-trade log returns, even across days
        price_frame["log_return"] = np.log(prices / prices.shift(1))

        # Aggregate on each day and compute daily realised variance (sum of square LRs)
        daily_realized_variance = price_frame.groupby("date", observed=True)["log_return"].apply(
            lambda group_vector: np.sum(group_vector.dropna() ** 2)
        )

        # Compute (rolling) average
        rolling_arv = daily_realized_variance.rolling(window=self.lookback_window).mean()

        # Annualise
        annualized_rolling_arv = rolling_arv * NUM_TRADING_DAYS

        # Convert variance to volatility
        annualized_rolling_ar_vol = annualized_rolling_arv.apply(np.sqrt)

        return pd.DataFrame(
            {
                "date": _date_field_to_timestamp(annualized_rolling_arv.index),
                "rolling_historical_volatility": annualized_rolling_ar_vol,
            }
        )


@register_estimator(VolatilityEstimatorName.CLOSE_TO_CLOSE_STD_DEVIATION)
class CloseToCloseStdDeviation(VolatilityEstimator):
    """Close to Close historical volatility estimator.

    See:
    - https://portfolioslab.com/tools/close-to-close-volatility
    """

    def estimate_volatility(self, price_frame: pd.DataFrame) -> pd.DataFrame:
        # Get last close price
        last_prices = price_frame.groupby("date", observed=True)["price"].last()

        # Calculate close-to-close log returns
        log_returns = (last_prices / last_prices.shift(1)).apply(np.log)

        # Calculate 30-day rolling standard deviation of returns
        rolling_volatility = log_returns.rolling(window=self.lookback_window).std()

        # Annualise
        rolling_volatility_annualised = rolling_volatility * np.sqrt(NUM_TRADING_DAYS)

        return pd.DataFrame(
            {
                "date": _date_field_to_timestamp(last_prices.index),
                "rolling_historical_volatility": rolling_volatility_annualised,
            }
        )


@register_estimator(VolatilityEstimatorName.YANG_ZHANG)
class YangZhang(VolatilityEstimator):
    """Yang Zhang HLOC historical volatility esimator.

    See:
    - https://harbourfronts.com/garman-klass-yang-zhang-historical-volatility-calculation-volatility-analysis-python/
    - https://portfolioslab.com/tools/yang-zhang
    """

    def estimate_volatility(self, price_frame: pd.DataFrame) -> pd.DataFrame:
        ohlc = price_frame.set_index("ts")["price"].resample("B").ohlc()

        rolling_variance = (NUM_TRADING_DAYS / self.lookback_window) * pd.DataFrame.rolling(
            (ohlc.loc[:, "open"] / ohlc.loc[:, "close"].shift(1)).apply(np.log).fillna(0) ** 2
            + 0.5 * (ohlc.loc[:, "high"] / ohlc.loc[:, "low"]).apply(np.log).fillna(0) ** 2
            - (2 * np.log(2) - 1)
            * (ohlc.loc[:, "close"] / ohlc.loc[:, "open"]).apply(np.log).fillna(0) ** 2,
            window=self.lookback_window,
        ).sum()

        rolling_volatility = rolling_variance.apply(np.sqrt)

        return pd.DataFrame(
            {
                "date": rolling_volatility.index,
                "rolling_historical_volatility": rolling_volatility,
            }
        )


def _date_field_to_timestamp(date_field):
    return np.asarray(pd.to_datetime(date_field))
