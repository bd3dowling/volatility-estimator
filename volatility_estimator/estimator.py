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
    CLOSE_TO_CLOSE_AVERAGE_REALISED_VARIANCE = auto()
    YANG_ZHANG = auto()


@dataclass
class VolatilityEstimator(ABC):
    lookback_window: int

    @abstractmethod
    def estimate_volatility(self, price_frame: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


@register_estimator(VolatilityEstimatorName.TICK_AVERAGE_REALISED_VARIANCE)
class TickAverageRealisedVariance(VolatilityEstimator):
    def estimate_volatility(self, price_frame: pd.DataFrame) -> pd.DataFrame:
        price_frame = price_frame.copy()
        prices = price_frame["price"]
        price_frame["log_return"] = np.log(prices / prices.shift(1))

        daily_realized_variance = price_frame.groupby("date", observed=True)["log_return"].apply(
            lambda group_vector: np.sum(group_vector.dropna() ** 2)
        )
        rolling_arv = daily_realized_variance.rolling(window=self.lookback_window).mean()
        annualized_rolling_arv = rolling_arv * NUM_TRADING_DAYS

        return pd.DataFrame(
            {
                "date": annualized_rolling_arv.index,
                "rolling_historical_volatility": np.sqrt(annualized_rolling_arv.values),
            }
        )


@register_estimator(VolatilityEstimatorName.CLOSE_TO_CLOSE_STD_DEVIATION)
class CloseToCloseStdDeviation(VolatilityEstimator):
    def estimate_volatility(self, price_frame: pd.DataFrame) -> pd.DataFrame:
        last_prices = price_frame.groupby("date", observed=True)["price"].last().reset_index()
        log_returns = np.log(last_prices["price"] / last_prices["price"].shift(1))
        rolling_volatility = log_returns.rolling(window=self.lookback_window).std()
        rolling_volatility_annualised = rolling_volatility * np.sqrt(NUM_TRADING_DAYS)

        return pd.DataFrame(
            {
                "date": rolling_volatility.index,
                "rolling_historical_volatility": rolling_volatility_annualised,
            }
        )


@register_estimator(VolatilityEstimatorName.CLOSE_TO_CLOSE_AVERAGE_REALISED_VARIANCE)
class CloseToCloseAverageRealisedVariance(VolatilityEstimator):
    def estimate_volatility(self, price_frame: pd.DataFrame) -> pd.DataFrame:
        last_prices = price_frame.groupby("date", observed=True)["price"].last().reset_index()
        log_returns = np.log(last_prices["price"] / last_prices["price"].shift(1))

        # Compute daily realized variance (squared log returns)
        daily_realized_variance = log_returns**2

        # Compute the rolling 30-day sum of daily realized variances
        rolling_variance = (
            daily_realized_variance.rolling(window=self.lookback_window).mean() * NUM_TRADING_DAYS
        )

        # Convert rolling variance to volatility (annualized)
        rolling_volatility = np.sqrt(rolling_variance)

        return pd.DataFrame(
            {"date": last_prices["date"], "rolling_historical_volatility": rolling_volatility}
        )


@register_estimator(VolatilityEstimatorName.YANG_ZHANG)
class YangZhang(VolatilityEstimator):
    def estimate_volatility(self, price_frame: pd.DataFrame) -> pd.DataFrame:
        ohlc = price_frame.set_index("ts")["price"].resample("B").ohlc()

        rolling_volatility = np.sqrt(
            (NUM_TRADING_DAYS / self.lookback_window)
            * pd.DataFrame.rolling(
                np.log(ohlc.loc[:, "open"] / ohlc.loc[:, "close"].shift(1)).fillna(0) ** 2
                + 0.5 * np.log(ohlc.loc[:, "high"] / ohlc.loc[:, "low"]).fillna(0) ** 2
                - (2 * np.log(2) - 1)
                * np.log(ohlc.loc[:, "close"] / ohlc.loc[:, "open"]).fillna(0) ** 2,
                window=self.lookback_window,
            ).sum()
        )

        return rolling_volatility.to_frame().reset_index()
