import numpy as np
import pandas as pd

# pd.bdate_range(end=pd.to_datetime("2017-05-30", format="%Y-%m-%d"), periods=30).date


def hist_vol_comp(
    price_frame: pd.DataFrame, window: int = 30, num_trading_days: int = 252
) -> pd.DataFrame:
    price_frame = price_frame.copy()
    prices = price_frame["price"]
    price_frame["log_return"] = np.log(prices / prices.shift(1))

    daily_realized_variance = price_frame.groupby("date", observed=True)["log_return"].apply(
        lambda group_vector: np.sum(group_vector.dropna() ** 2)
    )
    rolling_arv = daily_realized_variance.rolling(window=window).mean()
    annualized_rolling_arv = rolling_arv * num_trading_days

    return pd.DataFrame(
        {
            "date": annualized_rolling_arv.index,
            "rolling_historical_volatility": np.sqrt(annualized_rolling_arv.values),
        }
    )


# RV approach

last_prices = test_clean_frame.groupby("date", observed=True)["price"].last().reset_index()
log_returns = np.log(last_prices["price"] / last_prices["price"].shift(1))

# Compute daily realized variance (squared log returns)
daily_realized_variance = log_returns**2

# Compute the rolling 30-day sum of daily realized variances
rolling_variance = daily_realized_variance.rolling(window=30).mean() * 252

# Convert rolling variance to volatility (annualized)
rolling_volatility = np.sqrt(rolling_variance)

plot_frame_0 = pd.DataFrame(
    {"date": last_prices["date"], "rolling_historical_volatility": rolling_volatility}
)


# STD Dev approach

last_prices = test_clean_frame.groupby("date", observed=True)["price"].last().reset_index()

last_prices["log_returns"] = np.log(last_prices["price"] / last_prices["price"].shift(1))

# Compute the rolling 30-day sum of daily realized variances
rolling_std = last_prices["log_returns"].rolling(window=30).std()

# Convert rolling variance to volatility (annualized)
rolling_volatility = rolling_std * np.sqrt(252)

plot_frame_1 = pd.DataFrame(
    {"date": last_prices["date"], "rolling_historical_volatility": rolling_volatility}
)


# Yang Zhang

ohlc_df = test_clean_frame.set_index("ts")["price"].resample("B").ohlc()

gkyzhv = np.sqrt(
    (252 / 30)
    * pd.DataFrame.rolling(
        np.log(ohlc_df.loc[:, "open"] / ohlc_df.loc[:, "close"].shift(1)).fillna(0) ** 2
        + 0.5 * np.log(ohlc_df.loc[:, "high"] / ohlc_df.loc[:, "low"]).fillna(0) ** 2
        - (2 * np.log(2) - 1)
        * np.log(ohlc_df.loc[:, "close"] / ohlc_df.loc[:, "open"]).fillna(0) ** 2,
        window=30,
    ).sum()
)

plot_frame_2 = gkyzhv.to_frame().reset_index()
