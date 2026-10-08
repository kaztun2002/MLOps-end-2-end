from pathlib import Path
from typing import Union

import pandas as pd

from mlops_end_2_end.utils import get_logger


class TimeSeriesFeatureEngineering:
    """Create time-series features from cleaned data."""

    def __init__(
        self,
        project_root: Union[str, Path] | None = None,
        datetime_column: str = "date",
    ) -> None:
        self.logger = get_logger("feature_engineering")

        self.project_root = (
            Path(project_root)
            if project_root is not None
            else Path.cwd()
        )

        self.datetime_column = datetime_column

        self.engineered_features_dir = self.project_root / "data" / "engineered_features"
        self.engineered_features_dir.mkdir(parents=True, exist_ok=True)

    def create_lag_features(
        self,
        df: pd.DataFrame,
        column: str,
        lags: list[int],
    ) -> pd.DataFrame:
        """Create lag features from previous observations."""

        # # for multiple timeseries, you might want to group by an identifier column before creating lag features.
        # df = df.sort_values(["store_id", "product_id", "date"])
        # group = df.groupby("col_1", "col_2", ...)
        # df["lag_1"] = group.shift(1)
        # df["lag_7"] = group.shift(7)
        # df["lag_14"] = group.shift(14)

        df = df.copy()

        for lag in lags:
            df[f"{column}_lag_{lag}"] = df[column].shift(lag)

        self.logger.info(
            "Created lag features for %s: %s",
            column,
            lags,
        )

        return df

    def create_rolling_features(
        self,
        df: pd.DataFrame,
        column: str,
        windows: list[int],
    ) -> pd.DataFrame:
        """Create rolling statistical features using past observations."""

        # df = df.copy()

        # df["rolling_mean_7"] = (
        # df.groupby(["store_id", "product_id"])["sales"]
        # .transform(lambda x: x.shift(1).rolling(7).mean()))

        for window in windows:
            # Shift first to prevent data leakage.
            previous_values = df[column].shift(1)

            df[f"{column}_rolling_mean_{window}"] = (
                previous_values.rolling(window).mean()
            )

            df[f"{column}_rolling_std_{window}"] = (
                previous_values.rolling(window).std()
            )

        self.logger.info(
            "Created rolling features for %s: %s",
            column,
            windows,
        )

        return df

    def create_datetime_features(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Create calendar-based features."""

        df = df.copy()

        df[self.datetime_column] = pd.to_datetime(
            df[self.datetime_column]
        )

        df["year"] = df[self.datetime_column].dt.year
        df["month"] = df[self.datetime_column].dt.month
        df["day"] = df[self.datetime_column].dt.day
        df["day_of_week"] = df[self.datetime_column].dt.dayofweek
        df["week_of_year"] = (
            df[self.datetime_column].dt.isocalendar().week
        )

        return df

    def create_features(
        self,
        df: pd.DataFrame,
        target_column: str,
        lags: list[int] | None = None,
        rolling_windows: list[int] | None = None,
    ) -> pd.DataFrame:
        """Run the complete feature-engineering pipeline."""

        self.logger.info("Starting feature engineering")

        lags = lags or [1, 7, 14, 28]
        rolling_windows = rolling_windows or [7, 14, 28]

        df = df.copy()

        # Ensure chronological order
        df = df.sort_values(
            self.datetime_column
        ).reset_index(drop=True)

        # Calendar features
        df = self.create_datetime_features(df)

        # Lag features
        df = self.create_lag_features(
            df=df,
            column=target_column,
            lags=lags,
        )

        # Rolling features
        df = self.create_rolling_features(
            df=df,
            column=target_column,
            windows=rolling_windows,
        )

        df.to_csv(f"{self.engineered_features_dir}/engineered_features.csv", index=False)

        self.logger.info("Feature engineering completed")

        return df
