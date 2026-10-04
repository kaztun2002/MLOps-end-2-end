from pathlib import Path
from typing import Union

import pandas as pd

from mlops_end_2_end.utils import get_logger


class DataCleaning:
    """Clean and validate time-series CSV data."""

    def __init__(
        self,
        project_root: Union[str, Path] | None = None,
        datetime_column: str = "date",
    ) -> None:
        self.logger = get_logger("data_cleaning")

        self.project_root = (
            Path(project_root)
            if project_root is not None
            else Path.cwd()
        )

        self.datetime_column = datetime_column

        self.processed_data_dir = self.project_root / "data" / "processed"
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        

    def validate_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and convert the datetime column."""

        if self.datetime_column not in df.columns:
            raise ValueError(
                f"Datetime column '{self.datetime_column}' not found."
            )

        df = df.copy()

        df[self.datetime_column] = pd.to_datetime(
            df[self.datetime_column],
            errors="coerce",
        )

        if df[self.datetime_column].isna().any():
            raise ValueError(
                f"Invalid or missing values found in "
                f"'{self.datetime_column}'."
            )

        return df

    def sort_by_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sort observations chronologically."""

        df = df.sort_values(
            by=self.datetime_column
        ).reset_index(drop=True)

        self.logger.info(
            "Data sorted by %s in ascending order",
            self.datetime_column,
        )

        return df

    def check_duplicate_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Check for duplicate timestamps."""

        duplicates = df[self.datetime_column].duplicated()

        if duplicates.any():
            count = duplicates.sum()

            self.logger.warning(
                "Found %d duplicate timestamps",
                count,
            )

        return df

    def check_missing_dates(
        self,
        df: pd.DataFrame,
        frequency: str = "D",
    ) -> pd.DataFrame:
        """Check for missing dates in a time series."""

        date_range = pd.date_range(
            start=df[self.datetime_column].min(),
            end=df[self.datetime_column].max(),
            freq=frequency,
        )

        existing_dates = pd.DatetimeIndex(
            df[self.datetime_column]
        )

        missing_dates = date_range.difference(existing_dates)

        if len(missing_dates) > 0:
            self.logger.warning(
                "Found %d missing dates",
                len(missing_dates),
            )

            self.logger.warning(
                "Missing dates: %s",
                missing_dates.tolist(),
            )
        else:
            self.logger.info("No missing dates found.")

        return df

    def remove_duplicates(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Remove duplicate rows."""

        before = len(df)

        df = df.drop_duplicates().copy()

        self.logger.info(
            "Removed %d duplicate rows",
            before - len(df),
        )

        return df

    def handle_missing_values(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Handle missing feature values."""

        df = df.copy()

        numeric_columns = df.select_dtypes(
            include="number"
        ).columns

        categorical_columns = df.select_dtypes(
            exclude="number"
        ).columns

        df[numeric_columns] = df[numeric_columns].ffill()

        df[categorical_columns] = df[categorical_columns].ffill()

        return df

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run the complete cleaning pipeline."""

        self.logger.info("Starting time-series data cleaning")

        df = self.validate_datetime(df)
        df = self.sort_by_datetime(df)
        df = self.check_duplicate_dates(df)
        df = self.check_missing_dates(df)
        df = self.remove_duplicates(df)
        df = self.handle_missing_values(df)

        df.to_csv(f"{self.processed_data_dir}/cleaned_data.csv", index=False)

        self.logger.info("Data cleaning completed")

        return df
