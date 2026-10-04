from pathlib import Path
from typing import Union

import pandas as pd

from mlops_end_2_end.utils import get_logger


class DataCleaning:
    """Clean and validate ingested CSV data."""

    def __init__(self, project_root: Union[str, Path] | None = None) -> None:
        self.logger = get_logger("data_cleaning")
        self.project_root = (
            Path(project_root) if project_root is not None else Path.cwd()
        )
 
        self.logger.info("Data cleaning initialized with project root: %s", self.project_root)
        self.processed_data_dir = self.project_root / "data" / "processed"
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows."""
        before = len(df)

        df = df.drop_duplicates().copy()

        self.logger.info(
            "Removed %d duplicate rows",
            before - len(df),
        )

        return df

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values using simple defaults."""
        df = df.copy()

        numeric_columns = df.select_dtypes(include="number").columns
        categorical_columns = df.select_dtypes(exclude="number").columns

        df[numeric_columns] = df[numeric_columns].fillna(
            df[numeric_columns].median()
        )

        df[categorical_columns] = df[categorical_columns].fillna("Unknown")

        return df

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run the complete cleaning pipeline."""
        self.logger.info("Starting data cleaning")

        df = self.remove_duplicates(df)
        df = self.handle_missing_values(df)

        df.to_csv(f"{self.processed_data_dir}/cleaned_data.csv", index=False)

        self.logger.info("Data cleaning completed")

        return df
