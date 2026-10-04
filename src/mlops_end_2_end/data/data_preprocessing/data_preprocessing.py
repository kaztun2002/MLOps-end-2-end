from pathlib import Path
from typing import Union

import pandas as pd
from sklearn.preprocessing import StandardScaler

from mlops_end_2_end.utils import get_logger


class DataPreprocessing:
    """Transform cleaned data into model-ready features."""

    def __init__(self, project_root: Union[str, Path] | None = None) -> None:
        self.logger = get_logger("data_preprocessing")
        self.project_root = (
            Path(project_root) if project_root is not None else Path.cwd()
        )

        self.processed_data_dir = f"{self.project_root}/data/processed"
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)

    def to_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert categorical columns to category dtype."""
        df = df.copy()

        categorical_columns = df.select_dtypes(
            include=["object", "category"], exclude=["datetime"]
        ).columns

        for col in categorical_columns:
            df[col] = df[col].astype("category")

        return df

    def encode_categorical(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Encode categorical columns using one-hot encoding."""
        categorical_columns = df.select_dtypes(
            include=["object", "category"], exclude=["datetime"]
        ).columns

        return pd.get_dummies(
            df,
            columns=categorical_columns,
            drop_first=True,
        )

    def scale_numerical(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Standardize numerical columns."""
        df = df.copy()

        numerical_columns = df.select_dtypes(
            include="number"
        ).columns

        scaler = StandardScaler()

        df[numerical_columns] = scaler.fit_transform(
            df[numerical_columns]
        )

        return df

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run the complete preprocessing pipeline."""
        self.logger.info("Starting data preprocessing")
        
        df = self.to_categorical(df)
        df = self.encode_categorical(df)
        df = self.scale_numerical(df)

        df.to_csv(f"{self.processed_data_dir}/preprocessed_data.csv", index=False)

        self.logger.info("Data preprocessing completed")

        return df
