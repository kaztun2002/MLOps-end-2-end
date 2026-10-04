from pathlib import Path
from typing import Union

import pickle

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from mlops_end_2_end.utils import get_logger


class LightGBMModelTraining:
    """Train and validate a LightGBM model on time-series data."""

    def __init__(
        self,
        project_root: Union[str, Path] | None = None,
        target_column: str = "sales",
        datetime_column: str = "date",
        train_size: float = 0.8,
        random_state: int = 42,
    ) -> None:
        self.logger = get_logger("model_training")

        self.project_root = (
            Path(project_root)
            if project_root is not None
            else Path.cwd()
        )

        self.target_column = target_column
        self.datetime_column = datetime_column
        self.train_size = train_size
        self.random_state = random_state

        self.model = None
        self.metrics = {}

        self.trained_models_dir = self.project_root / "models" / "trained"
        self.trained_models_dir.mkdir(parents=True, exist_ok=True)

    def split_data(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split time-series data chronologically."""

        if not 0 < self.train_size < 1:
            raise ValueError(
                "train_size must be between 0 and 1."
            )

        df = df.sort_values(
            self.datetime_column
        ).reset_index(drop=True)

        split_index = int(len(df) * self.train_size)

        train_df = df.iloc[:split_index].copy()
        validation_df = df.iloc[split_index:].copy()

        self.logger.info(
            "Training samples: %d",
            len(train_df),
        )

        self.logger.info(
            "Validation samples: %d",
            len(validation_df),
        )

        return train_df, validation_df

    def prepare_features(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Separate features and target."""

        if self.target_column not in df.columns:
            raise ValueError(
                f"Target column '{self.target_column}' "
                "not found."
            )

        X = df.drop(
            columns=[
                self.target_column,
                self.datetime_column,
            ]
        )

        y = df[self.target_column]

        return X, y

    def train(
        self,
        df: pd.DataFrame,
        model_params: dict | None = None,
    ) -> dict:
        """Train LightGBM and evaluate on the final 20%."""

        self.logger.info(
            "Starting LightGBM model training"
        )

        train_df, validation_df = self.split_data(df)

        X_train, y_train = self.prepare_features(train_df)
        X_val, y_val = self.prepare_features(validation_df)

        params = model_params or {
            "objective": "regression",
            "n_estimators": 500,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "random_state": self.random_state,
            "verbosity": -1,
        }

        self.model = lgb.LGBMRegressor(**params)

        self.model.fit(
            X_train,
            y_train,
            categorical_feature="auto",
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(
                    50,
                    verbose=False,
                )
            ],
        )

        with open(
            f"{self.trained_models_dir}/lightgbm_model.pkl",
            "wb",
        ) as f:
            pickle.dump(self.model, f)

        predictions = self.model.predict(X_val)

        self.metrics = {
            "MAE": mean_absolute_error(
                y_val,
                predictions,
            ),
            "RMSE": mean_squared_error(
                y_val,
                predictions,
            ) ** 0.5,
            "R2": r2_score(
                y_val,
                predictions,
            ),
        }

        self.logger.info(
            "Validation MAE: %.4f",
            self.metrics["MAE"],
        )

        self.logger.info(
            "Validation RMSE: %.4f",
            self.metrics["RMSE"],
        )

        self.logger.info(
            "Validation R2: %.4f",
            self.metrics["R2"],
        )

        return self.metrics
