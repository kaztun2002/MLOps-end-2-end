from pathlib import Path
from typing import Union
import pickle
import numpy as np
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from mlops_end_2_end.utils import get_logger


class LightGBMModelTraining:
    """Train and validate a Quantile LightGBM model to produce calibrated intervals."""

    def __init__(
        self,
        project_root: Union[str, Path] | None = None,
        target_column: str = "sales",
        datetime_column: str = "date",
        train_size: float = 0.8,
        random_state: int = 42,
        quantiles: list[float] | None = None,
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
        
        # Default to 10th (lower), 50th (median/point), and 90th (upper) percentiles
        self.quantiles = quantiles or [0.10, 0.50, 0.90]

        # Container dictionary to hold the LightGBM models for each quantile
        self.models = {}
        self.metrics = {}

        self.trained_models_dir = self.project_root / "models" / "trained"
        self.trained_models_dir.mkdir(parents=True, exist_ok=True)

    def split_data(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split time-series data chronologically."""
        if not 0 < self.train_size < 1:
            raise ValueError("train_size must be between 0 and 1.")

        df = df.sort_values(self.datetime_column).reset_index(drop=True)
        split_index = int(len(df) * self.train_size)

        train_df = df.iloc[:split_index].copy()
        validation_df = df.iloc[split_index:].copy()

        self.logger.info("Training samples: %d", len(train_df))
        self.logger.info("Validation samples: %d", len(validation_df))

        return train_df, validation_df

    def prepare_features(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Separate features and target."""
        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found.")

        X = df.drop(columns=[self.target_column, self.datetime_column])
        y = df[self.target_column]

        return X, y

    def train(
        self,
        df: pd.DataFrame,
        model_params: dict | None = None,
    ) -> dict:
        """Train standard LightGBM Quantile models and evaluate coverage/calibration."""
        self.logger.info("Starting Quantile LightGBM model training")

        train_df, validation_df = self.split_data(df)
        X_train, y_train = self.prepare_features(train_df)
        X_val, y_val = self.prepare_features(validation_df)

        # Base parameters configurations
        base_params = model_params or {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "random_state": self.random_state,
            "verbosity": -1,
        }
        
        # Enforce quantile objective setup
        base_params["objective"] = "quantile"

        val_predictions = {}

        # Loop through each target alpha quantile independently
        for q in self.quantiles:
            self.logger.info(f"Training LightGBM Regressor for quantile: {q}")
            
            q_params = base_params.copy()
            q_params["alpha"] = q
            
            model = lgb.LGBMRegressor(**q_params)
            model.fit(
                X_train,
                y_train,
                categorical_feature="auto",
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
            
            self.models[q] = model
            val_predictions[q] = model.predict(X_val)

        # Save model dictionary artifact
        with open(self.trained_models_dir / "quantile_lightgbm_models.pkl", "wb") as f:
            pickle.dump(self.models, f)

        # Calculate standard point estimation metrics utilizing the 50th percentile (Median)
        pred_median = val_predictions[0.50]
        self.metrics = {
            "MAE_p50": mean_absolute_error(y_val, pred_median),
            "RMSE_p50": mean_squared_error(y_val, pred_median) ** 0.5,
            "R2_p50": r2_score(y_val, pred_median),
        }

        # Calculate Uncertainty Calibration Metrics (Empirical Coverage)
        if 0.10 in self.quantiles and 0.90 in self.quantiles:
            pred_low = val_predictions[0.10]
            pred_high = val_predictions[0.90]
            
            # Count how often the true value falls inside our target 80% interval
            inside_interval = (y_val >= pred_low) & (y_val <= pred_high)
            empirical_coverage = np.mean(inside_interval)
            
            self.metrics["Empirical_Coverage_80%"] = float(empirical_coverage)
            self.metrics["Calibration_Error_80%"] = float(empirical_coverage - 0.80)
            
            self.logger.info("Validation 80%% Prediction Interval Coverage: %.4f", self.metrics["Empirical_Coverage_80%"])
            self.logger.info("Validation Interval Calibration Error: %.4f", self.metrics["Calibration_Error_80%"])

        self.logger.info("Validation Median MAE: %.4f", self.metrics["MAE_p50"])
        self.logger.info("Validation Median RMSE: %.4f", self.metrics["RMSE_p50"])

        return self.metrics
