from pathlib import Path
from typing import Union

import lightgbm as lgb
import pandas as pd

from mlops_end_2_end.utils import get_logger


class LightGBMModelInference:
    """Load a fitted LightGBM model and generate predictions."""

    def __init__(
        self,
        model_path: Union[str, Path],
        project_root: Union[str, Path] | None = None,
    ) -> None:
        self.logger = get_logger("model_inference")

        self.project_root = (
            Path(project_root)
            if project_root is not None
            else Path.cwd()
        )

        self.model_path = Path(model_path)
        self.model = None

    def load_model(self) -> lgb.LGBMRegressor:
        """Load the fitted LightGBM model from a pickle file."""

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}"
            )

        self.logger.info(
            "Loading model from %s",
            self.model_path,
        )

        self.model = pd.read_pickle(self.model_path)

        self.logger.info("Model loaded successfully")

        return self.model

    def prepare_features(
        self,
        df: pd.DataFrame,
        datetime_column: str = "date",
    ) -> pd.DataFrame:
        """Prepare input data for inference."""

        if self.model is None:
            raise ValueError(
                "Model has not been loaded. "
                "Call load_model() first."
            )

        df = df.copy()

        # Datetime is not directly passed to the model.
        if datetime_column in df.columns:
            df = df.drop(columns=[datetime_column])

        return df

    def predict(
        self,
        df: pd.DataFrame,
        datetime_column: str = "date",
    ) -> pd.Series:
        """Generate predictions from input data."""

        if self.model is None:
            self.load_model()

        X = self.prepare_features(
            df,
            datetime_column=datetime_column,
        )

        self.logger.info(
            "Generating predictions for %d samples",
            len(X),
        )

        predictions = self.model.predict(X)

        self.logger.info("Prediction completed")

        return pd.Series(
            predictions,
            index=df.index,
            name="prediction",
        )
