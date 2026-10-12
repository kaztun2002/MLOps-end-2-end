from pathlib import Path
from typing import Union

import lightgbm as lgb
import pandas as pd

from mlops_end_2_end.utils import get_logger


class LightGBMModelInference:
    """Load fitted Quantile LightGBM models and generate calibrated prediction intervals."""

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
        
        # Will hold a dictionary of models mapped by quantile float keys (e.g., 0.1, 0.5, 0.9)
        self.models: dict[float, lgb.LGBMRegressor] = {}

    def load_model(self) -> dict[float, lgb.LGBMRegressor]:
        """Load the fitted dictionary of LightGBM quantile models."""

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Quantile models file not found: {self.model_path}"
            )

        self.logger.info(
            "Loading quantile models from %s",
            self.model_path,
        )

        # Expecting a serialized dictionary mapping quantiles to models
        self.models = pd.read_pickle(self.model_path)

        if not isinstance(self.models, dict):
            raise TypeError(
                "Expected model artifact to be a dictionary of quantile models. "
                f"Got {type(self.models)} instead."
            )

        self.logger.info(
            "Loaded models successfully for quantiles: %s",
            list(self.models.keys()),
        )

        return self.models

    def prepare_features(
        self,
        df: pd.DataFrame,
        datetime_column: str = "date",
    ) -> pd.DataFrame:
        """Prepare input data for inference by aligning with feature names."""

        if not self.models:
            raise ValueError(
                "Models have not been loaded. "
                "Call load_model() first."
            )

        # Datetime column is not directly passed to LightGBM
        if datetime_column in df.columns:
            df = df.drop(columns=[datetime_column])

        return df

    def predict(
        self,
        df: pd.DataFrame,
        datetime_column: str = "date",
    ) -> pd.DataFrame:
        """Generate point estimates along with calibrated uncertainty bounds.
        
        Returns:
            pd.DataFrame: A DataFrame with columns 'lower_bound', 'point_estimate', 
                          and 'upper_bound' aligned with input indexes.
        """

        if not self.models:
            self.load_model()

        X = self.prepare_features(
            df,
            datetime_column=datetime_column,
        )

        self.logger.info(
            "Generating quantile predictions for %d samples",
            len(X),
        )

        # Loop through the stored quantile models to generate matching vectors
        predictions = {}
        for q, model in self.models.items():
            predictions[q] = model.predict(X)

        self.logger.info("Quantile prediction completed")

        # Dynamically map expected quantiles into clean, scannable column profiles
        # Fallback cleanly if user customized the quantiles list during training
        return pd.DataFrame(
            {
                "lower_bound": predictions.get(0.10, predictions.get(min(self.models.keys()))),
                "point_estimate": predictions.get(0.50, predictions.get(0.5, list(predictions.values())[0])),
                "upper_bound": predictions.get(0.90, predictions.get(max(self.models.keys()))),
            },
            index=df.index,
        )
