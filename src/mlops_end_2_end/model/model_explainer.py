from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import pandas as pd
import shap

from mlops_end_2_end.utils import get_logger


class LightGBMSHAPExplainer:
    """Explain a fitted LightGBM model using SHAP."""

    def __init__(
        self,
        model,
        project_root: Union[str, Path] | None = None,
    ) -> None:
        self.logger = get_logger("model_explainability")

        self.project_root = (
            Path(project_root)
            if project_root is not None
            else Path.cwd()
        )

        self.model = model
        self.explainer = shap.TreeExplainer(model)

    def calculate_shap_values(
        self,
        X: pd.DataFrame,
    ) -> shap.Explanation:
        """Calculate SHAP values for the supplied data."""

        self.logger.info(
            "Calculating SHAP values for %d samples",
            len(X),
        )

        shap_values = self.explainer(X)

        return shap_values

    def feature_importance(
        self,
        X: pd.DataFrame,
    ) -> pd.DataFrame:
        """Return global feature importance based on mean absolute SHAP values."""

        shap_values = self.calculate_shap_values(X)

        importance = pd.DataFrame({
            "feature": X.columns,
            "importance": abs(shap_values.values).mean(axis=0),
        })

        importance = importance.sort_values(
            by="importance",
            ascending=False,
        ).reset_index(drop=True)

        return importance

    def explain_prediction(
        self,
        X: pd.DataFrame,
        index: int = 0,
    ) -> pd.DataFrame:
        """Explain an individual prediction."""

        shap_values = self.calculate_shap_values(X)

        explanation = pd.DataFrame({
            "feature": X.columns,
            "feature_value": X.iloc[index].values,
            "shap_value": shap_values.values[index],
        })

        explanation["abs_shap_value"] = (
            explanation["shap_value"].abs()
        )

        return explanation.sort_values(
            by="abs_shap_value",
            ascending=False,
        ).reset_index(drop=True)

    def plot_feature_importance(
        self,
        X: pd.DataFrame,
        max_display: int = 20,
    ) -> None:
        """Plot global SHAP feature importance."""

        shap_values = self.calculate_shap_values(X)

        shap.plots.bar(
            shap_values,
            max_display=max_display,
            show=False,
        )

        plt.tight_layout()
        plt.show()

    def plot_summary(
        self,
        X: pd.DataFrame,
        max_display: int = 20,
    ) -> None:
        """Plot SHAP summary/beeswarm plot."""

        shap_values = self.calculate_shap_values(X)

        shap.plots.beeswarm(
            shap_values,
            max_display=max_display,
            show=False,
        )

        plt.tight_layout()
        plt.show()

    def plot_prediction(
        self,
        X: pd.DataFrame,
        index: int = 0,
    ) -> None:
        """Plot explanation for an individual prediction."""

        shap_values = self.calculate_shap_values(X)

        shap.plots.waterfall(
            shap_values[index],
            show=False,
        )

        plt.tight_layout()
        plt.show()
