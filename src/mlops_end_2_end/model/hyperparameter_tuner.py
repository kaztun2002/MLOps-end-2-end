from pathlib import Path
from typing import Union

import pickle

import lightgbm as lgb
import optuna
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from mlops_end_2_end.utils import get_logger


class LightGBMHyperparameterTuning:
    """Tune a LightGBM regression model using Optuna."""

    def __init__(
        self,
        project_root: Union[str, Path] | None = None,
        n_trials: int = 50,
        random_state: int = 42,
    ) -> None:
        self.logger = get_logger("hyperparameter_tuning")

        self.project_root = (
            Path(project_root)
            if project_root is not None
            else Path.cwd()
        )

        self.tuned_models_dir = f"{self.project_root}/models/tuned"
        self.tuned_models_dir.mkdir(parents=True, exist_ok=True)

        self.n_trials = n_trials
        self.random_state = random_state

        self.best_params = None
        self.best_model = None

    def objective(
        self,
        trial: optuna.Trial,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> float:
        """Optuna objective function."""

        params = {
            "objective": "regression",
            "metric": "rmse",
            "verbosity": -1,
            "random_state": self.random_state,

            "n_estimators": trial.suggest_int(
                "n_estimators", 100, 1000
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.3, log=True
            ),
            "num_leaves": trial.suggest_int(
                "num_leaves", 20, 150
            ),
            "max_depth": trial.suggest_int(
                "max_depth", 3, 12
            ),
            "min_child_samples": trial.suggest_int(
                "min_child_samples", 10, 100
            ),
            "subsample": trial.suggest_float(
                "subsample", 0.6, 1.0
            ),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.6, 1.0
            ),
            "reg_alpha": trial.suggest_float(
                "reg_alpha", 1e-8, 10.0, log=True
            ),
            "reg_lambda": trial.suggest_float(
                "reg_lambda", 1e-8, 10.0, log=True
            ),
        }

        tscv = TimeSeriesSplit(n_splits=3)

        scores = []

        for train_idx, val_idx in tscv.split(X):
            X_train = X.iloc[train_idx]
            X_val = X.iloc[val_idx]

            y_train = y.iloc[train_idx]
            y_val = y.iloc[val_idx]

            model = lgb.LGBMRegressor(**params)

            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[
                    lgb.early_stopping(50, verbose=False)
                ],
            )

            predictions = model.predict(X_val)

            rmse = mean_squared_error(
                y_val,
                predictions,
                squared=False,
            )

            scores.append(rmse)

        return sum(scores) / len(scores)

    def tune(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> dict:
        """Run Optuna hyperparameter optimization."""

        self.logger.info(
            "Starting Optuna optimization with %d trials",
            self.n_trials,
        )

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(
                seed=self.random_state
            ),
        )

        study.optimize(
            lambda trial: self.objective(trial, X, y),
            n_trials=self.n_trials,
        )

        self.best_params = study.best_params

        self.logger.info(
            "Best RMSE: %.4f",
            study.best_value,
        )

        self.logger.info(
            "Best parameters: %s",
            self.best_params,
        )

        return self.best_params

    def train_best_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> lgb.LGBMRegressor:
        """Train a LightGBM model using the best parameters."""

        if self.best_params is None:
            raise ValueError(
                "Run tune() before train_best_model()."
            )

        params = {
            **self.best_params,
            "objective": "regression",
            "random_state": self.random_state,
            "verbosity": -1,
        }

        self.best_model = lgb.LGBMRegressor(**params)

        self.best_model.fit(X, y)

        model_path = self.tuned_models_dir / "best_lightgbm_model.pkl"

        with open(model_path, "wb") as f:
            pickle.dump(self.best_model, f)

        self.logger.info(
            "Best LightGBM model trained successfully"
        )

        return self.best_model

    def run(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> lgb.LGBMRegressor:
        """Run the complete hyperparameter tuning and training pipeline."""

        self.logger.info(
            "Starting hyperparameter tuning and model training"
        )

        self.tune(X, y)
        best_model = self.train_best_model(X, y)

        self.logger.info(
            "Hyperparameter tuning and model training completed"
        )

        return best_model
