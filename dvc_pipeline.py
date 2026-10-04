"""Command-line entry point for the DVC pipeline stages."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mlops_end_2_end.data.data_cleaning.data_cleaning_timeseries import DataCleaning
from mlops_end_2_end.data.data_ingestion.data_ingestion import DataIngestion
from mlops_end_2_end.data.data_preprocessing.data_preprocessing import DataPreprocessing
from mlops_end_2_end.model.feature_engineering import TimeSeriesFeatureEngineering
from mlops_end_2_end.model.model_explainer import LightGBMSHAPExplainer
from mlops_end_2_end.model.model_inference import LightGBMModelInference
from mlops_end_2_end.model.model_monitoring import EvidentlyModelMonitoring
from mlops_end_2_end.model.model_training import LightGBMModelTraining


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required pipeline input not found: {path}")
    return pd.read_csv(path)


def _run_data_ingestion() -> None:
    source = PROJECT_ROOT / "notebooks" / "sales_timeseries_2025.csv"
    DataIngestion(project_root=PROJECT_ROOT).load_data(source)


def _run_data_cleaning() -> None:
    source = PROJECT_ROOT / "data" / "raw" / "sales_timeseries_2025.csv"
    DataCleaning(project_root=PROJECT_ROOT).clean(_read_csv(source))


def _run_data_preprocessing() -> None:
    source = PROJECT_ROOT / "data" / "processed" / "cleaned_data.csv"
    DataPreprocessing(project_root=PROJECT_ROOT).preprocess(_read_csv(source))


def _run_feature_engineering() -> None:
    source = PROJECT_ROOT / "data" / "processed" / "cleaned_data.csv"
    TimeSeriesFeatureEngineering(project_root=PROJECT_ROOT).create_features(
        _read_csv(source),
        target_column="sales",
    )


def _run_model_training() -> None:
    feature_path = PROJECT_ROOT / "data" / "engineered_features" / "engineered_features.csv"
    parameters = yaml.safe_load((PROJECT_ROOT / "params.yaml").read_text(encoding="utf-8"))
    hyperparameters = parameters["hyperparameter"]
    trainer = LightGBMModelTraining(
        project_root=PROJECT_ROOT,
        train_size=float(hyperparameters["train_fraction"]),
        random_state=int(hyperparameters["random_seed"]),
    )
    model_parameters = {
        "objective": "regression",
        "verbosity": -1,
        "learning_rate": float(hyperparameters["learning_rate"]),
        "n_estimators": int(hyperparameters["n_estimators"]),
        "num_leaves": int(hyperparameters["num_leaves"]),
        "max_depth": int(hyperparameters["max_depth"]),
        "subsample": float(hyperparameters["subsample"]),
        "colsample_bytree": float(hyperparameters["colsample_bytree"]),
        "reg_alpha": float(hyperparameters["reg_alpha"]),
        "reg_lambda": float(hyperparameters["reg_lambda"]),
        "random_state": int(hyperparameters["random_seed"]),
    }
    metrics = trainer.train(_read_csv(feature_path), model_params=model_parameters)
    metrics_path = PROJECT_ROOT / "reports" / "training_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def _load_model() -> object:
    model_path = PROJECT_ROOT / "models" / "trained" / "lightgbm_model.pkl"
    if not model_path.is_file():
        raise FileNotFoundError(f"Trained model not found: {model_path}")
    with model_path.open("rb") as model_file:
        return pickle.load(model_file)


def _run_model_explainer() -> None:
    feature_path = PROJECT_ROOT / "data" / "engineered_features" / "engineered_features.csv"
    data = _read_csv(feature_path)
    trainer = LightGBMModelTraining(project_root=PROJECT_ROOT)
    _, validation = trainer.split_data(data)
    features, _ = trainer.prepare_features(validation)
    explainer = LightGBMSHAPExplainer(_load_model(), project_root=PROJECT_ROOT)
    importance = explainer.feature_importance(features)

    output_dir = PROJECT_ROOT / "reports" / "explanations"
    output_dir.mkdir(parents=True, exist_ok=True)
    importance.to_csv(output_dir / "shap_feature_importance.csv", index=False)

    shap_values = explainer.calculate_shap_values(features)
    import shap

    shap.plots.beeswarm(shap_values, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_summary.png", dpi=160, bbox_inches="tight")
    plt.close()


def _run_model_inference() -> None:
    feature_path = PROJECT_ROOT / "data" / "engineered_features" / "engineered_features.csv"
    data = _read_csv(feature_path)
    model_path = PROJECT_ROOT / "models" / "trained" / "lightgbm_model.pkl"
    inference = LightGBMModelInference(model_path=model_path, project_root=PROJECT_ROOT)
    predictions = inference.predict(data.drop(columns=["sales"]))
    result = pd.DataFrame(
        {
            "date": data["date"],
            "actual": data["sales"],
            "prediction": predictions.to_numpy(),
        }
    )
    output_path = PROJECT_ROOT / "data" / "predictions" / "predictions.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)


def _run_model_monitoring() -> None:
    EvidentlyModelMonitoring(project_root=PROJECT_ROOT, n_runs=10).run()


STAGES = {
    "data_ingestion": _run_data_ingestion,
    "data_cleaning": _run_data_cleaning,
    "data_preprocessing": _run_data_preprocessing,
    "feature_engineering": _run_feature_engineering,
    "model_training": _run_model_training,
    "model_explainer": _run_model_explainer,
    "model_inference": _run_model_inference,
    "model_monitoring": _run_model_monitoring,
}


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in STAGES:
        valid_stages = ", ".join(STAGES)
        raise SystemExit(f"Usage: python dvc_pipeline.py <stage>; stages: {valid_stages}")
    STAGES[sys.argv[1]]()


if __name__ == "__main__":
    main()