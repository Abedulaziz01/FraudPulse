from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_cleaning import CREDIT_TARGET, FRAUD_TARGET, DataCleaner
from src.feature_engineering import FeatureEngineer
from src.modeling import DatasetTrainingResult, train_and_compare_models

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class PipelineConfig:
    """Runtime configuration for the training workflow."""

    data_dir: Path = Path("data")
    artifact_dir: Path = Path("artifacts")
    test_size: float = 0.2
    random_state: int = 42


def resolve_creditcard_path(data_dir: Path) -> Path:
    """Support the current nested credit-card path and a flat file layout."""
    candidates = [
        data_dir / "creditcard.csv",
        data_dir / "creditcard.csv" / "creditcard.csv",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    raise FileNotFoundError("Could not locate creditcard.csv in the data directory.")


def plot_class_distribution(target: pd.Series, output_path: Path, title: str) -> None:
    """Persist a simple class balance chart for dashboard use."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    distribution = target.value_counts().sort_index()

    figure, axis = plt.subplots(figsize=(6, 4))
    distribution.plot(kind="bar", ax=axis, color=["#9cc5a1", "#d1495b"])
    axis.set_title(title)
    axis.set_xlabel("Class")
    axis.set_ylabel("Count")
    axis.set_xticklabels([str(index) for index in distribution.index], rotation=0)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def load_fraud_dataset(data_dir: Path) -> pd.DataFrame:
    fraud_data = pd.read_csv(data_dir / "Fraud_Data.csv")
    ip_mapping = pd.read_csv(data_dir / "IpAddress_to_Country.csv")

    cleaned_fraud = DataCleaner.prepare_fraud_data(fraud_data)
    cleaned_mapping = DataCleaner.prepare_ip_country_data(ip_mapping)
    return FeatureEngineer.build_fraud_model_frame(cleaned_fraud, cleaned_mapping)


def load_creditcard_dataset(data_dir: Path) -> pd.DataFrame:
    credit_data = pd.read_csv(resolve_creditcard_path(data_dir))
    return DataCleaner.prepare_creditcard_data(credit_data)


def train_dataset(
    dataset_name: str,
    frame: pd.DataFrame,
    target_column: str,
    config: PipelineConfig,
) -> DatasetTrainingResult:
    """Split, train, evaluate, and persist one dataset workflow."""
    features = frame.drop(columns=[target_column]).copy()
    target = frame[target_column].astype(int).copy()

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=target,
    )

    dataset_dir = config.artifact_dir / dataset_name
    plot_class_distribution(
        target,
        dataset_dir / "plots" / f"{dataset_name}_class_distribution.png",
        f"{dataset_name} class distribution",
    )

    return train_and_compare_models(
        dataset_name=dataset_name,
        features=features,
        target=target,
        split_data=(x_train, x_test, y_train, y_test),
        artifact_dir=config.artifact_dir,
    )


def save_manifest(results: list[DatasetTrainingResult], config: PipelineConfig) -> Path:
    """Save a project-level manifest the dashboard can read quickly."""
    manifest_path = config.artifact_dir / "project_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "project_name": "FraudPulse",
        "artifact_dir": str(config.artifact_dir),
        "datasets": {
            result.dataset_name: {
                "best_model": result.best_model_name,
                "best_model_path": str(result.best_model_path),
                "report_path": str(result.report_path),
                "feature_importance_path": str(result.feature_importance_path) if result.feature_importance_path else None,
            }
            for result in results
        },
    }

    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    return manifest_path


def run_project_pipeline(config: PipelineConfig | None = None) -> Path:
    """Execute the full fraud-detection project pipeline."""
    active_config = config or PipelineConfig()
    active_config.artifact_dir.mkdir(parents=True, exist_ok=True)

    fraud_frame = load_fraud_dataset(active_config.data_dir)
    credit_frame = load_creditcard_dataset(active_config.data_dir)

    fraud_result = train_dataset("ecommerce_fraud", fraud_frame, FRAUD_TARGET, active_config)
    credit_result = train_dataset("credit_card_fraud", credit_frame, CREDIT_TARGET, active_config)

    return save_manifest([fraud_result, credit_result], active_config)
