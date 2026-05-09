from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils import resample

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from imblearn.over_sampling import SMOTE
except ImportError:  # pragma: no cover
    SMOTE = None

try:
    import shap
except ImportError:  # pragma: no cover
    shap = None


RANDOM_STATE = 42


@dataclass
class DatasetTrainingResult:
    """Summary of a dataset training run."""

    dataset_name: str
    best_model_name: str
    best_model_path: Path
    report_path: Path
    feature_importance_path: Path | None


@dataclass
class ModelBundle:
    """Serializable wrapper for preprocessing plus classifier inference."""

    preprocessor: ColumnTransformer
    classifier: Any
    feature_columns: list[str]

    def _align_features(self, features: pd.DataFrame) -> pd.DataFrame:
        aligned = features.copy()
        for column in self.feature_columns:
            if column not in aligned.columns:
                aligned[column] = pd.NA
        return aligned[self.feature_columns]

    def transform(self, features: pd.DataFrame) -> np.ndarray:
        aligned = self._align_features(features)
        return self.preprocessor.transform(aligned)

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        transformed = self.transform(features)
        return self.classifier.predict_proba(transformed)

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        transformed = self.transform(features)
        return self.classifier.predict(transformed)


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Create a preprocessing graph based on the input feature types."""
    categorical_columns = features.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric_columns = [column for column in features.columns if column not in categorical_columns]

    transformers: list[tuple[str, Pipeline, list[str]]] = []

    if numeric_columns:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            )
        )

    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_columns,
            )
        )

    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_candidate_models() -> dict[str, Any]:
    """Return the candidate classifiers required by the challenge."""
    return {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            solver="liblinear",
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=160,
            max_depth=14,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def balance_training_data(features: np.ndarray, target: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Balance the training set with SMOTE when available, else random oversampling."""
    minority_count = int(target.sum())
    if minority_count > 1 and SMOTE is not None:
        neighbors = min(5, minority_count - 1)
        sampler = SMOTE(random_state=RANDOM_STATE, k_neighbors=neighbors)
        balanced_features, balanced_target = sampler.fit_resample(features, target)
        return balanced_features, balanced_target

    target_array = target.to_numpy()
    classes, counts = np.unique(target_array, return_counts=True)
    if len(classes) < 2:
        return features, target_array

    majority_class = classes[np.argmax(counts)]
    minority_class = classes[np.argmin(counts)]

    majority_features = features[target_array == majority_class]
    minority_features = features[target_array == minority_class]

    if len(minority_features) == 0:
        return features, target_array

    oversampled_minority = resample(
        minority_features,
        replace=True,
        n_samples=len(majority_features),
        random_state=RANDOM_STATE,
    )

    balanced_features = np.vstack([majority_features, oversampled_minority])
    balanced_target = np.concatenate(
        [
            np.full(len(majority_features), majority_class),
            np.full(len(oversampled_minority), minority_class),
        ]
    )
    return balanced_features, balanced_target


def fit_model_bundle(
    classifier: Any,
    x_train: pd.DataFrame,
    y_train: pd.Series,
) -> ModelBundle:
    """Fit the preprocessing stack and classifier, then wrap them together."""
    preprocessor = build_preprocessor(x_train)
    x_train_processed = preprocessor.fit_transform(x_train)
    x_balanced, y_balanced = balance_training_data(x_train_processed, y_train)
    classifier.fit(x_balanced, y_balanced)

    return ModelBundle(
        preprocessor=preprocessor,
        classifier=classifier,
        feature_columns=x_train.columns.tolist(),
    )


def evaluate_classifier(model: ModelBundle, features: pd.DataFrame, target: pd.Series) -> dict[str, Any]:
    """Compute core metrics for imbalanced fraud classification."""
    probabilities = model.predict_proba(features)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    precision_curve, recall_curve, _ = precision_recall_curve(target, probabilities)
    fpr, tpr, _ = roc_curve(target, probabilities)
    matrix = confusion_matrix(target, predictions)

    return {
        "roc_auc": float(roc_auc_score(target, probabilities)),
        "pr_auc": float(average_precision_score(target, probabilities)),
        "f1": float(f1_score(target, predictions, zero_division=0)),
        "precision": float(precision_score(target, predictions, zero_division=0)),
        "recall": float(recall_score(target, predictions, zero_division=0)),
        "support": int(len(target)),
        "fraud_rate": float(target.mean()),
        "confusion_matrix": matrix.tolist(),
        "roc_curve": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
        },
        "pr_curve": {
            "precision": precision_curve.tolist(),
            "recall": recall_curve.tolist(),
        },
    }


def plot_evaluation_artifacts(
    dataset_name: str,
    model_name: str,
    metrics: dict[str, Any],
    plot_dir: Path,
) -> dict[str, str]:
    """Persist confusion matrix, ROC, and PR plots for one model."""
    plot_dir.mkdir(parents=True, exist_ok=True)
    generated_paths: dict[str, str] = {}

    matrix = np.array(metrics["confusion_matrix"])
    figure, axis = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(confusion_matrix=matrix).plot(ax=axis, colorbar=False)
    axis.set_title(f"{dataset_name} - {model_name} confusion matrix")
    figure.tight_layout()
    confusion_path = plot_dir / f"{dataset_name}_{model_name}_confusion_matrix.png"
    figure.savefig(confusion_path, dpi=200)
    plt.close(figure)
    generated_paths["confusion_matrix"] = str(confusion_path)

    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot(metrics["roc_curve"]["fpr"], metrics["roc_curve"]["tpr"], label=f"ROC AUC = {metrics['roc_auc']:.3f}")
    axis.plot([0, 1], [0, 1], linestyle="--", color="grey")
    axis.set_title(f"{dataset_name} - {model_name} ROC curve")
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.legend(loc="lower right")
    figure.tight_layout()
    roc_path = plot_dir / f"{dataset_name}_{model_name}_roc_curve.png"
    figure.savefig(roc_path, dpi=200)
    plt.close(figure)
    generated_paths["roc_curve"] = str(roc_path)

    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot(metrics["pr_curve"]["recall"], metrics["pr_curve"]["precision"], label=f"PR AUC = {metrics['pr_auc']:.3f}")
    axis.set_title(f"{dataset_name} - {model_name} precision-recall curve")
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.legend(loc="lower left")
    figure.tight_layout()
    pr_path = plot_dir / f"{dataset_name}_{model_name}_pr_curve.png"
    figure.savefig(pr_path, dpi=200)
    plt.close(figure)
    generated_paths["pr_curve"] = str(pr_path)

    return generated_paths


def _clean_feature_names(feature_names: np.ndarray) -> list[str]:
    return [name.split("__", 1)[-1] for name in feature_names.tolist()]


def generate_feature_importance_table(model: ModelBundle, sample: pd.DataFrame) -> pd.DataFrame:
    """Generate a feature-importance table from the fitted classifier."""
    feature_names = _clean_feature_names(model.preprocessor.get_feature_names_out())

    if hasattr(model.classifier, "feature_importances_"):
        importance_values = model.classifier.feature_importances_
    else:
        importance_values = np.abs(model.classifier.coef_).ravel()

    table = pd.DataFrame({"feature": feature_names, "importance": importance_values})
    return table.sort_values("importance", ascending=False).reset_index(drop=True)


def generate_shap_artifacts(
    model: ModelBundle,
    sample: pd.DataFrame,
    output_dir: Path,
    dataset_name: str,
) -> Path | None:
    """Create SHAP plots and a SHAP-based feature ranking if possible."""
    if shap is None:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    transformed = model.transform(sample)
    feature_names = _clean_feature_names(model.preprocessor.get_feature_names_out())
    background = transformed[: min(len(transformed), 150)]
    explain_target = transformed[: min(len(transformed), 300)]

    try:
        explainer = shap.Explainer(model.classifier, background, feature_names=feature_names)
        explanation = explainer(explain_target)
        if len(explanation.shape) == 3:
            explanation = explanation[..., 1]

        figure_path = output_dir / f"{dataset_name}_shap_summary.png"
        plt.figure(figsize=(9, 6))
        shap.plots.beeswarm(explanation, max_display=15, show=False)
        plt.tight_layout()
        plt.savefig(figure_path, dpi=200, bbox_inches="tight")
        plt.close()

        values = explanation.values if hasattr(explanation, "values") else np.asarray(explanation)
        importance = np.abs(values).mean(axis=0)
        shap_table = pd.DataFrame({"feature": feature_names, "importance": importance})
        shap_table.sort_values("importance", ascending=False).to_csv(
            output_dir / f"{dataset_name}_shap_importance.csv",
            index=False,
        )

        return figure_path
    except Exception:
        return None


def save_json(payload: dict[str, Any], output_path: Path) -> None:
    """Serialize a JSON report with stable formatting."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def train_and_compare_models(
    dataset_name: str,
    features: pd.DataFrame,
    target: pd.Series,
    split_data: tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
    artifact_dir: Path,
) -> DatasetTrainingResult:
    """Train both required models, compare them, and persist the outputs."""
    x_train, x_test, y_train, y_test = split_data
    dataset_dir = artifact_dir / dataset_name
    model_dir = dataset_dir / "models"
    plot_dir = dataset_dir / "plots"
    report_dir = dataset_dir / "reports"

    model_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    metrics_by_model: dict[str, dict[str, Any]] = {}
    saved_model_paths: dict[str, str] = {}
    plot_paths: dict[str, dict[str, str]] = {}

    for model_name, classifier in build_candidate_models().items():
        model_bundle = fit_model_bundle(classifier, x_train, y_train)

        metrics = evaluate_classifier(model_bundle, x_test, y_test)
        metrics_by_model[model_name] = metrics
        plot_paths[model_name] = plot_evaluation_artifacts(dataset_name, model_name, metrics, plot_dir)

        model_path = model_dir / f"{model_name}.joblib"
        joblib.dump(model_bundle, model_path)
        saved_model_paths[model_name] = str(model_path)

    best_model_name = max(
        metrics_by_model,
        key=lambda name: (
            metrics_by_model[name]["pr_auc"],
            metrics_by_model[name]["f1"],
            metrics_by_model[name]["roc_auc"],
            metrics_by_model[name]["precision"],
        ),
    )
    best_model_path = Path(saved_model_paths[best_model_name])
    best_model = joblib.load(best_model_path)

    importance_table = generate_feature_importance_table(best_model, x_test.head(min(len(x_test), 2000)))
    feature_importance_path = report_dir / f"{dataset_name}_feature_importance.csv"
    importance_table.to_csv(feature_importance_path, index=False)

    shap_path = generate_shap_artifacts(
        best_model,
        x_test.sample(min(len(x_test), 300), random_state=RANDOM_STATE),
        plot_dir,
        dataset_name,
    )

    report_payload = {
        "dataset_name": dataset_name,
        "feature_columns": features.columns.tolist(),
        "target_name": target.name,
        "row_count": int(len(features)),
        "class_distribution": {
            "non_fraud": int((target == 0).sum()),
            "fraud": int((target == 1).sum()),
            "fraud_rate": float(target.mean()),
        },
        "best_model": best_model_name,
        "best_model_path": str(best_model_path),
        "metrics_by_model": metrics_by_model,
        "model_paths": saved_model_paths,
        "plot_paths": plot_paths,
        "feature_importance_path": str(feature_importance_path),
        "shap_summary_path": str(shap_path) if shap_path else None,
    }
    report_path = report_dir / f"{dataset_name}_report.json"
    save_json(report_payload, report_path)

    return DatasetTrainingResult(
        dataset_name=dataset_name,
        best_model_name=best_model_name,
        best_model_path=best_model_path,
        report_path=report_path,
        feature_importance_path=feature_importance_path,
    )
