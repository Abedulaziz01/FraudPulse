from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_cleaning import DataCleaner
from src.feature_engineering import FeatureEngineer


st.set_page_config(page_title="FraudPulse Dashboard", layout="wide")


ECOMMERCE_SOURCE_FEATURES = [
    ("user_id", "Unique identifier for the customer."),
    ("signup_time", "Timestamp when the account was created."),
    ("purchase_time", "Timestamp when the transaction happened."),
    ("purchase_value", "Dollar value of the transaction."),
    ("device_id", "Identifier for the device used to purchase."),
    ("source", "Traffic source such as SEO or Ads."),
    ("browser", "Browser used during the transaction."),
    ("sex", "Customer gender field from the dataset."),
    ("age", "Customer age."),
    ("ip_address", "Original IP address used for the transaction."),
    ("class", "Fraud target label in the raw dataset."),
]

ECOMMERCE_ENGINEERED_FEATURES = [
    ("country", "Derived by mapping IP ranges to countries for geolocation analysis."),
    ("hour_of_day", "Hour extracted from purchase time."),
    ("day_of_week", "Weekday extracted from purchase time."),
    ("time_since_signup_hours", "Hours between signup and purchase."),
    ("time_since_signup_days", "Days between signup and purchase."),
    ("transaction_count", "Transaction frequency per user."),
    ("user_average_purchase", "Average purchase value for the same user."),
    ("device_shared_users", "How many users share the same device."),
    ("seconds_since_previous_purchase", "Gap from the previous transaction for that user."),
    ("purchase_value_to_user_mean", "Current purchase value relative to the user's average."),
    ("velocity", "Purchase value divided by time since signup."),
]

CREDITCARD_FEATURES = [
    ("Time", "Elapsed seconds since the first transaction in the dataset."),
    ("V1-V28", "Anonymized PCA-transformed input features."),
    ("Amount", "Transaction amount in dollars."),
    ("Class", "Fraud target label in the raw dataset."),
]


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_local_image(path_value: str | None) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    return path if path.exists() else None


def prepare_uploaded_frame(dataset_name: str, raw_frame: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    if dataset_name == "ecommerce_fraud":
        ip_mapping_path = data_dir / "IpAddress_to_Country.csv"
        mapping = DataCleaner.prepare_ip_country_data(pd.read_csv(ip_mapping_path))
        cleaned = DataCleaner.prepare_fraud_data(raw_frame)
        features = FeatureEngineer.build_fraud_model_frame(cleaned, mapping)
        if "class" in features.columns:
            features = features.drop(columns=["class"])
        return features

    cleaned = DataCleaner.prepare_creditcard_data(raw_frame)
    if "Class" in cleaned.columns:
        cleaned = cleaned.drop(columns=["Class"])
    return cleaned


def align_features(frame: pd.DataFrame, expected_columns: list[str]) -> pd.DataFrame:
    aligned = frame.copy()
    for column in expected_columns:
        if column not in aligned.columns:
            aligned[column] = pd.NA
    return aligned[expected_columns]


def render_feature_catalog(dataset_name: str, report: dict) -> None:
    st.subheader("Feature Catalog")

    if dataset_name == "ecommerce_fraud":
        source_df = pd.DataFrame(ECOMMERCE_SOURCE_FEATURES, columns=["feature", "description"])
        engineered_df = pd.DataFrame(ECOMMERCE_ENGINEERED_FEATURES, columns=["feature", "description"])

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Raw Task-Document Fields**")
            st.dataframe(source_df, use_container_width=True, hide_index=True)
        with col2:
            st.markdown("**Engineered Fraud Features**")
            st.dataframe(engineered_df, use_container_width=True, hide_index=True)

        st.caption(
            "The dashboard model uses the engineered features, including transaction frequency, velocity, "
            "hour_of_day, day_of_week, time_since_signup, and IP-to-country mapping requested in the task document."
        )
        return

    feature_rows = []
    for feature_name, description in CREDITCARD_FEATURES:
        if feature_name == "V1-V28":
            for index in range(1, 29):
                feature_rows.append((f"V{index}", description))
        else:
            feature_rows.append((feature_name, description))

    credit_df = pd.DataFrame(feature_rows, columns=["feature", "description"])
    st.dataframe(credit_df, use_container_width=True, hide_index=True)
    st.caption("The credit-card workflow uses the original anonymized task-document features plus Amount and Time.")


def render_summary(dataset_name: str, report: dict) -> None:
    best_metrics = report["metrics_by_model"][report["best_model"]]
    fraud_rate = report["class_distribution"]["fraud_rate"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Dataset", dataset_name.replace("_", " ").title())
    col2.metric("Best Model", report["best_model"].replace("_", " ").title())
    col3.metric("AUC-PR", f"{best_metrics['pr_auc']:.3f}")
    col4.metric("Fraud Rate", f"{fraud_rate:.2%}")


def render_plots(report: dict) -> None:
    best_model = report["best_model"]
    model_plots = report["plot_paths"][best_model]

    st.subheader("Evaluation Plots")
    tabs = st.tabs(["Class Balance", "Confusion Matrix", "ROC Curve", "PR Curve", "SHAP"])

    class_plot = Path(model_plots["confusion_matrix"]).parent / f"{report['dataset_name']}_class_distribution.png"
    confusion_plot = read_local_image(model_plots.get("confusion_matrix"))
    roc_plot = read_local_image(model_plots.get("roc_curve"))
    pr_plot = read_local_image(model_plots.get("pr_curve"))
    shap_plot = read_local_image(report.get("shap_summary_path"))

    with tabs[0]:
        if class_plot.exists():
            st.image(str(class_plot), use_container_width=True)
        else:
            st.info("Class balance plot not found yet.")
    with tabs[1]:
        if confusion_plot:
            st.image(str(confusion_plot), use_container_width=True)
    with tabs[2]:
        if roc_plot:
            st.image(str(roc_plot), use_container_width=True)
    with tabs[3]:
        if pr_plot:
            st.image(str(pr_plot), use_container_width=True)
    with tabs[4]:
        if shap_plot:
            st.image(str(shap_plot), use_container_width=True)
        else:
            st.info("SHAP output was not generated. Install the full requirements and rerun the pipeline.")


def render_feature_importance(report: dict) -> None:
    feature_path = Path(report["feature_importance_path"])
    if not feature_path.exists():
        st.info("Feature importance file is missing.")
        return

    importance = pd.read_csv(feature_path).head(15)
    chart = px.bar(
        importance.sort_values("importance"),
        x="importance",
        y="feature",
        orientation="h",
        title="Top model drivers",
        color="importance",
        color_continuous_scale="Tealgrn",
    )
    st.plotly_chart(chart, use_container_width=True)


def render_batch_scoring(dataset_name: str, report: dict, data_dir: Path) -> None:
    st.subheader("Batch Scoring")
    uploaded_file = st.file_uploader(
        "Upload a CSV file to score transactions with the saved best model.",
        type="csv",
        key=f"uploader_{dataset_name}",
    )

    if not uploaded_file:
        st.caption("Upload raw rows that follow the original dataset schema.")
        return

    raw_frame = pd.read_csv(uploaded_file)
    features = prepare_uploaded_frame(dataset_name, raw_frame, data_dir)
    features = align_features(features, report["feature_columns"])

    model = joblib.load(report["best_model_path"])
    probabilities = model.predict_proba(features)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    results = raw_frame.copy()
    results["fraud_probability"] = probabilities
    results["predicted_class"] = predictions

    st.dataframe(results.head(100), use_container_width=True)
    st.plotly_chart(
        px.histogram(results, x="fraud_probability", nbins=25, title="Predicted fraud probability distribution"),
        use_container_width=True,
    )


def main() -> None:
    st.title("FraudPulse")
    st.caption("Fraud analytics for e-commerce and bank transactions.")

    artifact_dir = Path(st.sidebar.text_input("Artifact directory", "artifacts"))
    data_dir = Path(st.sidebar.text_input("Data directory", "data"))
    manifest_path = artifact_dir / "project_manifest.json"

    if not manifest_path.exists():
        st.warning(
            "Run `python -m src.preprocess_fraud_data --data-dir data --artifact-dir artifacts` first to create reports."
        )
        return

    manifest = load_json(manifest_path)
    dataset_name = st.sidebar.selectbox("Dataset", list(manifest["datasets"].keys()))
    report_path = Path(manifest["datasets"][dataset_name]["report_path"])
    report = load_json(report_path)

    render_summary(dataset_name, report)
    render_feature_catalog(dataset_name, report)
    render_plots(report)

    st.subheader("Model Comparison")
    comparison = pd.DataFrame(report["metrics_by_model"]).T.reset_index().rename(columns={"index": "model"})
    st.dataframe(
        comparison[["model", "pr_auc", "roc_auc", "f1", "precision", "recall"]],
        use_container_width=True,
    )

    render_feature_importance(report)
    render_batch_scoring(dataset_name, report, data_dir)


if __name__ == "__main__":
    main()
