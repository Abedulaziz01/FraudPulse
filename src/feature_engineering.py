from __future__ import annotations

import numpy as np
import pandas as pd


class FeatureEngineer:
    """Feature engineering helpers for the fraud detection workflows."""

    @staticmethod
    def map_ip_to_country(
        df: pd.DataFrame,
        ip_col: str,
        ip_mapping: pd.DataFrame,
    ) -> pd.DataFrame:
        """Attach a country value to each transaction using IP ranges."""
        frame = df.copy().reset_index(drop=True)
        frame["_row_id"] = np.arange(len(frame))

        mapping = ip_mapping.copy().sort_values("lower_bound_ip_address").reset_index(drop=True)

        merged = pd.merge_asof(
            frame.sort_values(ip_col),
            mapping,
            left_on=ip_col,
            right_on="lower_bound_ip_address",
            direction="backward",
        )

        is_valid_match = merged["upper_bound_ip_address"].fillna(-1) >= merged[ip_col]
        merged["country"] = np.where(is_valid_match, merged["country"].fillna("Unknown"), "Unknown")

        merged = merged.sort_values("_row_id").drop(columns=["_row_id"])
        return merged.reset_index(drop=True)

    @staticmethod
    def add_time_features(
        df: pd.DataFrame,
        purchase_time_col: str,
        signup_time_col: str,
    ) -> pd.DataFrame:
        """Create time-based signals from timestamps."""
        frame = df.copy()
        elapsed_seconds = (
            frame[purchase_time_col] - frame[signup_time_col]
        ).dt.total_seconds()
        frame["hour_of_day"] = frame[purchase_time_col].dt.hour.astype(int)
        frame["day_of_week"] = frame[purchase_time_col].dt.day_name()
        frame["time_since_signup_hours"] = elapsed_seconds.clip(lower=0) / 3600.0
        frame["time_since_signup_days"] = elapsed_seconds.clip(lower=0) / 86400.0
        return frame

    @staticmethod
    def add_behavioral_features(
        df: pd.DataFrame,
        user_id_col: str,
        device_id_col: str,
        purchase_value_col: str,
        purchase_time_col: str,
        time_since_signup_col: str,
    ) -> pd.DataFrame:
        """Create user-level and device-level behavior indicators."""
        frame = df.copy().reset_index(drop=True)
        frame["_row_id"] = np.arange(len(frame))

        ordered = frame.sort_values([user_id_col, purchase_time_col]).copy()
        ordered["transaction_count"] = ordered.groupby(user_id_col).cumcount() + 1
        ordered["user_average_purchase"] = ordered.groupby(user_id_col)[purchase_value_col].transform("mean")
        ordered["device_shared_users"] = ordered.groupby(device_id_col)[user_id_col].transform("nunique")
        ordered["seconds_since_previous_purchase"] = (
            ordered.groupby(user_id_col)[purchase_time_col]
            .diff()
            .dt.total_seconds()
            .fillna(0)
        )

        denominator = ordered["user_average_purchase"].replace(0, np.nan)
        ordered["purchase_value_to_user_mean"] = (
            ordered[purchase_value_col] / denominator
        ).replace([np.inf, -np.inf], np.nan).fillna(1.0)

        ordered["velocity"] = ordered[purchase_value_col] / (ordered[time_since_signup_col] + 1.0)

        ordered = ordered.sort_values("_row_id").drop(columns=["_row_id"])
        return ordered.reset_index(drop=True)

    @staticmethod
    def build_fraud_model_frame(
        fraud_df: pd.DataFrame,
        ip_mapping: pd.DataFrame,
    ) -> pd.DataFrame:
        """Build the final model-ready e-commerce fraud dataset."""
        frame = FeatureEngineer.map_ip_to_country(fraud_df, "ip_address", ip_mapping)
        frame = FeatureEngineer.add_time_features(frame, "purchase_time", "signup_time")
        frame = FeatureEngineer.add_behavioral_features(
            frame,
            user_id_col="user_id",
            device_id_col="device_id",
            purchase_value_col="purchase_value",
            purchase_time_col="purchase_time",
            time_since_signup_col="time_since_signup_hours",
        )

        frame["country"] = frame["country"].fillna("Unknown").astype(str)
        frame["source"] = frame["source"].fillna("Unknown").astype(str)
        frame["browser"] = frame["browser"].fillna("Unknown").astype(str)
        frame["sex"] = frame["sex"].fillna("Unknown").astype(str)

        selected_columns = [
            "purchase_value",
            "age",
            "source",
            "browser",
            "sex",
            "country",
            "hour_of_day",
            "day_of_week",
            "time_since_signup_hours",
            "time_since_signup_days",
            "transaction_count",
            "user_average_purchase",
            "device_shared_users",
            "seconds_since_previous_purchase",
            "purchase_value_to_user_mean",
            "velocity",
            "class",
        ]
        return frame[selected_columns].copy()
