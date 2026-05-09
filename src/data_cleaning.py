from __future__ import annotations

import numpy as np
import pandas as pd


FRAUD_TARGET = "class"
CREDIT_TARGET = "Class"


class DataCleaner:
    """Utility methods for consistent dataset preparation."""

    @staticmethod
    def _copy_frame(df: pd.DataFrame) -> pd.DataFrame:
        frame = df.copy()
        frame.columns = [str(column).strip() for column in frame.columns]
        return frame

    @staticmethod
    def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicated rows while preserving the original order."""
        return df.drop_duplicates().reset_index(drop=True)

    @staticmethod
    def convert_timestamps(df: pd.DataFrame, timestamp_cols: list[str]) -> pd.DataFrame:
        """Convert timestamp columns to pandas datetime."""
        frame = df.copy()
        for column in timestamp_cols:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
        return frame

    @staticmethod
    def convert_ip_columns(df: pd.DataFrame, ip_cols: list[str]) -> pd.DataFrame:
        """Convert IP address columns to integer values compatible with joins."""
        frame = df.copy()
        max_ipv4 = np.iinfo(np.uint32).max
        for column in ip_cols:
            numeric = pd.to_numeric(frame[column], errors="coerce")
            numeric = numeric.fillna(0).clip(lower=0, upper=max_ipv4)
            frame[column] = np.floor(numeric).astype(np.int64)
        return frame

    @staticmethod
    def prepare_fraud_data(df: pd.DataFrame) -> pd.DataFrame:
        """Clean the e-commerce fraud dataset and keep it analysis-ready."""
        frame = DataCleaner._copy_frame(df)
        frame = DataCleaner.drop_duplicates(frame)
        frame = DataCleaner.convert_timestamps(frame, ["signup_time", "purchase_time"])

        numeric_columns = ["user_id", "purchase_value", "age", "ip_address", FRAUD_TARGET]
        for column in numeric_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

        frame = DataCleaner.convert_ip_columns(frame, ["ip_address"])
        frame[FRAUD_TARGET] = frame[FRAUD_TARGET].fillna(0).astype(int)

        required_columns = ["signup_time", "purchase_time", "purchase_value", "age", "ip_address"]
        frame = frame.dropna(subset=required_columns).reset_index(drop=True)
        frame["device_id"] = frame["device_id"].fillna("UNKNOWN_DEVICE").astype(str)
        frame["source"] = frame["source"].fillna("Unknown").astype(str)
        frame["browser"] = frame["browser"].fillna("Unknown").astype(str)
        frame["sex"] = frame["sex"].fillna("Unknown").astype(str)

        return frame

    @staticmethod
    def prepare_ip_country_data(df: pd.DataFrame) -> pd.DataFrame:
        """Clean and sort the IP-to-country mapping table."""
        frame = DataCleaner._copy_frame(df)
        frame = DataCleaner.drop_duplicates(frame)
        for column in ["lower_bound_ip_address", "upper_bound_ip_address"]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame.dropna(subset=["lower_bound_ip_address", "upper_bound_ip_address", "country"])
        frame["lower_bound_ip_address"] = frame["lower_bound_ip_address"].astype(np.int64)
        frame["upper_bound_ip_address"] = frame["upper_bound_ip_address"].astype(np.int64)
        frame["country"] = frame["country"].astype(str)
        return frame.sort_values("lower_bound_ip_address").reset_index(drop=True)

    @staticmethod
    def prepare_creditcard_data(df: pd.DataFrame) -> pd.DataFrame:
        """Clean the credit-card fraud dataset."""
        frame = DataCleaner._copy_frame(df)
        frame = DataCleaner.drop_duplicates(frame)

        for column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

        frame = frame.dropna(subset=[CREDIT_TARGET]).reset_index(drop=True)
        frame[CREDIT_TARGET] = frame[CREDIT_TARGET].astype(int)
        frame = frame.fillna(frame.median(numeric_only=True))

        return frame
