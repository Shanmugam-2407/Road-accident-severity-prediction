"""Feature engineering and schema alignment for RoadGuard AI."""

import logging
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from src.utils.config import (
    MODEL_FEATURE_COLUMNS,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    LEAKAGE_AND_EXCLUDED_COLUMNS,
    TARGET_COLUMN,
    SEVERITY_MAPPING,
)

logger = logging.getLogger(__name__)


def extract_accident_hour(df: pd.DataFrame) -> pd.Series:
    """
    Extract accident hour (0-23) from time/datetime columns.
    
    Checks in order:
    1. 'accident_hour' (already present)
    2. 'accident_time' or 'time' (formatted as HH:MM or HH:MM:SS)
    3. 'accident_date' or 'date' (if datetime with time component)
    Defaults to 12 if absent or unparseable.
    """
    if "accident_hour" in df.columns:
        return pd.to_numeric(df["accident_hour"], errors="coerce").fillna(12).clip(0, 23).astype(int)

    time_col = None
    for candidate in ["accident_time", "time", "Time"]:
        if candidate in df.columns:
            time_col = candidate
            break

    if time_col is not None:
        def parse_hour(val: Any) -> int:
            if pd.isna(val):
                return 12
            val_str = str(val).strip()
            if ":" in val_str:
                parts = val_str.split(":")
                try:
                    h = int(parts[0])
                    return max(0, min(23, h))
                except (ValueError, IndexError):
                    return 12
            try:
                h = int(float(val_str))
                return max(0, min(23, h))
            except ValueError:
                return 12

        return df[time_col].apply(parse_hour).astype(int)

    # Check datetime columns
    for date_candidate in ["accident_date", "date", "Date", "datetime"]:
        if date_candidate in df.columns:
            try:
                parsed_dt = pd.to_datetime(df[date_candidate], errors="coerce")
                if parsed_dt.notna().any():
                    return parsed_dt.dt.hour.fillna(12).astype(int)
            except Exception:
                pass

    # Fallback default hour (midday)
    return pd.Series(12, index=df.index, dtype=int)


def extract_day_of_week(df: pd.DataFrame) -> pd.Series:
    """
    Extract or standardize day of the week ('Monday' through 'Sunday').
    """
    for col in ["day_of_week", "Day_of_Week", "day"]:
        if col in df.columns:
            return df[col].astype(str).str.strip().str.capitalize()

    for date_candidate in ["accident_date", "date", "Date"]:
        if date_candidate in df.columns:
            try:
                parsed_dt = pd.to_datetime(df[date_candidate], errors="coerce")
                if parsed_dt.notna().any():
                    return parsed_dt.dt.day_name().fillna("Friday")
            except Exception:
                pass

    return pd.Series("Friday", index=df.index, dtype=str)


def normalize_target_severity(severity_series: pd.Series) -> pd.Series:
    """
    Normalize accident severity target values into canonical classes:
    'Slight', 'Serious', 'Fatal'.
    """
    return severity_series.map(lambda v: SEVERITY_MAPPING.get(v, SEVERITY_MAPPING.get(str(v).strip(), None)))


def engineer_features(df: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
    """
    Derive engineered features while strictly isolating training and inference pipelines.
    
    Guarantees:
    - Excludes target and leakage columns from feature set.
    - Creates 'accident_hour' if not directly provided.
    - Standardizes 'day_of_week'.
    - Returns a clean DataFrame containing exactly MODEL_FEATURE_COLUMNS.
    """
    out = df.copy()

    # Extract hour
    if "accident_hour" not in out.columns:
        out["accident_hour"] = extract_accident_hour(out)
    else:
        out["accident_hour"] = pd.to_numeric(out["accident_hour"], errors="coerce").fillna(12).clip(0, 23).astype(int)

    # Standardize day of week
    if "day_of_week" not in out.columns or out["day_of_week"].isna().any():
        out["day_of_week"] = extract_day_of_week(out)
    else:
        out["day_of_week"] = out["day_of_week"].astype(str).str.strip().str.capitalize()

    # Clean numerical columns
    if "number_of_vehicles" in out.columns:
        out["number_of_vehicles"] = pd.to_numeric(out["number_of_vehicles"], errors="coerce").fillna(1).clip(lower=1)
    else:
        out["number_of_vehicles"] = 1

    if "speed_limit" in out.columns:
        out["speed_limit"] = pd.to_numeric(out["speed_limit"], errors="coerce").fillna(30).clip(lower=10, upper=100)
    else:
        out["speed_limit"] = 30

    # Ensure all required categorical columns exist with string type
    for cat_col in CATEGORICAL_FEATURES:
        if cat_col in out.columns:
            out[cat_col] = out[cat_col].astype(str).str.strip()
        else:
            out[cat_col] = "Missing"

    # Select strictly the permitted model feature columns
    selected_features = out[MODEL_FEATURE_COLUMNS].copy()
    return selected_features

