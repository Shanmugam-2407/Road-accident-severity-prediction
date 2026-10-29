"""Data loading, validation, quality profiling, and pipeline preprocessing."""

import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.feature_engineering import (
    engineer_features,
    normalize_target_severity,
)
from src.utils.config import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    MODEL_FEATURE_COLUMNS,
    TARGET_CLASSES,
    TARGET_COLUMN,
)

logger = logging.getLogger(__name__)


class DatasetValidationError(ValueError):
    """Raised when an input dataset fails schema or data integrity validation."""
    pass


def load_dataset(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load accident dataset from a CSV file with integrity checks.
    
    Raises:
        DatasetValidationError: If file does not exist, is empty, or cannot be parsed.
    """
    path = Path(file_path)
    if not path.is_file():
        raise DatasetValidationError(f"Dataset file not found at path: {file_path}")

    if path.stat().st_size == 0:
        raise DatasetValidationError(f"Dataset '{file_path}' is completely empty (0 bytes).")

    try:
        df = pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError:
        raise DatasetValidationError(f"Dataset '{file_path}' is empty: No columns to parse.")
    except Exception as e:
        raise DatasetValidationError(f"Failed to read CSV file '{file_path}': {str(e)}")

    if df.empty:
        raise DatasetValidationError(f"Dataset '{file_path}' is completely empty (0 rows).")

    return df


def validate_dataset_schema(
    df: pd.DataFrame,
    require_target: bool = True,
    target_column: str = TARGET_COLUMN
) -> Dict[str, Any]:
    """
    Validate that the DataFrame satisfies basic schema requirements.
    
    Returns:
        Validation report dictionary with status, errors, and warnings.
    """
    errors = []
    warnings = []

    if df is None or df.empty:
        errors.append("Dataset is empty or None.")
        return {"is_valid": False, "errors": errors, "warnings": warnings}

    # Validate target column if required
    if require_target:
        if target_column not in df.columns:
            errors.append(f"Required target column '{target_column}' is missing from dataset.")
        else:
            valid_targets = df[target_column].dropna()
            if valid_targets.empty:
                errors.append(f"Target column '{target_column}' contains only missing/null values.")
            else:
                normalized = normalize_target_severity(valid_targets)
                valid_count = normalized.notna().sum()
                if valid_count == 0:
                    errors.append(
                        f"Target column '{target_column}' contains no recognized severity categories. "
                        f"Expected one of: {TARGET_CLASSES}."
                    )
                elif normalized.nunique() < 2:
                    warnings.append(
                        f"Target column '{target_column}' contains only {normalized.nunique()} unique class. "
                        f"Classification requires at least 2 distinct classes."
                    )

    # Check available features
    available_model_features = [col for col in MODEL_FEATURE_COLUMNS if col in df.columns]
    if len(available_model_features) == 0:
        # Check if alternative raw columns can construct features
        fallback_candidates = ["speed_limit", "road_type", "weather_conditions", "time", "accident_time"]
        matched_fallbacks = [c for c in fallback_candidates if c in df.columns]
        if not matched_fallbacks:
            errors.append("Dataset does not contain any recognized feature columns for accident analysis.")

    return {
        "is_valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "rows": len(df),
        "columns": list(df.columns),
    }


def summarize_dataset_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate a comprehensive data quality summary for reporting and dashboards.
    """
    total_rows = len(df)
    total_cols = len(df.columns)
    missing_by_col = df.isnull().sum().to_dict()
    total_missing_cells = int(df.isnull().sum().sum())
    missing_pct = round((total_missing_cells / (total_rows * total_cols)) * 100, 2) if total_rows * total_cols > 0 else 0.0

    duplicate_rows = int(df.duplicated().sum())

    severity_counts = {}
    if TARGET_COLUMN in df.columns:
        normalized_target = normalize_target_severity(df[TARGET_COLUMN])
        severity_counts = normalized_target.value_counts(dropna=False).to_dict()
        severity_counts = {str(k): int(v) for k, v in severity_counts.items()}

    has_coords = "latitude" in df.columns and "longitude" in df.columns
    valid_coords_count = 0
    if has_coords:
        valid_coords = df["latitude"].notna() & df["longitude"].notna()
        valid_coords_count = int(valid_coords.sum())

    return {
        "total_records": total_rows,
        "total_columns": total_cols,
        "total_missing_cells": total_missing_cells,
        "missing_percentage": missing_pct,
        "duplicate_records": duplicate_rows,
        "severity_distribution": severity_counts,
        "has_geographic_coordinates": has_coords,
        "valid_coordinates_count": valid_coords_count,
        "column_missing_counts": {k: int(v) for k, v in missing_by_col.items() if v > 0},
    }


def build_preprocessor() -> ColumnTransformer:
    """
    Construct a scikit-learn ColumnTransformer for feature preprocessing.
    
    - Numerical: Median imputation + Standard scaling.
    - Categorical: Most frequent imputation + OneHotEncoder with handle_unknown='ignore'.
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERICAL_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


def prepare_training_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Validate, clean, normalize, and split dataset into X_train, X_test, y_train, y_test.
    
    Prevents data leakage by removing IDs, timestamps, and post-outcome casualty counts.
    """
    validation = validate_dataset_schema(df, require_target=True)
    if not validation["is_valid"]:
        raise DatasetValidationError(f"Invalid dataset schema: {'; '.join(validation['errors'])}")

    # Clean target
    clean_df = df.copy()
    clean_df["clean_target"] = normalize_target_severity(clean_df[TARGET_COLUMN])
    clean_df = clean_df.dropna(subset=["clean_target"])

    if len(clean_df) < 20:
        raise DatasetValidationError(f"Dataset has only {len(clean_df)} valid labeled rows; at least 20 required.")

    # Engineer model features
    X = engineer_features(clean_df, is_training=True)
    y = clean_df["clean_target"].copy()

    # Determine stratification
    stratify_target = y if stratify and y.value_counts().min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_target,
    )

    logger.info(
        f"Data prepared successfully: X_train shape={X_train.shape}, X_test shape={X_test.shape}, "
        f"classes={list(y.unique())}"
    )
    return X_train, X_test, y_train, y_test
