"""Model training, candidate model comparison, and artifact serialization pipeline."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

from src.data.preprocessing import (
    load_dataset,
    prepare_training_data,
    build_preprocessor,
    summarize_dataset_quality,
)
from src.evaluation.metrics import (
    compute_classification_metrics,
    compute_feature_importance,
)
from src.utils.config import (
    DATASET_PATH,
    MODEL_ARTIFACT_PATH,
    MODEL_METADATA_PATH,
    EVALUATION_SUMMARY_PATH,
    MODEL_COMPARISON_PATH,
    TARGET_CLASSES,
    MODEL_FEATURE_COLUMNS,
    TARGET_COLUMN,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class XGBoostStringClassifier(BaseEstimator, ClassifierMixin):
    """
    Wrapper around XGBClassifier supporting native string class labels
    in scikit-learn pipelines.
    """
    def __init__(
        self,
        n_estimators: int = 150,
        max_depth: int = 6,
        learning_rate: float = 0.08,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.encoder_ = LabelEncoder()
        self.model_ = None

    def fit(self, X, y):
        y_enc = self.encoder_.fit_transform(y)
        self.classes_ = self.encoder_.classes_
        self.model_ = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            eval_metric="mlogloss",
        )
        self.model_.fit(X, y_enc)
        return self

    def predict(self, X):
        y_pred_enc = self.model_.predict(X)
        return self.encoder_.inverse_transform(y_pred_enc)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


def get_candidate_models(random_state: int = 42) -> Dict[str, Any]:
    """Define candidate baseline and machine learning classifiers."""
    return {
        "Dummy Baseline": DummyClassifier(strategy="stratified", random_state=random_state),
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            C=1.0,
            random_state=random_state,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            class_weight="balanced",
            min_samples_split=4,
            random_state=random_state,
            n_jobs=-1,
        ),
        "XGBoost Classifier": XGBoostStringClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.08,
            random_state=random_state,
        ),
    }


def train_and_evaluate(dataset_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Execute full training workflow:
    1. Load data
    2. Validate schema and target distribution
    3. Train candidate models
    4. Evaluate and compare models
    5. Select best model based on Macro-F1 & Balanced Accuracy
    6. Persist model pipeline, metadata, and evaluation reports
    """
    path_to_use = dataset_path or DATASET_PATH
    logger.info(f"Loading dataset from: {path_to_use}")
    df = load_dataset(path_to_use)

    # Profiling data
    quality = summarize_dataset_quality(df)
    logger.info(f"Dataset summary: {quality['total_records']} rows, {quality['severity_distribution']}")

    # Prepare data split
    X_train, X_test, y_train, y_test = prepare_training_data(df, test_size=0.2, random_state=42)

    labels = sorted(list(np.unique(y_train)))
    candidate_models = get_candidate_models(random_state=42)

    comparison_records = []
    fitted_pipelines = {}
    evaluation_reports = {}

    logger.info("Beginning model training and evaluation...")
    for model_name, classifier in candidate_models.items():
        logger.info(f"Training '{model_name}'...")
        preprocessor = build_preprocessor()
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        metrics = compute_classification_metrics(y_test, y_pred, labels=labels)
        fitted_pipelines[model_name] = pipeline
        evaluation_reports[model_name] = metrics

        comparison_records.append({
            "model_name": model_name,
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
        })
        logger.info(
            f"Finished '{model_name}': Macro-F1={metrics['macro_f1']:.4f}, "
            f"Balanced Acc={metrics['balanced_accuracy']:.4f}, Accuracy={metrics['accuracy']:.4f}"
        )

    comparison_df = pd.DataFrame(comparison_records)
    # Model Selection Criterion: Sort primarily by macro_f1, then balanced_accuracy
    comparison_df = comparison_df.sort_values(by=["macro_f1", "balanced_accuracy"], ascending=False).reset_index(drop=True)
    best_model_name = comparison_df.iloc[0]["model_name"]
    best_pipeline = fitted_pipelines[best_model_name]
    best_metrics = evaluation_reports[best_model_name]

    logger.info(
        f"Selected Best Model: '{best_model_name}' based on highest Macro-F1 "
        f"({comparison_df.iloc[0]['macro_f1']:.4f}) and Balanced Accuracy ({comparison_df.iloc[0]['balanced_accuracy']:.4f})."
    )

    # Compute Feature Importance for the best model
    logger.info(f"Computing permutation importance for '{best_model_name}'...")
    importance_info = compute_feature_importance(best_pipeline, X_test, y_test)

    # Ensure output directories exist
    MODEL_ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVALUATION_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Save Best Pipeline
    joblib.dump(best_pipeline, MODEL_ARTIFACT_PATH)
    logger.info(f"Saved trained pipeline to: {MODEL_ARTIFACT_PATH}")

    # Save Comparison Table
    comparison_df.to_csv(MODEL_COMPARISON_PATH, index=False)
    logger.info(f"Saved model comparison table to: {MODEL_COMPARISON_PATH}")

    # Build and Save Model Metadata
    metadata = {
        "model_name": best_model_name,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "target_column": TARGET_COLUMN,
        "classes": list(best_pipeline.classes_),
        "features": MODEL_FEATURE_COLUMNS,
        "selection_criterion": "Highest Macro-F1 on held-out test set to prioritize equitable class recall across imbalanced severity tiers.",
        "algorithm_family": str(type(best_pipeline.named_steps["classifier"]).__name__),
    }
    with open(MODEL_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved model metadata to: {MODEL_METADATA_PATH}")

    # Build and Save Evaluation Summary
    eval_summary = {
        "selected_model": best_model_name,
        "selection_criterion": metadata["selection_criterion"],
        "metrics": best_metrics,
        "all_models_comparison": comparison_records,
        "feature_importance": importance_info,
        "dataset_profile": quality,
        "metadata": metadata,
    }
    with open(EVALUATION_SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, indent=2)
    logger.info(f"Saved evaluation summary to: {EVALUATION_SUMMARY_PATH}")

    return {
        "best_model_name": best_model_name,
        "best_metrics": best_metrics,
        "comparison": comparison_records,
        "feature_importance": importance_info,
    }


if __name__ == "__main__":
    train_and_evaluate()
