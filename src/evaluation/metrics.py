"""Model evaluation metrics, confusion matrix computation, and feature importance."""

import logging
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

logger = logging.getLogger(__name__)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Calculate comprehensive multi-class classification metrics.
    
    Handles class imbalance using balanced accuracy, macro and weighted averages.
    """
    if labels is None:
        labels = sorted(list(set(y_true) | set(y_pred)))

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))

    prec_macro = float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    prec_weighted = float(precision_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0))

    rec_macro = float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    rec_weighted = float(recall_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0))

    f1_macro = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    report_dict = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )

    # Class-wise recall / sensitivity & support
    per_class_metrics = {}
    for lbl in labels:
        if lbl in report_dict:
            per_class_metrics[lbl] = {
                "precision": round(float(report_dict[lbl]["precision"]), 4),
                "recall": round(float(report_dict[lbl]["recall"]), 4),
                "f1_score": round(float(report_dict[lbl]["f1-score"]), 4),
                "support": int(report_dict[lbl]["support"]),
            }

    return {
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "macro_precision": round(prec_macro, 4),
        "weighted_precision": round(prec_weighted, 4),
        "macro_recall": round(rec_macro, 4),
        "weighted_recall": round(rec_weighted, 4),
        "macro_f1": round(f1_macro, 4),
        "weighted_f1": round(f1_weighted, 4),
        "labels": labels,
        "confusion_matrix": cm,
        "per_class": per_class_metrics,
    }


def compute_feature_importance(
    pipeline: Any,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_repeats: int = 5,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Compute permutation importance on the test/validation set for explainability.
    
    Includes explanatory disclaimer that importance reflects statistical association,
    not direct causation.
    """
    try:
        # Extract preprocessor and feature names
        preprocessor = pipeline.named_steps.get("preprocessor")
        classifier = pipeline.named_steps.get("classifier")

        # Compute permutation importance directly on the full pipeline
        perm_res = permutation_importance(
            pipeline,
            X_val,
            y_val,
            n_repeats=n_repeats,
            random_state=random_state,
            scoring="f1_macro",
            n_jobs=1,
        )

        raw_feature_names = list(X_val.columns)
        importances = []
        for name, mean_score, std_score in zip(
            raw_feature_names, perm_res.importances_mean, perm_res.importances_std
        ):
            importances.append({
                "feature": name,
                "importance_mean": round(float(mean_score), 5),
                "importance_std": round(float(std_score), 5),
            })

        # Sort descending
        importances.sort(key=lambda x: x["importance_mean"], reverse=True)

        return {
            "method": "Permutation Importance (held-out test set, metric=macro-f1)",
            "features": importances,
            "interpretation_note": (
                "Feature importance reflects statistical association with accident severity categories "
                "within the historical dataset. It does not establish direct causation or driver fault."
            ),
        }
    except Exception as e:
        logger.warning(f"Could not compute permutation importance: {e}")
        return {
            "method": "Unavailable",
            "features": [],
            "error": str(e),
            "interpretation_note": "Feature importance calculation failed or unsupported for this estimator.",
        }
