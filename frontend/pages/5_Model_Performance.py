"""Model evaluation audit, candidate comparison, and explainability dashboard."""

import json
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.utils.config import EVALUATION_SUMMARY_PATH, MODEL_COMPARISON_PATH
from frontend.api_client import get_model_metrics_api

st.set_page_config(page_title="Model Performance | ROADGUARD AI", page_icon="🎯", layout="wide")

st.title("🎯 Machine Learning Model Performance & Audit")
st.markdown(
    "Audited validation metrics, multi-algorithm benchmarking, confusion matrices, "
    "and feature importance from actual test-set evaluations."
)


def load_eval_data():
    """Load evaluation data either through API or directly from generated reports."""
    try:
        api_data = get_model_metrics_api()
        return api_data
    except Exception:
        if EVALUATION_SUMMARY_PATH.is_file():
            with open(EVALUATION_SUMMARY_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        return None


data = load_eval_data()

if not data:
    st.error(
        "Model evaluation reports are not yet available. "
        "Please train the model pipeline by running: `python -m src.models.train`"
    )
    st.stop()

metrics = data.get("metrics", {})
meta = data.get("metadata", {})
model_name = data.get("selected_model", meta.get("model_name", "Random Forest"))
criterion = data.get("selection_criterion", "Highest Macro-F1 score on held-out test data.")

# Top Metrics Row
st.subheader(f"Selected Production Model: {model_name}")
st.caption(f"**Model Selection Criterion**: {criterion}")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Macro F1-Score", f"{metrics.get('macro_f1', 0):.4f}")
k2.metric("Balanced Accuracy", f"{metrics.get('balanced_accuracy', 0):.4f}")
k3.metric("Overall Accuracy", f"{metrics.get('accuracy', 0):.4f}")
k4.metric("Macro Precision", f"{metrics.get('macro_precision', 0):.4f}")
k5.metric("Macro Recall", f"{metrics.get('macro_recall', 0):.4f}")

st.markdown("---")

tab_comp, tab_cm, tab_imp, tab_audit = st.tabs([
    "📊 Algorithm Comparison",
    "🔲 Confusion Matrix",
    "🌟 Feature Importance",
    "📋 Model Audit & Limitations",
])

with tab_comp:
    st.subheader("Candidate Model Comparison Table")
    st.markdown(
        "Candidate algorithms evaluated on identical held-out test split (20%) using stratified sampling:"
    )

    comp_records = data.get("model_comparison", [])
    if not comp_records and MODEL_COMPARISON_PATH.is_file():
        comp_df = pd.read_csv(MODEL_COMPARISON_PATH)
    else:
        comp_df = pd.DataFrame(comp_records)

    if not comp_df.empty:
        st.dataframe(
            comp_df.style.highlight_max(
                subset=["macro_f1", "balanced_accuracy", "accuracy"],
                color="#DCFCE7"
            ),
            use_container_width=True,
        )

        fig_comp = px.bar(
            comp_df,
            x="model_name",
            y=["macro_f1", "balanced_accuracy", "accuracy"],
            barmode="group",
            labels={"value": "Score (0-1)", "model_name": "Algorithm", "variable": "Metric"},
            color_discrete_map={
                "macro_f1": "#3B82F6",
                "balanced_accuracy": "#10B981",
                "accuracy": "#64748B",
            },
        )
        fig_comp.update_layout(height=350, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_comp, use_container_width=True)
    else:
        st.info("Comparison data not available.")

with tab_cm:
    c1, c2 = st.columns([1, 1])

    with c1:
        st.subheader("Confusion Matrix (Held-out Test Split)")
        cm = metrics.get("confusion_matrix", [])
        labels = metrics.get("labels", ["Fatal", "Serious", "Slight"])

        if cm:
            cm_df = pd.DataFrame(cm, index=[f"Actual {l}" for l in labels], columns=[f"Pred {l}" for l in labels])
            fig_cm = px.imshow(
                cm_df,
                text_auto=True,
                color_continuous_scale="Blues",
                labels=dict(x="Predicted Class", y="Actual Ground Truth", color="Cases"),
            )
            fig_cm.update_layout(height=380, margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig_cm, use_container_width=True)
        else:
            st.info("Confusion matrix data missing.")

    with c2:
        st.subheader("Class-Wise Evaluation Metrics")
        per_class = metrics.get("per_class", {})
        if per_class:
            pc_rows = []
            for cls_name, vals in per_class.items():
                pc_rows.append({
                    "Severity Class": cls_name,
                    "Precision": vals.get("precision"),
                    "Recall / Sensitivity": vals.get("recall"),
                    "F1-Score": vals.get("f1_score"),
                    "Support (Test Count)": vals.get("support"),
                })
            pc_df = pd.DataFrame(pc_rows)
            st.dataframe(pc_df, use_container_width=True)

            st.caption(
                "Notice that standard Accuracy is biased toward the majority class (Slight), "
                "which is why Macro-F1 and Balanced Accuracy serve as the primary selection criteria."
            )

with tab_imp:
    st.subheader("Permutation Feature Importance (Held-out Test Split)")
    feat_imp = data.get("feature_importance", {})
    features_list = feat_imp.get("features", [])

    if features_list:
        fi_df = pd.DataFrame(features_list)
        fig_imp = px.bar(
            fi_df,
            x="importance_mean",
            y="feature",
            orientation="h",
            labels={"importance_mean": "Mean Decrease in Macro-F1", "feature": "Accident Circumstance Feature"},
            color="importance_mean",
            color_continuous_scale="Teal",
        )
        fig_imp.update_layout(yaxis={"categoryorder": "total ascending"}, height=420, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_imp, use_container_width=True)

        st.info(f"💡 **Explainability Note**: {feat_imp.get('interpretation_note')}")
    else:
        st.info("Feature importance data unavailable.")

with tab_audit:
    st.subheader("Model Governance & Operational Constraints")
    audit_c1, audit_c2 = st.columns(2)

    with audit_c1:
        st.markdown(
            f"""
            #### Model Metadata
            * **Trained Pipeline**: `{meta.get('model_name', 'Random Forest')}`
            * **Training Timestamp**: `{meta.get('trained_at', 'N/A')}`
            * **Training Samples**: `{meta.get('training_samples', 'N/A'):,}`
            * **Test Samples**: `{meta.get('test_samples', 'N/A'):,}`
            * **Algorithm Family**: `{meta.get('algorithm_family', 'RandomForestClassifier')}`
            * **Target Variable**: `{meta.get('target_column', 'accident_severity')}`
            """
        )

    with audit_c2:
        st.markdown(
            """
            #### Known Limitations & Responsible Use
            1. **Advisory Nature**: The model predicts statistical severity tiers based on historical correlations; it does not replace expert on-scene engineering or forensic collision investigations.
            2. **Target Leakage Prevention**: Post-incident details (such as casualty counts or hospital admissions) are strictly omitted from input features.
            3. **Imbalance Constraints**: While class-weighting and balanced tuning maximize recall, severe accident categories remain naturally infrequent in empirical road safety datasets.
            """
        )

