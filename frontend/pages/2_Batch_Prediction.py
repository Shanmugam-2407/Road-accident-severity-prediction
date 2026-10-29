"""Batch accident severity prediction and downloadable CSV reporting."""

import streamlit as st
import pandas as pd
import plotly.express as px
from frontend.api_client import predict_batch_api, get_health

st.set_page_config(page_title="Batch Prediction | ROADGUARD AI", page_icon="📁", layout="wide")

st.title("📁 Batch Severity Prediction")
st.markdown(
    "Upload a CSV file containing multiple collision records to perform batch inference. "
    "The application validates input attributes, computes severity predictions with confidence scores, "
    "and returns an enriched downloadable CSV."
)

health = get_health()
if health.get("status") != "healthy":
    st.warning(
        f"⚠️ **Backend Notification**: FastAPI prediction service is {health.get('status')}. "
        f"Please start `uvicorn backend.main:app` to execute batch predictions."
    )

uploaded_file = st.file_uploader(
    "Choose a collision dataset CSV file",
    type=["csv"],
    help="Upload a standard CSV file with accident environmental and roadway circumstances."
)

if uploaded_file is not None:
    try:
        preview_df = pd.read_csv(uploaded_file)
        st.subheader("Dataset Preview")
        st.write(f"Uploaded **{len(preview_df):,}** records and **{len(preview_df.columns)}** columns.")
        st.dataframe(preview_df.head(5), use_container_width=True)

        # Upload & Process Button
        if st.button("🚀 Run Batch Severity Prediction", type="primary"):
            # Reset file pointer
            uploaded_file.seek(0)
            file_bytes = uploaded_file.read()

            with st.spinner("Processing batch records through RoadGuard AI..."):
                success, result = predict_batch_api(file_bytes, uploaded_file.name)

            if success:
                results_df: pd.DataFrame = result
                st.success(f"Batch prediction completed successfully for {len(results_df):,} records!")

                # Distribution Summary
                c1, c2 = st.columns([1, 1])

                with c1:
                    st.subheader("Predicted Severity Breakdown")
                    pred_counts = results_df["predicted_severity"].value_counts().reset_index()
                    pred_counts.columns = ["Severity", "Count"]

                    fig_pie = px.pie(
                        pred_counts,
                        names="Severity",
                        values="Count",
                        color="Severity",
                        color_discrete_map={"Slight": "#38BDF8", "Serious": "#FBBF24", "Fatal": "#F87171"},
                        hole=0.4,
                    )
                    fig_pie.update_layout(height=280, margin=dict(t=10, b=10, l=10, r=10))
                    st.plotly_chart(fig_pie, use_container_width=True)

                with c2:
                    st.subheader("Prediction Confidence Statistics")
                    if "prediction_confidence" in results_df.columns:
                        avg_conf = results_df["prediction_confidence"].mean() * 100
                        min_conf = results_df["prediction_confidence"].min() * 100
                        max_conf = results_df["prediction_confidence"].max() * 100
                        st.metric("Mean Prediction Confidence", f"{avg_conf:.1f}%")
                        st.metric("Min Confidence", f"{min_conf:.1f}%")
                        st.metric("Max Confidence", f"{max_conf:.1f}%")

                st.subheader("Enriched Predictions Table")
                st.dataframe(results_df.head(25), use_container_width=True)

                # Download CSV
                csv_data = results_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Predicted Records as CSV",
                    data=csv_data,
                    file_name=f"predicted_{uploaded_file.name}",
                    mime="text/csv",
                    type="primary",
                )
            else:
                st.error(f"Batch prediction failed: {result}")

    except Exception as e:
        st.error(f"Failed to read CSV file: {str(e)}")
else:
    st.info("💡 **Tip**: You can test batch predictions using the sample dataset located at `data/sample/sample_accidents.csv`.")
