"""Interactive single accident severity prediction page."""

import streamlit as st
import plotly.express as px
import pandas as pd

from src.utils.config import CATEGORICAL_CHOICES
from frontend.api_client import predict_severity_api, APIClientError, get_health

st.set_page_config(page_title="Predict Severity | ROADGUARD AI", page_icon="🔮", layout="wide")

st.markdown("""
<style>
    .prediction-box-Slight {
        background-color: #E0F2FE;
        border: 2px solid #38BDF8;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        color: #0369A1;
    }
    .prediction-box-Serious {
        background-color: #FEF3C7;
        border: 2px solid #F59E0B;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        color: #B45309;
    }
    .prediction-box-Fatal {
        background-color: #FEE2E2;
        border: 2px solid #EF4444;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        color: #B91C1C;
    }
    .severity-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 5px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🔮 Accident Severity Prediction")
st.markdown(
    "Configure collision environmental variables and road attributes below. "
    "The trained machine learning pipeline will infer the estimated severity risk tier."
)

# API Status check banner
health = get_health()
if health.get("status") != "healthy":
    st.warning(
        f"⚠️ **Backend Notification**: FastAPI prediction service is currently {health.get('status')}. "
        f"Ensure `uvicorn backend.main:app --port 8000` is active to receive real-time inferences."
    )

with st.form("severity_prediction_form"):
    st.subheader("1. Collision Circumstances")
    col1, col2, col3 = st.columns(3)

    with col1:
        num_vehicles = st.number_input(
            "Number of Vehicles Involved",
            min_value=1,
            max_value=20,
            value=2,
            step=1,
            help="Total count of vehicles involved in the collision."
        )
        speed_limit = st.selectbox(
            "Speed Limit (mph)",
            options=[20, 30, 40, 50, 60, 70],
            index=1,
            help="Posted legal speed limit for the roadway section."
        )

    with col2:
        accident_hour = st.slider(
            "Time of Collision (Hour of Day: 0-23)",
            min_value=0,
            max_value=23,
            value=17,
            help="Approximate hour of collision (24-hour clock)."
        )
        day_of_week = st.selectbox(
            "Day of Week",
            options=CATEGORICAL_CHOICES["day_of_week"],
            index=4,  # Friday
        )

    with col3:
        urban_rural = st.selectbox(
            "Urban or Rural Classification",
            options=CATEGORICAL_CHOICES["urban_or_rural_area"],
            index=0,
        )
        junction_detail = st.selectbox(
            "Junction Detail",
            options=CATEGORICAL_CHOICES["junction_detail"],
            index=1,
        )

    st.subheader("2. Environmental & Road Attributes")
    col4, col5, col6 = st.columns(3)

    with col4:
        road_type = st.selectbox(
            "Road Carriageway Type",
            options=CATEGORICAL_CHOICES["road_type"],
            index=0,
        )

    with col5:
        road_surface = st.selectbox(
            "Road Surface Condition",
            options=CATEGORICAL_CHOICES["road_surface_conditions"],
            index=0,
        )

    with col6:
        weather = st.selectbox(
            "Weather Condition",
            options=CATEGORICAL_CHOICES["weather_conditions"],
            index=0,
        )

    light = st.selectbox(
        "Lighting Condition",
        options=CATEGORICAL_CHOICES["light_conditions"],
        index=0,
    )

    submitted = st.form_submit_button("⚡ Predict Collision Severity", type="primary", use_container_width=True)

if submitted:
    payload = {
        "number_of_vehicles": int(num_vehicles),
        "speed_limit": int(speed_limit),
        "accident_hour": int(accident_hour),
        "day_of_week": day_of_week,
        "road_type": road_type,
        "road_surface_conditions": road_surface,
        "weather_conditions": weather,
        "light_conditions": light,
        "urban_or_rural_area": urban_rural,
        "junction_detail": junction_detail,
    }

    with st.spinner("Analyzing collision factors via RoadGuard ML Pipeline..."):
        try:
            result = predict_severity_api(payload)
            severity = result["predicted_severity"]
            confidence = result.get("confidence", 0.0)
            class_probs = result.get("class_probabilities", {})

            st.success("Analysis complete.")

            res_col1, res_col2 = st.columns([1, 1])

            with res_col1:
                st.markdown(
                    f'<div class="prediction-box-{severity}">'
                    f'<div style="font-size: 0.95rem; font-weight: 600; text-transform: uppercase;">Estimated Risk Level</div>'
                    f'<div class="severity-title">{severity.upper()}</div>'
                    f'<div style="font-size: 1.1rem; font-weight: 500;">Confidence: {confidence * 100:.1f}%</div>'
                    f'<div style="margin-top: 10px; font-size: 0.85rem; opacity: 0.85;">Model: {result.get("model_identifier", "Trained ML Model")}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            with res_col2:
                st.subheader("Severity Class Probability Distribution")
                if class_probs:
                    prob_df = pd.DataFrame([
                        {"Severity": k, "Probability": v * 100}
                        for k, v in class_probs.items()
                    ])
                    fig_bar = px.bar(
                        prob_df,
                        x="Severity",
                        y="Probability",
                        color="Severity",
                        color_discrete_map={"Slight": "#38BDF8", "Serious": "#FBBF24", "Fatal": "#F87171"},
                        text_auto=".1f",
                    )
                    fig_bar.update_layout(
                        yaxis_title="Probability (%)",
                        yaxis_range=[0, 100],
                        height=260,
                        margin=dict(t=10, b=10, l=10, r=10),
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.info(f"💡 **Responsible AI Disclaimer**: {result.get('disclaimer')}")

        except APIClientError as e:
            st.error(f"❌ Prediction request failed: {e}")
        except Exception as e:
            st.error(f"❌ Unexpected error during prediction: {e}")

