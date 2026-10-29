"""Main overview dashboard for RoadGuard AI."""

import os
from pathlib import Path
import streamlit as pd_st
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.utils.config import DATASET_PATH, TARGET_COLUMN
from frontend.api_client import get_health

# Page Configuration
st.set_page_config(
    page_title="ROADGUARD AI | Accident Severity & Risk Analytics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0F172A;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .status-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .status-online {
        background-color: #DCFCE7;
        color: #166534;
    }
    .status-offline {
        background-color: #FEE2E2;
        color: #991B1B;
    }
</style>
""", unsafe_allow_html=True)


def load_historical_data() -> pd.DataFrame:
    """Load historical accident dataset safely with caching."""
    if not DATASET_PATH.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(DATASET_PATH, low_memory=False)
    except Exception as e:
        st.error(f"Error loading historical dataset: {e}")
        return pd.DataFrame()


def render_dashboard():
    # Header
    col_header, col_status = st.columns([3, 1])
    with col_header:
        st.markdown('<div class="main-title">🛡️ ROADGUARD AI</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-title">Understand Risk. Improve Road Safety.</div>', unsafe_allow_html=True)

    with col_status:
        health = get_health()
        if health.get("status") == "healthy":
            st.markdown(
                '<div style="text-align: right; padding-top: 10px;">'
                '<span class="status-badge status-online">● API ONLINE (Model Active)</span>'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="text-align: right; padding-top: 10px;">'
                '<span class="status-badge status-offline">● API OFFLINE / DEGRADED</span>'
                '</div>',
                unsafe_allow_html=True,
            )

    df = load_historical_data()
    if df.empty:
        st.warning(
            f"No dataset found at '{DATASET_PATH}'. Please configure your CSV dataset or run the generator."
        )
        return

    # Top KPI Cards
    total_accidents = len(df)
    slight_count = (df[TARGET_COLUMN] == "Slight").sum() if TARGET_COLUMN in df.columns else 0
    serious_count = (df[TARGET_COLUMN] == "Serious").sum() if TARGET_COLUMN in df.columns else 0
    fatal_count = (df[TARGET_COLUMN] == "Fatal").sum() if TARGET_COLUMN in df.columns else 0
    fatal_pct = (fatal_count / total_accidents * 100) if total_accidents > 0 else 0

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Total Collisions</div>'
            f'<div class="metric-value">{total_accidents:,}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with kpi2:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Slight Severity</div>'
            f'<div class="metric-value" style="color: #0284C7;">{slight_count:,}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with kpi3:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Serious Severity</div>'
            f'<div class="metric-value" style="color: #D97706;">{serious_count:,}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with kpi4:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Fatal Incidents</div>'
            f'<div class="metric-value" style="color: #DC2626;">{fatal_count:,} <span style="font-size: 0.9rem; font-weight: normal; color: #64748B;">({fatal_pct:.1f}%)</span></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 1: Severity Distribution & Accident Trends by Time
    c1, c2 = st.columns([1, 1])

    with c1:
        st.subheader("Severity Distribution")
        if TARGET_COLUMN in df.columns:
            counts = df[TARGET_COLUMN].value_counts().reset_index()
            counts.columns = ["Severity", "Count"]
            colors = {"Slight": "#38BDF8", "Serious": "#FBBF24", "Fatal": "#F87171"}
            fig_pie = px.pie(
                counts,
                names="Severity",
                values="Count",
                color="Severity",
                color_discrete_map=colors,
                hole=0.45,
            )
            fig_pie.update_traces(textposition="inside", textinfo="percent+label")
            fig_pie.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=320)
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("Severity target column not available in dataset.")

    with c2:
        st.subheader("Accident Frequency by Time of Day")
        time_col = None
        for cand in ["accident_time", "time", "accident_hour"]:
            if cand in df.columns:
                time_col = cand
                break

        if time_col:
            if time_col in ["accident_time", "time"]:
                hours = df[time_col].dropna().astype(str).str.extract(r"^(\d{1,2})")[0]
                hours = pd.to_numeric(hours, errors="coerce").dropna().astype(int)
            else:
                hours = pd.to_numeric(df[time_col], errors="coerce").dropna().astype(int)

            hour_counts = hours.value_counts().sort_index().reset_index()
            hour_counts.columns = ["Hour", "Count"]

            fig_time = px.bar(
                hour_counts,
                x="Hour",
                y="Count",
                labels={"Hour": "Hour of Day (0-23)", "Count": "Number of Accidents"},
                color_discrete_sequence=["#475569"],
            )
            fig_time.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=320)
            st.plotly_chart(fig_time, use_container_width=True)
        else:
            st.info("Timestamp data not available in dataset.")

    # Row 2: Weather and Road Surface Breakdown
    c3, c4 = st.columns([1, 1])

    with c3:
        st.subheader("Accident Circumstances by Weather")
        if "weather_conditions" in df.columns and TARGET_COLUMN in df.columns:
            weather_sev = (
                df.groupby(["weather_conditions", TARGET_COLUMN])
                .size()
                .reset_index(name="Accident Count")
            )
            fig_weather = px.bar(
                weather_sev,
                x="weather_conditions",
                y="Accident Count",
                color=TARGET_COLUMN,
                barmode="stack",
                color_discrete_map={"Slight": "#38BDF8", "Serious": "#FBBF24", "Fatal": "#F87171"},
            )
            fig_weather.update_layout(
                xaxis_title="Weather Condition",
                margin=dict(t=10, b=10, l=10, r=10),
                height=350,
                xaxis_tickangle=-30,
            )
            st.plotly_chart(fig_weather, use_container_width=True)
        else:
            st.info("Weather condition field not present in dataset.")

    with c4:
        st.subheader("Accident Circumstances by Road Surface")
        if "road_surface_conditions" in df.columns and TARGET_COLUMN in df.columns:
            surface_sev = (
                df.groupby(["road_surface_conditions", TARGET_COLUMN])
                .size()
                .reset_index(name="Accident Count")
            )
            fig_surface = px.bar(
                surface_sev,
                x="road_surface_conditions",
                y="Accident Count",
                color=TARGET_COLUMN,
                barmode="stack",
                color_discrete_map={"Slight": "#38BDF8", "Serious": "#FBBF24", "Fatal": "#F87171"},
            )
            fig_surface.update_layout(
                xaxis_title="Road Surface Condition",
                margin=dict(t=10, b=10, l=10, r=10),
                height=350,
                xaxis_tickangle=-30,
            )
            st.plotly_chart(fig_surface, use_container_width=True)
        else:
            st.info("Road surface field not present in dataset.")

    # Footer Notes
    st.markdown("---")
    st.caption(
        "🛡️ **RoadGuard AI System Notice:** This machine learning platform estimates accident severity "
        "distributions from historical statistical patterns. Predictions are advisory and do not supersede "
        "human engineering, certified collision reconstruction, or operational emergency response protocol."
    )


if __name__ == "__main__":
    render_dashboard()

