"""Interactive exploratory accident analytics and multi-factor risk visualization."""

import streamlit as st
import pandas as pd
import plotly.express as px

from src.utils.config import DATASET_PATH, TARGET_COLUMN

st.set_page_config(page_title="Accident Analytics | ROADGUARD AI", page_icon="📈", layout="wide")

st.title("📈 Historical Collision & Risk Analytics")
st.markdown(
    "Explore multi-dimensional historical collision dynamics across weather, road surfaces, "
    "lighting parameters, and temporal patterns."
)


@st.cache_data
def load_data():
    if not DATASET_PATH.is_file():
        return pd.DataFrame()
    return pd.read_csv(DATASET_PATH, low_memory=False)


df_raw = load_data()
if df_raw.empty:
    st.warning("Historical dataset not found. Please ensure `data/sample/sample_accidents.csv` exists.")
    st.stop()

# Interactive Sidebar Filters
st.sidebar.header("🔍 Filter Analytics Data")

# Filter: Severity
severities = df_raw[TARGET_COLUMN].dropna().unique().tolist() if TARGET_COLUMN in df_raw.columns else []
selected_severities = st.sidebar.multiselect(
    "Accident Severity",
    options=severities,
    default=severities,
)

# Filter: Urban / Rural
areas = df_raw["urban_or_rural_area"].dropna().unique().tolist() if "urban_or_rural_area" in df_raw.columns else []
selected_areas = st.sidebar.multiselect(
    "Urban / Rural Area",
    options=areas,
    default=areas,
)

# Filter: Speed limit
speed_limits = sorted(df_raw["speed_limit"].dropna().unique().tolist()) if "speed_limit" in df_raw.columns else []
selected_speeds = st.sidebar.multiselect(
    "Speed Limit (mph)",
    options=speed_limits,
    default=speed_limits,
)

# Apply Filters
filtered_df = df_raw.copy()
if selected_severities and TARGET_COLUMN in filtered_df.columns:
    filtered_df = filtered_df[filtered_df[TARGET_COLUMN].isin(selected_severities)]
if selected_areas and "urban_or_rural_area" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["urban_or_rural_area"].isin(selected_areas)]
if selected_speeds and "speed_limit" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["speed_limit"].isin(selected_speeds)]

st.markdown(
    f"Displaying **{len(filtered_df):,}** of **{len(df_raw):,}** collision records based on active filters."
)

tab1, tab2, tab3 = st.tabs(["🌦️ Environmental Factors", "🛣️ Road & Infrastructure", "⏱️ Temporal & Correlation"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Accidents by Weather Condition")
        if "weather_conditions" in filtered_df.columns and TARGET_COLUMN in filtered_df.columns:
            w_df = filtered_df.groupby(["weather_conditions", TARGET_COLUMN]).size().reset_index(name="Count")
            fig_w = px.bar(
                w_df,
                x="weather_conditions",
                y="Count",
                color=TARGET_COLUMN,
                barmode="group",
                color_discrete_map={"Slight": "#38BDF8", "Serious": "#FBBF24", "Fatal": "#F87171"},
            )
            fig_w.update_layout(xaxis_tickangle=-30, height=380, margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig_w, use_container_width=True)

    with col2:
        st.subheader("Accidents by Light Condition")
        if "light_conditions" in filtered_df.columns and TARGET_COLUMN in filtered_df.columns:
            l_df = filtered_df.groupby(["light_conditions", TARGET_COLUMN]).size().reset_index(name="Count")
            fig_l = px.bar(
                l_df,
                x="light_conditions",
                y="Count",
                color=TARGET_COLUMN,
                barmode="stack",
                color_discrete_map={"Slight": "#38BDF8", "Serious": "#FBBF24", "Fatal": "#F87171"},
            )
            fig_l.update_layout(xaxis_tickangle=-25, height=380, margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig_l, use_container_width=True)

with tab2:
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Road Surface Condition Analysis")
        if "road_surface_conditions" in filtered_df.columns and TARGET_COLUMN in filtered_df.columns:
            s_df = filtered_df.groupby(["road_surface_conditions", TARGET_COLUMN]).size().reset_index(name="Count")
            fig_s = px.bar(
                s_df,
                x="road_surface_conditions",
                y="Count",
                color=TARGET_COLUMN,
                barmode="group",
                color_discrete_map={"Slight": "#38BDF8", "Serious": "#FBBF24", "Fatal": "#F87171"},
            )
            fig_s.update_layout(xaxis_tickangle=-20, height=380, margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig_s, use_container_width=True)

    with col4:
        st.subheader("Carriageway Road Type")
        if "road_type" in filtered_df.columns and TARGET_COLUMN in filtered_df.columns:
            rt_df = filtered_df.groupby(["road_type", TARGET_COLUMN]).size().reset_index(name="Count")
            fig_rt = px.bar(
                rt_df,
                x="road_type",
                y="Count",
                color=TARGET_COLUMN,
                barmode="stack",
                color_discrete_map={"Slight": "#38BDF8", "Serious": "#FBBF24", "Fatal": "#F87171"},
            )
            fig_rt.update_layout(xaxis_tickangle=-25, height=380, margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig_rt, use_container_width=True)

with tab3:
    col5, col6 = st.columns(2)
    with col5:
        st.subheader("Collision Frequency by Day of Week")
        if "day_of_week" in filtered_df.columns:
            day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            day_df = filtered_df["day_of_week"].value_counts().reindex(day_order).dropna().reset_index()
            day_df.columns = ["Day", "Accidents"]
            fig_day = px.bar(
                day_df,
                x="Day",
                y="Accidents",
                color_discrete_sequence=["#3B82F6"],
            )
            fig_day.update_layout(height=380, margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig_day, use_container_width=True)

    with col6:
        st.subheader("Numerical Attributes Correlation")
        num_cols = [c for c in ["speed_limit", "number_of_vehicles", "number_of_casualties"] if c in filtered_df.columns]
        if len(num_cols) >= 2:
            corr = filtered_df[num_cols].corr()
            fig_corr = px.imshow(
                corr,
                text_auto=True,
                aspect="auto",
                color_continuous_scale="Blues",
                labels=dict(color="Correlation"),
            )
            fig_corr.update_layout(height=380, margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("Insufficient numerical variables for correlation matrix.")
