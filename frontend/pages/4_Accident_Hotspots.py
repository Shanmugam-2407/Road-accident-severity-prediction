"""Geographic accident hotspot visualization and spatial density analysis."""

import streamlit as st
import pandas as pd
import plotly.express as px

from src.utils.config import DATASET_PATH, TARGET_COLUMN

st.set_page_config(page_title="Accident Hotspots | ROADGUARD AI", page_icon="🗺️", layout="wide")

st.title("🗺️ Geographic Accident Hotspots")
st.markdown(
    "Visualize spatial collision density and geographic clusters based on validated historical coordinates."
)


@st.cache_data
def load_geo_data():
    if not DATASET_PATH.is_file():
        return pd.DataFrame()
    return pd.read_csv(DATASET_PATH, low_memory=False)


df = load_geo_data()

has_geo = (
    "latitude" in df.columns
    and "longitude" in df.columns
    and df["latitude"].notna().any()
    and df["longitude"].notna().any()
)

if not has_geo:
    st.info(
        "📍 Geographic coordinates (`latitude`, `longitude`) are not available in the current dataset. "
        "Alternative regional analysis is shown below."
    )
    if "urban_or_rural_area" in df.columns and TARGET_COLUMN in df.columns:
        regional = df.groupby(["urban_or_rural_area", TARGET_COLUMN]).size().reset_index(name="Accident Count")
        fig_alt = px.bar(
            regional,
            x="urban_or_rural_area",
            y="Accident Count",
            color=TARGET_COLUMN,
            barmode="group",
            title="Accidents by Urban / Rural Categorization",
        )
        st.plotly_chart(fig_alt, use_container_width=True)
    st.stop()

# Clean and validate coordinates
geo_clean = df.dropna(subset=["latitude", "longitude"]).copy()
geo_clean = geo_clean[
    (geo_clean["latitude"] >= -90)
    & (geo_clean["latitude"] <= 90)
    & (geo_clean["longitude"] >= -180)
    & (geo_clean["longitude"] <= 180)
]

# Coordinate filter by severity
severities = geo_clean[TARGET_COLUMN].dropna().unique().tolist() if TARGET_COLUMN in geo_clean.columns else []
selected_severities = st.multiselect(
    "Filter Hotspot Map by Collision Severity",
    options=severities,
    default=severities,
)

if selected_severities and TARGET_COLUMN in geo_clean.columns:
    geo_filtered = geo_clean[geo_clean[TARGET_COLUMN].isin(selected_severities)]
else:
    geo_filtered = geo_clean

st.write(f"Displaying **{len(geo_filtered):,}** validated coordinates on map.")

# Coordinate bounding & sampling for smooth rendering
MAX_POINTS = 3000
if len(geo_filtered) > MAX_POINTS:
    st.caption(f"ℹ️ Rendering a representative sample of {MAX_POINTS:,} coordinates to optimize rendering responsiveness.")
    plot_df = geo_filtered.sample(n=MAX_POINTS, random_state=42)
else:
    plot_df = geo_filtered

# Use Plotly scatter_map (or scatter_mapbox) for clean interactive map
try:
    fig_map = px.scatter_map(
        plot_df,
        lat="latitude",
        lon="longitude",
        color=TARGET_COLUMN if TARGET_COLUMN in plot_df.columns else None,
        color_discrete_map={"Slight": "#38BDF8", "Serious": "#FBBF24", "Fatal": "#EF4444"},
        hover_name="accident_id" if "accident_id" in plot_df.columns else None,
        hover_data=["speed_limit", "weather_conditions", "road_surface_conditions"],
        zoom=5,
        height=620,
    )
except AttributeError:
    # Older Plotly compatibility
    fig_map = px.scatter_mapbox(
        plot_df,
        lat="latitude",
        lon="longitude",
        color=TARGET_COLUMN if TARGET_COLUMN in plot_df.columns else None,
        color_discrete_map={"Slight": "#38BDF8", "Serious": "#FBBF24", "Fatal": "#EF4444"},
        hover_name="accident_id" if "accident_id" in plot_df.columns else None,
        hover_data=["speed_limit", "weather_conditions", "road_surface_conditions"],
        mapbox_style="carto-positron",
        zoom=5,
        height=620,
    )

fig_map.update_layout(margin=dict(t=10, b=10, l=10, r=10))
st.plotly_chart(fig_map, use_container_width=True)

# Hotspot summary notes
st.markdown("### Spatial Cluster Insights")
c1, c2 = st.columns(2)
with c1:
    st.markdown(
        """
        - **Metropolitan Focus**: High-density clusters typically center around urban ring roads and major transport corridors.
        - **Severity Elevation**: Isolated points along arterial motorways and rural dual carriageways have significantly higher proportion of Fatal & Serious classifications due to higher speeds.
        """
    )
with c2:
    st.markdown(
        """
        - **Data Integrity**: All coordinates depicted are strictly verified against legitimate geographical latitude/longitude domains.
        - **Disclaimer**: Cluster density highlights historical event density, not necessarily inherent infrastructural flaw.
        """
    )

