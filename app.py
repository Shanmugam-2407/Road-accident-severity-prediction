"""Main Streamlit application entrypoint for RoadGuard AI."""

import streamlit as st

# Define multi-page navigation across RoadGuard AI modules
pages = [
    st.Page("frontend/dashboard.py", title="Executive Dashboard", icon="📊", default=True),
    st.Page("frontend/pages/1_Predict_Severity.py", title="Predict Severity", icon="🔮"),
    st.Page("frontend/pages/2_Batch_Prediction.py", title="Batch Prediction", icon="📁"),
    st.Page("frontend/pages/3_Accident_Analytics.py", title="Accident Analytics", icon="📈"),
    st.Page("frontend/pages/4_Accident_Hotspots.py", title="Accident Hotspots", icon="🗺️"),
    st.Page("frontend/pages/5_Model_Performance.py", title="Model Performance", icon="🎯"),
]

pg = st.navigation(pages)
pg.run()
