import streamlit as st
import requests
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide")

st.title("🌍 Real-Time 3 Day AQI Dashboard")

API_URL = "http://127.0.0.1:8000/predict"

# ---------------- LOADER ---------------- #
with st.spinner("Fetching AQI Forecast..."):
    response = requests.get(API_URL)
    data = response.json()

df = pd.DataFrame(data["predictions"])
df["timestamp"] = pd.to_datetime(df["timestamp"])

# ---------------- LINE CHART ---------------- #
fig = px.area(
    df,
    x="timestamp",
    y="predicted_aqi_class",
)

fig.update_traces(mode="lines")

fig.update_layout(
    title="3-Day AQI Forecast (Hourly)",
    yaxis=dict(
        tickmode="array",
        tickvals=[1, 2, 3, 4, 5],
        ticktext=["Good", "Moderate", "Unhealthy", "Very Unhealthy", "Hazardous"]
    ),
    showlegend=False
)

st.plotly_chart(fig, use_container_width=True)

# ---------------- DAILY SUMMARY ---------------- #

# Get daily average AQI class
df["date"] = df["timestamp"].dt.date
daily = (
    df.groupby("date")["predicted_aqi_class"]
    .mean()
    .round()
    .reset_index()
    .head(3)
)


aqi_labels = {
    1: "Good",
    2: "Moderate",
    3: "Unhealthy",
    4: "Very Unhealthy",
    5: "Hazardous"
}

colors = {
    1: "green",
    2: "#FFD700",
    3: "orange",
    4: "red",
    5: "purple"
}

st.markdown("## 📅 3-Day AQI Summary")

for _, row in daily.iterrows():
    aqi_value = int(row["predicted_aqi_class"])
    label = aqi_labels[aqi_value]
    color = colors[aqi_value]

    st.markdown(
        f"""
        <div style="
            display:flex;
            justify-content:space-between;
            align-items:center;
            padding:18px;
            border-left:8px solid {color};
            background-color:#f9f9f9;
            margin-bottom:8px;
            border-radius:8px;
        ">
            <div style="font-size:18px;">
                {row['date'].strftime('%A, %b %d')}
            </div>
            <div style="font-size:28px; font-weight:bold; color:{color};">
                AQI {aqi_value} - {label}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

