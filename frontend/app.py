import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from datetime import datetime

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="AQI Forecast Dashboard",
    page_icon="🌍",
    layout="wide"
)

st.markdown("<h1 style='text-align: center;'>🌍 3-Day AQI Forecast Dashboard</h1>", unsafe_allow_html=True)

API_URL = "https://aqipredictor-production.up.railway.app/predict"

# ---------------- FETCH DATA ---------------- #
with st.spinner("Fetching AQI Forecast..."):
    try:
        response = requests.get(API_URL)
        response.raise_for_status()
        data = response.json()
    except:
        st.error("⚠ Unable to fetch AQI data. Please check backend.")
        st.stop()

df = pd.DataFrame(data["predictions"])
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp")

# ---------------- AQI LABELS ---------------- #
aqi_labels = {
    1: "Good",
    2: "Moderate",
    3: "Poor",
    4: "Very Poor",
    5: "Hazardous"
}

colors = {
    1: "#2ECC71",     # Green
    2: "#F1C40F",     # Yellow
    3: "#E67E22",     # Orange
    4: "#E74C3C",     # Red
    5: "#8E44AD"      # Purple
}

# ---------------- AQI HEALTH MESSAGES ---------------- #
aqi_messages = {
    1: {
        "message": "Air quality is excellent. Enjoy outdoor activities!",
        "advice": "No precautions needed.",
    },
    2: {
        "message": "Air quality is acceptable.",
        "advice": "Sensitive individuals should limit prolonged outdoor exposure.",
    },
    3: {
        "message": "Air quality is unhealthy for sensitive groups.",
        "advice": "Children, elderly and respiratory patients should reduce outdoor activity.",
    },
    4: {
        "message": "Air quality is very unhealthy.",
        "advice": "Avoid outdoor activities. Consider wearing a mask outside.",
    },
    5: {
        "message": "Air quality is hazardous!",
        "advice": "Stay indoors. Keep windows closed. Use air purifiers if available.",
    }
}


df["AQI Category"] = df["predicted_aqi_class"].map(aqi_labels)

# ---------------- TOP KPI CARDS ---------------- #
st.markdown("## 📊 Current Status")

latest = df.iloc[0]
current_value = int(latest["predicted_aqi_class"])
current_label = aqi_labels[current_value]
current_color = colors[current_value]

col1, col2 = st.columns(2)

# with col1:
#     st.metric(
#         "Current AQI Level",
#         f"{current_value} - {current_label}"
#     )

with col1:
    st.markdown(
        f"""
        <div style="
            padding:20px;
            border-radius:12px;
            background-color:{current_color}20;
            border-left:8px solid {current_color};
        ">
            <h4>Current AQI</h4>
            <h2 style="color:{current_color};">
                {current_value} - {current_label}
            </h2>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.metric("Date & Time", latest["timestamp"].strftime("%A, %d %B %Y | %I:%M %p"))

st.markdown("---")

# ---------------- AREA CHART ---------------- #
st.markdown("## 📈 Hourly AQI Forecast")

fig = px.area(
    df,
    x="timestamp",
    y="predicted_aqi_class",
    line_shape="spline"
)

fig.update_layout(
    yaxis=dict(
        tickmode="array",
        tickvals=[1, 2, 3, 4, 5],
        ticktext=["Good", "Moderate", "Poor", "Very Poor", "Hazardous"]
    ),
    xaxis_title="Date & Time",
    yaxis_title="AQI Category",
    showlegend=False,
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)

# ---------------- DAILY SUMMARY ---------------- #
st.markdown("## 📅 3-Day AQI Summary")

df["date"] = df["timestamp"].dt.date

daily = (
    df.groupby("date")["predicted_aqi_class"]
    .mean()
    .round()
    .reset_index()
    .head(3)
)

for _, row in daily.iterrows():
    aqi_value = int(row["predicted_aqi_class"])
    label = aqi_labels[aqi_value]
    color = colors[aqi_value]

    is_today = row["date"] == datetime.today().date()

    st.markdown(
        f"""
        <div style="
            display:flex;
            justify-content:space-between;
            align-items:center;
            padding:20px;
            border-left:10px solid {color};
            background-color:#1f2937;
            margin-bottom:10px;
            border-radius:12px;
        ">
            <div style="font-size:20px; font-weight:600;">
                {row['date'].strftime('%A, %d %B %Y')}
            </div>
            <div style="font-size:26px; font-weight:bold; color:{color};">
                {label}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")

# ---------------- HEALTH ADVISORY ---------------- #
st.markdown("## 🚨 Health Advisory")

current_info = aqi_messages[current_value]

st.markdown(
    f"""
    <div style="
        padding:25px;
        border-radius:15px;
        background-color:{current_color}20;
        border-left:12px solid {current_color};
        margin-bottom:20px;
    ">
        <h3 style="color:{current_color}; margin-bottom:10px;">
            {current_info['message']}
        </h3>
        <p style="font-size:18px;">
            👉 {current_info['advice']}
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.caption("AQI Categories: 1=Good | 2=Moderate | 3=Poor | 4=Very Poor | 5=Hazardous")
