# =========================================
# AQI FORECAST DASHBOARD
# =========================================

import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import tensorflow as tf
import hopsworks
from datetime import timedelta


MODEL_NAME = "aqi_best_model"
HOURS_TO_FORECAST = 72


# -----------------------------------------
# LOAD DATA + MODEL
# -----------------------------------------
@st.cache_resource
def load_artifacts():

    project = hopsworks.login()
    fs = project.get_feature_store()
    mr = project.get_model_registry()

    fg = fs.get_feature_group(
        name="aqi_features",
        version=1
    )

    df = fg.read()
    df = df.sort_values("timestamp").reset_index(drop=True)

    model = mr.get_model(MODEL_NAME, version=1)
    model_dir = model.download()

    FEATURES = joblib.load(f"{model_dir}/features.pkl")

    try:
        model_obj = joblib.load(f"{model_dir}/best_model.pkl")
        model_type = "sklearn"
        scaler = None
    except:
        model_obj = tf.keras.models.load_model(f"{model_dir}/best_model")
        model_type = "tensorflow"
        scaler = joblib.load(f"{model_dir}/nn_scaler.pkl")

    return df, model_obj, model_type, scaler, FEATURES


# -----------------------------------------
# FORECAST FUNCTION
# -----------------------------------------
def forecast(df, model, model_type, scaler, FEATURES):

    latest_row = df.iloc[-1:].copy()
    predictions = []

    for _ in range(HOURS_TO_FORECAST):

        X_input = latest_row[FEATURES]

        if model_type == "sklearn":
            pred = model.predict(X_input)[0]
        else:
            X_scaled = scaler.transform(X_input)
            pred = model.predict(X_scaled).flatten()[0]

        predictions.append(pred)

        # Update row for next hour
        next_row = latest_row.copy()
        next_row["timestamp"] += timedelta(hours=1)
        next_row["aqi_lag_1"] = pred
        next_row["aqi"] = pred

        latest_row = next_row

    forecast_df = pd.DataFrame({
        "datetime": pd.date_range(
            start=df["timestamp"].iloc[-1] + timedelta(hours=1),
            periods=HOURS_TO_FORECAST,
            freq="H"
        ),
        "predicted_aqi": predictions
    })

    return forecast_df


# -----------------------------------------
# STREAMLIT UI
# -----------------------------------------

st.set_page_config(page_title="AQI 3-Day Forecast", layout="wide")

st.title("🌫️ AQI 3-Day Forecast Dashboard")
st.markdown("Real-time AQI forecasting using automated ML pipeline")

with st.spinner("Loading model and computing forecast..."):

    df, model, model_type, scaler, FEATURES = load_artifacts()
    forecast_df = forecast(df, model, model_type, scaler, FEATURES)
    # Convert datetime
    forecast_df["datetime"] = pd.to_datetime(forecast_df["datetime"])

    # Extract date + day
    forecast_df["date"] = forecast_df["datetime"].dt.date
    forecast_df["day_name"] = forecast_df["datetime"].dt.day_name()

    # Aggregate per day
    daily_summary = forecast_df.groupby(["date", "day_name"]).agg(
    min_aqi=("predicted_aqi", "min"),
    max_aqi=("predicted_aqi", "max"),
    avg_aqi=("predicted_aqi", "mean")
    ).reset_index()


st.success("Forecast Generated Successfully!")

# -----------------------------------------
# Display Metrics
# -----------------------------------------

st.subheader("📅 3-Day AQI Forecast Summary")

for _, row in daily_summary.iterrows():
    st.markdown(f"""
    ### {row['day_name']} ({row['date']})
    - 🔽 Minimum AQI: **{row['min_aqi']:.2f}**
    - 🔼 Maximum AQI: **{row['max_aqi']:.2f}**
    - 📊 Average AQI: **{row['avg_aqi']:.2f}**
    """)


# -----------------------------------------
# Interactive Chart
# -----------------------------------------

st.subheader("📈 AQI Forecast Trend")

fig = px.bar(
    daily_summary,
    x="day_name",
    y="avg_aqi",
    title="Average AQI for Next 3 Days"
)


st.plotly_chart(fig, use_container_width=True)


# -----------------------------------------
# Raw Data
# -----------------------------------------

st.subheader("📄 Forecast Table")
st.dataframe(forecast_df)
