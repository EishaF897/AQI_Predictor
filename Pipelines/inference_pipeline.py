# =========================================
# AQI 3-DAY FORECAST PIPELINE (FINAL)
# =========================================

import joblib
import pandas as pd
import numpy as np
from datetime import timedelta
import hopsworks
import tensorflow as tf


MODEL_NAME = "aqi_best_model"
HOURS_TO_FORECAST = 72


def load_artifacts():

    project = hopsworks.login()
    fs = project.get_feature_store()
    mr = project.get_model_registry()

    # Load feature group
    fg = fs.get_feature_group(
        name="aqi_features",
        version=1
    )

    df = fg.read()
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Load best model
    model = mr.get_model(MODEL_NAME, version=1)
    model_dir = model.download()

    # Load features list
    FEATURES = joblib.load(f"{model_dir}/features.pkl")

    # Try loading sklearn model
    try:
        model_obj = joblib.load(f"{model_dir}/best_model.pkl")
        model_type = "sklearn"
        scaler = None
    except:
        model_obj = tf.keras.models.load_model(f"{model_dir}/best_model")
        model_type = "tensorflow"
        scaler = joblib.load(f"{model_dir}/nn_scaler.pkl")

    return df, model_obj, model_type, scaler, FEATURES


def forecast():

    df, model, model_type, scaler, FEATURES = load_artifacts()

    latest_row = df.iloc[-1:].copy()
    predictions = []

    for i in range(HOURS_TO_FORECAST):

        X_input = latest_row[FEATURES]

        if model_type == "sklearn":
            pred = model.predict(X_input)[0]

        else:
            X_scaled = scaler.transform(X_input)
            pred = model.predict(X_scaled).flatten()[0]

        predictions.append(pred)

        # Update for next hour
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


if __name__ == "__main__":
    df_forecast = forecast()
    print(df_forecast.head())
