# =========================================
# AQI 3-DAY FORECAST PIPELINE
# =========================================

import joblib
import pandas as pd
import numpy as np
from datetime import timedelta
import hopsworks


MODEL_NAME = "aqi_best_model"
HOURS_TO_FORECAST = 72


def load_data_and_model():

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

    try:
        model = joblib.load(f"{model_dir}/best_model.pkl")
        model_type = "sklearn"
    except:
        import tensorflow as tf
        model = tf.keras.models.load_model(f"{model_dir}/best_model")
        model_type = "tensorflow"

    return df, model, model_type


def forecast():

    df, model, model_type = load_data_and_model()

    latest_row = df.iloc[-1:].copy()
    predictions = []

    for i in range(HOURS_TO_FORECAST):

        FEATURES = [
            col for col in df.columns
            if col not in ["timestamp", "aqi"]
        ]

        X_input = latest_row[FEATURES]

        if model_type == "sklearn":
            pred = model.predict(X_input)[0]
        else:
            pred = model.predict(X_input).flatten()[0]

        predictions.append(pred)

        # Update next row
        next_row = latest_row.copy()
        next_row["timestamp"] += timedelta(hours=1)
        next_row["aqi_lag_1"] = pred
        next_row["aqi"] = pred

        latest_row = next_row

    forecast_df = pd.DataFrame({
        "hour": range(1, HOURS_TO_FORECAST + 1),
        "predicted_aqi": predictions
    })

    print(forecast_df.head())
    return forecast_df


if __name__ == "__main__":
    forecast()
