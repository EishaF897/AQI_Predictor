# -------- LIBRARIES ---------
import hopsworks
import joblib
import pandas as pd
import requests
import os
import numpy as np


# ------ FETCH FORECAST FROM OPENWEATHER -------
def fetch_forecast(lat, lon, api_key):

    url = (
        f"http://api.openweathermap.org/data/2.5/air_pollution/forecast"
        f"?lat={lat}&lon={lon}&appid={api_key}"
    )

    response = requests.get(url, timeout=10)
    response.raise_for_status()
    data = response.json()

    rows = []

    for item in data["list"]:
        rows.append({
            "timestamp": pd.to_datetime(item["dt"], unit="s", utc=True),
            "pm25": item["components"]["pm2_5"],
            "pm10": item["components"]["pm10"],
            "co": item["components"]["co"],
            "no2": item["components"]["no2"],
            "o3": item["components"]["o3"],
            "so2": item["components"]["so2"],
            "nh3": item["components"]["nh3"],
            "aqi": item["main"]["aqi"]  # used for lag features only
        })

    return pd.DataFrame(rows)


# ------ LOAD BEST MODEL (LATEST VERSION ONLY) -------
def load_best_model(project):

    mr = project.get_model_registry()

    candidate_model_names = [
        "aqi_logistic_regression",
        "aqi_random_forest",
        "aqi_xgboost"
    ]

    best_model = None
    best_f1 = -1

    for name in candidate_model_names:
        try:
            models = mr.get_models(name)

            if not models:
                continue

            # Select latest version only
            latest_model = max(models, key=lambda m: m.version)

            metrics = latest_model.training_metrics

            if metrics and "f1_score" in metrics:
                f1 = float(metrics["f1_score"])

                if f1 > best_f1:
                    best_f1 = f1
                    best_model = latest_model

        except Exception:
            continue

    if best_model is None:
        raise ValueError("No valid model found in registry.")

    model_dir = best_model.download()

    clf = joblib.load(f"{model_dir}/model.pkl")

    try:
        scaler = joblib.load(f"{model_dir}/scaler.pkl")
    except Exception:
        scaler = None

    return clf, scaler, best_model.name, best_model.version, best_f1


# ----------- MAIN PREDICTION FUNCTION -------------
def get_predictions(lat, lon, api_key):

    project = hopsworks.login(
        api_key_value=os.getenv("HOPSWORKS_API_KEY")
    )

    fs = project.get_feature_store()

    feature_view = fs.get_feature_view(
        name="aqi_feature_view",
        version=1
    )

    clf, scaler, model_name, version, f1 = load_best_model(project)

    # ------ Get historical data ------
    history_df = (
        feature_view.get_batch_data()
        .sort_values("timestamp")
        .tail(6)
    )

    # ------ Get forecast ------
    forecast_df = fetch_forecast(lat, lon, api_key)

    combined = pd.concat([history_df, forecast_df], ignore_index=True)
    combined["timestamp"] = pd.to_datetime(combined["timestamp"], utc=True)
    combined = combined.sort_values("timestamp").reset_index(drop=True)

    # ------- Feature Engineering (MUST MATCH TRAINING) -------
    combined["hour"] = combined["timestamp"].dt.hour
    combined["day"] = combined["timestamp"].dt.day
    combined["month"] = combined["timestamp"].dt.month
    combined["day_of_week"] = combined["timestamp"].dt.dayofweek
    combined["is_weekend"] = combined["day_of_week"].isin([5, 6]).astype("int32")

    combined["aqi_lag_1"] = combined["aqi"].shift(1)
    combined["aqi_lag_2"] = combined["aqi"].shift(2)
    combined["aqi_lag_3"] = combined["aqi"].shift(3)

    combined["aqi_roll_6h"] = combined["aqi"].rolling(6).mean()
    combined["aqi_change_rate"] = combined["aqi"] - combined["aqi_lag_1"]

    combined["pm25_roll_3h"] = combined["pm25"].rolling(3).mean()
    combined["pm10_roll_6h"] = combined["pm10"].rolling(6).mean()

    combined = combined.dropna()

    # Only future rows
    future_df = combined[
        combined["timestamp"] > history_df["timestamp"].max()
    ].copy()

    # Ensure exact feature order from feature view
    feature_columns = [
        feature.name
        for feature in feature_view.schema
        if feature.name not in ["aqi", "timestamp"]
    ]

    X = future_df[feature_columns]

    if scaler is not None:
        X = scaler.transform(X)

    # ---- Prediction ----
    predictions = clf.predict(X)

    # Fixing XGBoost label shift (trained on 0–4)
    if model_name == "aqi_xgboost":
        predictions = predictions + 1

    # ---- Confidence ----
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(X)
        confidence = np.max(probs, axis=1)
    else:
        confidence = np.ones(len(predictions))

    future_df["predicted_aqi_class"] = predictions
    future_df["confidence"] = np.round(confidence * 100, 2)

    return future_df, model_name, version, f1


# ---------- JSON OUTPUT -----------
def get_predictions_json():

    lat = 33.6844
    lon = 73.0479
    api_key = os.getenv("OPENWEATHER_API_KEY")

    df, model_name, version, f1 = get_predictions(lat, lon, api_key)

    return {
        "model": model_name,
        "version": version,
        "f1_score": f1,
        "predictions": df[
            ["timestamp", "predicted_aqi_class", "confidence"]
        ].to_dict(orient="records")
    }
