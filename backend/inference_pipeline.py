
# -------- LIBRARIES ---------
import hopsworks
import joblib
import pandas as pd
import requests
import os


# ------ FETCH FORECAST FROM OPENWEATER -------

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
            "aqi": item["main"]["aqi"]
        })

    return pd.DataFrame(rows)


# ------ LOAD BEST MODEL FROM HOPSWORKS REGISTRY ------

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
            for model in models:
                metrics = model.training_metrics
                if metrics and "f1_score" in metrics:
                    f1 = metrics["f1_score"]
                    if f1 > best_f1:
                        best_f1 = f1
                        best_model = model
        except:
            continue
            
    if best_model is None:
        raise ValueError("No model with F1 score found.")

    model_dir = best_model.download()
    clf = joblib.load(f"{model_dir}/model.pkl")

    try:
        scaler = joblib.load(f"{model_dir}/scaler.pkl")
    except:
        scaler = None

    return clf, scaler, best_model.name, best_model.version, best_f1


# ----------- PREDICTION -------------

def get_predictions(lat, lon, api_key):
    
    # ----- Login -----
    project = hopsworks.login(
    api_key_value=os.getenv("HOPSWORKS_API_KEY")
    )

    fs = project.get_feature_store()
    feature_view = fs.get_feature_view(
        name="aqi_feature_view",
        version=1
    )
    clf, scaler, model_name, version, f1 = load_best_model(project)

    # ------ Get history ------
    history_df = feature_view.get_batch_data() \
        .sort_values("timestamp") \
        .tail(6)

    # ------- forecast --------
    forecast_df = fetch_forecast(lat, lon, api_key)

    combined = pd.concat([history_df, forecast_df], ignore_index=True)
    combined["timestamp"] = pd.to_datetime(combined["timestamp"], utc=True)
    combined = combined.sort_values("timestamp").reset_index(drop=True)

    # ------- Feature Engineering -------
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

    future_df = combined[
        combined["timestamp"] > history_df["timestamp"].max()
    ]
    feature_columns = [
        feature.name
        for feature in feature_view.schema
        if feature.name not in ["aqi", "timestamp"]
    ]
    X = future_df[feature_columns]

    if scaler:
        X = scaler.transform(X)

    predictions = clf.predict(X)
    future_df["predicted_aqi_class"] = predictions

    return future_df, model_name, version, f1


# ---------- GET PREDICTIONS (JSON FORMAT) -----------
def get_predictions_json():
    lat = 33.6844
    lon = 73.0479
    api_key = os.getenv("OPENWEATHER_API_KEY")
    df, model_name, version, f1 = get_predictions(lat, lon, api_key)
    output = {
        "model": model_name,
        "version": version,
        "f1_score": f1,
        "predictions": df[["timestamp", "predicted_aqi_class"]]
            .to_dict(orient="records")
    }
    return output
