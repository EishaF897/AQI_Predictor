# =========================================
# AQI INFERENCE PIPELINE (AUTO BEST MODEL)
# =========================================

import joblib
import numpy as np
import pandas as pd
import hopsworks
import tensorflow as tf


def load_best_model(project):

    mr = project.get_model_registry()

    model_names = [
        "aqi_random_forest",
        "aqi_logistic_regression",
        "aqi_neural_network"
    ]

    best_f1 = -1
    best_model = None
    best_name = None

    for name in model_names:
        model = mr.get_model(name, version=None)   # latest version
        metrics = model.training_metrics
        f1 = metrics["f1_weighted"]

        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_name = name


    print("Best model selected:", best_name)

    model_dir = best_model.download()

    FEATURES = joblib.load(f"{model_dir}/features.pkl")

    if "neural" in best_name:
        model_obj = tf.keras.models.load_model(f"{model_dir}/nn_model")
        scaler = joblib.load(f"{model_dir}/nn_scaler.pkl")
        model_type = "nn"
    elif "logistic" in best_name:
        model_obj = joblib.load(f"{model_dir}/log_model.pkl")
        scaler = joblib.load(f"{model_dir}/log_scaler.pkl")
        model_type = "log"
    else:
        model_obj = joblib.load(f"{model_dir}/rf_model.pkl")
        scaler = None
        model_type = "rf"

    return model_obj, scaler, model_type, FEATURES


def run_inference():

    project = hopsworks.login()
    fs = project.get_feature_store()

    fg = fs.get_feature_group(
        name="aqi_features",
        version=1
    )

    df = fg.read()
    df = df.sort_values("timestamp")

    latest_row = df.iloc[-1:]
    model, scaler, model_type, FEATURES = load_best_model(project)

    X = latest_row[FEATURES]

    if model_type == "rf":
        prediction = model.predict(X)[0]

    elif model_type == "log":
        X_scaled = scaler.transform(X)
        prediction = model.predict(X_scaled)[0]

    else:
        X_scaled = scaler.transform(X)
        probs = model.predict(X_scaled)
        prediction = np.argmax(probs, axis=1)[0] + 1

    print("\nPredicted AQI Class:", prediction)


if __name__ == "__main__":
    run_inference()
