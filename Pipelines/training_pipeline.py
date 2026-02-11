# =========================================
# AQI TRAINING PIPELINE (FINAL VERSION)
# =========================================

import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import hopsworks


# -----------------------------------------
# EVALUATION FUNCTION
# -----------------------------------------
def evaluate(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred))
    }


# -----------------------------------------
# LOAD DATA
# -----------------------------------------
def load_data():

    project = hopsworks.login()
    fs = project.get_feature_store()

    fg = fs.get_feature_group(
        name="aqi_features",
        version=1
    )

    df = fg.read()
    df = df.sort_values("timestamp").reset_index(drop=True)

    if len(df) < 50:
        raise ValueError("Not enough data for training.")

    TARGET = "aqi"

    FEATURES = [
        col for col in df.columns
        if col not in ["timestamp", TARGET, "aqi_lag_1"]
    ]

    X = df[FEATURES]
    y = df[TARGET]

    return X, y, FEATURES, project


# -----------------------------------------
# TRAIN FUNCTION
# -----------------------------------------
def train():

    X, y, FEATURES, project = load_data()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        shuffle=False
    )

    os.makedirs("models", exist_ok=True)

    # Save feature list (important for inference)
    joblib.dump(FEATURES, "models/features.pkl")

    # ======================
    # 1️⃣ RIDGE REGRESSION
    # ======================
    ridge_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0))
    ])

    ridge_pipeline.fit(X_train, y_train)
    ridge_preds = ridge_pipeline.predict(X_val)
    ridge_metrics = evaluate(y_val, ridge_preds)

    # ======================
    # 2️⃣ RANDOM FOREST
    # ======================
    rf_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )

    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_val)
    rf_metrics = evaluate(y_val, rf_preds)

    # ======================
    # 3️⃣ NEURAL NETWORK
    # ======================
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    nn_model = Sequential([
        Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
        Dense(32, activation="relu"),
        Dense(1)
    ])

    nn_model.compile(optimizer="adam", loss="mse")

    nn_model.fit(
        X_train_scaled,
        y_train,
        epochs=25,
        batch_size=32,
        validation_data=(X_val_scaled, y_val),
        verbose=0
    )

    nn_preds = nn_model.predict(X_val_scaled).flatten()
    nn_metrics = evaluate(y_val, nn_preds)

    # -----------------------------------------
    # PRINT RESULTS
    # -----------------------------------------
    results = pd.DataFrame.from_dict({
        "ridge": ridge_metrics,
        "random_forest": rf_metrics,
        "neural_net": nn_metrics
    }, orient="index")

    print("\nModel Performance:")
    print(results)

    # -----------------------------------------
    # MODEL SELECTION (Lowest RMSE Wins)
    # -----------------------------------------
    all_models = {
        "ridge": (ridge_pipeline, ridge_metrics),
        "random_forest": (rf_model, rf_metrics),
        "neural_net": (nn_model, nn_metrics)
    }

    best_model_name = min(all_models, key=lambda x: all_models[x][1]["rmse"])
    best_model, best_metrics = all_models[best_model_name]

    print(f"\nBest Model Selected: {best_model_name}")
    print(best_metrics)

    # -----------------------------------------
    # REGISTER BEST MODEL
    # -----------------------------------------
    mr = project.get_model_registry()

    if best_model_name == "neural_net":

        # Save NN + scaler
        best_model.save("models/best_model")
        joblib.dump(scaler, "models/nn_scaler.pkl")

        model_registry = mr.tensorflow.create_model(
            name="aqi_best_model",
            metrics=best_metrics
        )

        model_registry.save("models/best_model")

    else:

        joblib.dump(best_model, "models/best_model.pkl")

        model_registry = mr.python.create_model(
            name="aqi_best_model",
            metrics=best_metrics
        )

        model_registry.save("models")

    print("\nTraining & Registration Completed Successfully ✅")


# -----------------------------------------
# ENTRY POINT
# -----------------------------------------
if __name__ == "__main__":
    train()
