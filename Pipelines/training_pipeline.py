# =========================================
# AQI CLASSIFICATION TRAINING PIPELINE
# =========================================

import os
import joblib
import numpy as np
import pandas as pd
import hopsworks

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical


# -----------------------------------------
# EVALUATION
# -----------------------------------------
def evaluate(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted"))
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

    TARGET = "aqi"

    FEATURES = [
        col for col in df.columns
        if col not in ["timestamp", TARGET]
    ]

    X = df[FEATURES]
    y = df[TARGET]

    return X, y, FEATURES, project


# -----------------------------------------
# TRAIN
# -----------------------------------------
def train():

    X, y, FEATURES, project = load_data()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        shuffle=False
    )

    os.makedirs("models", exist_ok=True)
    joblib.dump(FEATURES, "models/features.pkl")

    results = {}

    # ======================================
    # 1️⃣ RANDOM FOREST
    # ======================================
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )

    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_val)
    rf_metrics = evaluate(y_val, rf_preds)

    joblib.dump(rf, "models/rf_model.pkl")
    results["random_forest"] = (rf_metrics, "models/rf_model.pkl")

    # ======================================
    # 2️⃣ LOGISTIC REGRESSION
    # ======================================
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    log_model = LogisticRegression(
        max_iter=1000,
        multi_class="multinomial"
    )

    log_model.fit(X_train_scaled, y_train)
    log_preds = log_model.predict(X_val_scaled)
    log_metrics = evaluate(y_val, log_preds)

    joblib.dump(log_model, "models/log_model.pkl")
    joblib.dump(scaler, "models/log_scaler.pkl")

    results["logistic_regression"] = (log_metrics, "models/log_model.pkl")

    # ======================================
    # 3️⃣ NEURAL NETWORK
    # ======================================
    # Convert labels from 1–5 → 0–4
    y_train_nn = y_train - 1
    y_val_nn = y_val - 1

    nn = Sequential([
        Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
        Dense(32, activation="relu"),
        Dense(5, activation="softmax")  # fixed 5 classes
    ])

    nn.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",   # no one-hot needed
        metrics=["accuracy"]
    )

    nn.fit(
        X_train_scaled,
        y_train_nn,
        epochs=20,
        batch_size=32,
        verbose=0
    )

    nn_probs = nn.predict(X_val_scaled)
    nn_preds = np.argmax(nn_probs, axis=1) + 1   # convert back to 1–5

    nn_metrics = evaluate(y_val, nn_preds)

    nn.save("models/nn_model")
    joblib.dump(scaler, "models/nn_scaler.pkl")

    results["neural_network"] = (nn_metrics, "models/nn_model")


    # ======================================
    # REGISTER ALL MODELS
    # ======================================
    mr = project.get_model_registry()

    for name, (metrics, path) in results.items():

        if name == "neural_network":
            model_registry = mr.tensorflow.create_model(
                name=f"aqi_{name}",
                metrics=metrics
            )
            model_registry.save(path)

        else:
            model_registry = mr.python.create_model(
                name=f"aqi_{name}",
                metrics=metrics
            )
            model_registry.save("models")

        print(f"{name} registered with metrics:", metrics)

    print("\nTraining Completed Successfully ✅")


if __name__ == "__main__":
    train()
