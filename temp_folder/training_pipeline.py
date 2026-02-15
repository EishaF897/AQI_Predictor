
# ------- IMPORT ALL LIBRARIES ---------
import hopsworks
import pandas as pd
import joblib
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_class_weight

# ------ LOGIN & GET FEATURE GROUP ---------
project = hopsworks.login()
fs = project.get_feature_store()
fg = fs.get_feature_group(
    name="aqi_features",
    version=1
)
df = fg.read()

# ----- Sort by timestamp (CRITICAL for time series) ----
df = df.sort_values("timestamp")

# ----- DEFINE FEATURES AND TARGET -----
le = LabelEncoder()
df["aqi"] = le.fit_transform(df["aqi"])

target = "aqi"

features = [
    "pm25","pm10","co","no2","o3","so2","nh3",
    "hour","day","month","day_of_week","is_weekend",
    "aqi_lag_1","aqi_lag_2","aqi_lag_3",
    "aqi_roll_6h","aqi_change_rate",
    "pm25_roll_3h","pm10_roll_6h"
]

X = df[features]
y = df[target]

#  ---- Time-Based Train/Test Split ----
split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]
y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]


#  ------ SCALING (FOR LOGISTIC REGRESSION) -----
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#  ------ TRAIN MODELS (BALANCED) ------
models = {}

# ---- Logistic Regression (Balanced) ----
lr = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)
lr.fit(X_train_scaled, y_train)
models["logistic_regression"] = (lr, X_test_scaled)

# ---- Random Forest (Balanced) ----
rf = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced"
)
rf.fit(X_train, y_train)
models["random_forest"] = (rf, X_test)


# ---- XGBoost (Manual Class Weights Handling) ----

# Calculate class weights manually
classes = np.unique(y_train)
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

weight_dict = dict(zip(classes, class_weights))

# Assigning sample weights
sample_weights = y_train.map(weight_dict)

xgb = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    eval_metric="mlogloss",
    random_state=42
)

xgb.fit(X_train, y_train, sample_weight=sample_weights)
models["xgboost"] = (xgb, X_test)


#  ------- EVALUATE MODELS -------
results = {}

for name, (model, X_eval) in models.items():
    y_pred = model.predict(X_eval)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    results[name] = {
        "accuracy": acc,
        "f1_score": f1
    }

    # ---- Classification Report ----
    print(f"\n{name} Classification Report:")
    print(classification_report(y_test, y_pred))

print("Model Performance:")
for name, metrics in results.items():
    print(name, metrics)

# ------- REGISTER ALL MODELS IN HOPSWORKS -------
mr = project.get_model_registry()
best_model_name = max(results, key=lambda x: results[x]["f1_score"])
print(f"\nBest Model: {best_model_name}")

for name, (model, _) in models.items():
   base_model_dir = "models"
    os.makedirs(base_model_dir, exist_ok=True)
    model_dir = os.path.join(base_model_dir, f"{name}_model")
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, "model.pkl"))
    if name == "logistic_regression":
        joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))

    model_registry = mr.python.create_model(
        name=f"aqi_{name}",
        metrics=results[name],
        description="AQI multiclass classification model"
    )

    model_registry.save(model_dir)

