# ------- IMPORT ALL LIBRARIES ---------
import hopsworks
import pandas as pd
import joblib
import os
import numpy as np

from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

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
# ❌ REMOVED LabelEncoder (AQI already numeric 1–5)

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

# ---- Time-Based Train/Test Split ----
split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test  = X.iloc[split_index:]
y_train = y.iloc[:split_index]
y_test  = y.iloc[split_index:]

print("\nTrain Class Distribution:")
print(y_train.value_counts(normalize=True))

print("\nTest Class Distribution:")
print(y_test.value_counts(normalize=True))

# ------ SCALING (ONLY FOR LOGISTIC REGRESSION) -----
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ------ TRAIN MODELS (BALANCED) ------
models = {}

# ---- Logistic Regression ----
lr = LogisticRegression(
    max_iter=1500,
    class_weight="balanced",
    random_state=42
)

lr.fit(X_train_scaled, y_train)
models["logistic_regression"] = (lr, X_test_scaled)

# ---- Random Forest ----
rf = RandomForestClassifier(
    n_estimators=400,
    max_depth=12,
    min_samples_leaf=3,
    class_weight="balanced",
    random_state=42
)

rf.fit(X_train, y_train)
models["random_forest"] = (rf, X_test)

# -------- XGBoost --------

# Shifting labels to start from 0
y_train_xgb = y_train - y_train.min()
y_test_xgb  = y_test - y_train.min()

classes = np.unique(y_train_xgb)

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train_xgb
)

weight_dict = dict(zip(classes, class_weights))
sample_weights = y_train_xgb.map(weight_dict)

xgb = XGBClassifier(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.08,
    objective="multi:softprob",
    num_class=len(classes),
    eval_metric="mlogloss",
    random_state=42
)

xgb.fit(X_train, y_train_xgb, sample_weight=sample_weights)

models["xgboost"] = (xgb, X_test)


# ------- EVALUATE MODELS -------
results = {}

for name, (model, X_eval) in models.items():

    if name == "xgboost":
        y_pred = model.predict(X_eval)
        y_pred = y_pred + y_train.min()   # Shifting back to original labels
    else:
        y_pred = model.predict(X_eval)

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="macro")

    results[name] = {
        "accuracy": acc,
        "f1_score": f1
    }

    print(f"\n{name.upper()} Classification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

print("\nModel Performance Summary:")
for name, metrics in results.items():
    print(name, metrics)

# ------- REGISTER MODELS IN HOPSWORKS -------
mr = project.get_model_registry()

best_model_name = max(results, key=lambda x: results[x]["f1_score"])
print(f"\nBest Model Based on Macro F1: {best_model_name}")

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
        description="AQI multiclass classification model (time-based split, imbalance handled)"
    )

    model_registry.save(model_dir)
