# ================================
# AQI FEATURE PIPELINE SCRIPT
# ================================

# -------- IMPORT LIBRARIES --------
import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import hopsworks


# -------- CONSTANTS --------
LAT = 33.6844   # Islamabad Latitude
LON = 73.0479   # Islamabad Longitude
API_KEY = "YOUR_API_KEY_HERE"   # 🔒 Replace with your API key


# -------- FETCH HISTORICAL AIR POLLUTION DATA --------
def fetch_air_pollution_history(lat, lon, api_key, start_date, end_date):
    rows = []
    current = start_date

    while current < end_date:
        chunk_end = min(current + timedelta(days=5), end_date)

        url = (
            f"http://api.openweathermap.org/data/2.5/air_pollution/history"
            f"?lat={lat}&lon={lon}"
            f"&start={int(current.timestamp())}"
            f"&end={int(chunk_end.timestamp())}"
            f"&appid={api_key}"
        )

        res = requests.get(url, timeout=10)
        res.raise_for_status()
        data = res.json()

        for item in data.get("list", []):
            rows.append({
                "timestamp": pd.to_datetime(item["dt"], unit="s"),
                "pm25": item["components"]["pm2_5"],
                "pm10": item["components"]["pm10"],
                "co": item["components"]["co"],
                "no2": item["components"]["no2"],
                "o3": item["components"]["o3"],
                "so2": item["components"]["so2"],
                "nh3": item["components"]["nh3"],
                "aqi": item["main"]["aqi"]
            })

        current = chunk_end
        time.sleep(1)

    return pd.DataFrame(rows)


# -------- FEATURE ENGINEERING --------
def compute_features(df):
    df = df.sort_values("timestamp").reset_index(drop=True)

    # ---- Time-based features ----
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # ---- Lag & trend features ----
    df["aqi_lag_1"] = df["aqi"].shift(1)
    df["aqi_change_rate"] = df["aqi"] - df["aqi_lag_1"]

    # ---- Rolling averages ----
    df["pm25_roll_3h"] = df["pm25"].rolling(window=3).mean()
    df["pm10_roll_6h"] = df["pm10"].rolling(window=6).mean()

    df = df.dropna()

    return df


# -------- MAIN PIPELINE --------
def run_pipeline():

    # Login to Hopsworks
    project = hopsworks.login()
    fs = project.get_feature_store()

    # Get feature group
    aqi_fg = fs.get_feature_group(
        name="aqi_features",
        version=1
    )

    # Fetch last 1 hour data
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(hours=1)

    df_raw = fetch_air_pollution_history(
        LAT,
        LON,
        API_KEY,
        start_date,
        end_date
    )

    if df_raw.empty:
        print("No raw data fetched.")
        return

    df_features = compute_features(df_raw)

    if df_features.empty:
        print("No new features to insert.")
    else:
        aqi_fg.insert(df_features)
        print(f"Inserted {len(df_features)} new rows into Feature Group.")


# -------- ENTRY POINT --------
if __name__ == "__main__":
    run_pipeline()
