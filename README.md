# **Pearls AQI Predictor (Internship Project)**

## **Overview**

This project predicts the Air Quality Index (AQI) category based on different air pollutants such as PM2.5, PM10, CO, NO₂, SO₂, O₃, and NH₃.
The goal of this project is to build a complete machine learning pipeline — from data collection to model training and real-time prediction.
Air pollution is a serious environmental issue, and predicting AQI can help people stay aware and take precautions when air quality becomes unhealthy.

## **Problem Statement**

Given pollutant concentration values and time-related features, the model predicts the AQI category.
This is a classification problem, where the output is an AQI class. The AQI is divided into classes from 1 (Highest) - 5 (Lowest) in training data gathered from OpenWeather API

## **Technologies Used**

- Python
- Pandas
- Scikit-learn
- Joblib
- Requests
- Feature Store using Hopsworks
- Air Pollution Forecast API from OpenWeather
- Git & GitHub

## **Dataset and Features**

The model is trained on historical air pollution data.

**Features used:**
- pm25
- pm10
- co
- no2
- o3
- so2
- nh3
- hour
- day_of_week
- month

**Target:**
- AQI category (encoded using LabelEncoder)

## **Project Workflow**

This project follows a proper ML pipeline approach:

**1. Feature Pipeline**
- Fetches air pollution data
- Performs preprocessing and feature engineering
- Stores features in Hopsworks Feature Store

**2. Training Pipeline**
- Reads data from Feature Store
- Trains machine learning models
- Evaluates performance
- Saves the best model using Joblib

**3. Inference Pipeline**
- Fetches forecast data from OpenWeather API
- Loads trained model
- Predicts AQI category for future timestamps

## **Model Used**

Multiple models were tested, and the best performing model was selected.

**Models Trained:**
- Logistic Regression
- Random Forest
- XGBoost

**Evaluation metrics used:**
- f1-Score and Accuracy
- Classification Report
- Confusion Matrix

The final model is saved and reused during inference.


### **Streamlit App Link**
https://aqipredictor-vgpmfey4cmdxuts5haqdmg.streamlit.app/