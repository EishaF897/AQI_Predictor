from fastapi import FastAPI
from pipelines.inference_pipeline import get_predictions_json

app = FastAPI()

@app.get("/predict")
def predict():
    return get_predictions_json()
