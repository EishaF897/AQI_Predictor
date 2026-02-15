from fastapi import FastAPI
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipelines.inference_pipeline import get_predictions_json

app = FastAPI()

@app.get("/")
def health():
    return {"status": "API running"}
    
@app.get("/predict")
def predict():
    return get_predictions_json()
