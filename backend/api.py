from fastapi import FastAPI

from inference_pipeline import get_predictions_json

app = FastAPI()

@app.get("/")
def health():
    return {"status": "API running"}
    
@app.get("/predict")
def predict():
    return get_predictions_json()
