"""FastAPI service. Run:  uvicorn api:app --reload   (docs at http://127.0.0.1:8000/docs)"""
import joblib, pandas as pd
from fastapi import FastAPI

app = FastAPI(title="Adult Income API")
model = joblib.load("best_adult_income_model.joblib")   # saved by q3_automl.py


@app.post("/predict")
def predict(features: dict):
    """Body = the 14 raw columns, using the dataset's names, e.g. {"age": 39, "education.num": 13, ...}"""
    df = pd.DataFrame([features])
    df["net.capital"] = df["capital.gain"] - df["capital.loss"]      # same derived feature as training
    pred = int(model.predict(df)[0])
    return {"prediction": ">50K" if pred == 1 else "<=50K"}
