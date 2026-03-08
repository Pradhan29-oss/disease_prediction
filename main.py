from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List
import joblib
import numpy as np
import os

app = FastAPI(title="Disease Prediction API")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

MODELS = {}
METADATA = {}

@app.on_event("startup")
def load_all_models():
    for name in ["diabetes", "heart", "parkinsons"]:
        MODELS[name] = joblib.load(f"models/{name}_model.pkl")
    METADATA.update(joblib.load("models/metadata.pkl"))
    print("Sab models load ho gaye!")

def run_prediction(model_name, features, expected_count):
    if len(features) != expected_count:
        raise HTTPException(status_code=400, detail=f"{expected_count} features chahiye, {len(features)} mile")
    model = MODELS[model_name]
    meta = METADATA[model_name]
    X = np.array(features).reshape(1, -1)
    pred = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0]
    classes = meta["classes"]
    return {
        "result": classes[pred],
        "confidence": f"{proba[pred]*100:.1f}%",
        "probabilities": {cls: f"{p*100:.1f}%" for cls, p in zip(classes, proba)},
        "model_accuracy": f"{meta['accuracy']*100:.2f}%"
    }

class DiabetesInput(BaseModel):
    Pregnancies: float
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: float

class HeartInput(BaseModel):
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float

class ParkinsonsInput(BaseModel):
    MDVP_Fo: float
    MDVP_Fhi: float
    MDVP_Flo: float
    MDVP_Jitter: float
    MDVP_Jitter_Abs: float
    MDVP_RAP: float
    MDVP_PPQ: float
    Jitter_DDP: float
    MDVP_Shimmer: float
    MDVP_Shimmer_dB: float
    Shimmer_APQ3: float
    Shimmer_APQ5: float
    MDVP_APQ: float
    Shimmer_DDA: float
    NHR: float
    HNR: float
    RPDE: float
    DFA: float
    spread1: float
    spread2: float
    D2: float
    PPE: float

@app.get("/")
def home():
    return {"message": "Disease Prediction API chal raha hai!", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "OK", "models": list(MODELS.keys())}

@app.post("/predict/diabetes")
def predict_diabetes(req: DiabetesInput):
    features = [req.Pregnancies, req.Glucose, req.BloodPressure, req.SkinThickness,
                req.Insulin, req.BMI, req.DiabetesPedigreeFunction, req.Age]
    return run_prediction("diabetes", features, 8)

@app.post("/predict/heart")
def predict_heart(req: HeartInput):
    features = [req.age, req.sex, req.cp, req.trestbps, req.chol, req.fbs,
                req.restecg, req.thalach, req.exang, req.oldpeak, req.slope, req.ca, req.thal]
    return run_prediction("heart", features, 13)

@app.post("/predict/parkinsons")
def predict_parkinsons(req: ParkinsonsInput):
    features = [req.MDVP_Fo, req.MDVP_Fhi, req.MDVP_Flo, req.MDVP_Jitter,
                req.MDVP_Jitter_Abs, req.MDVP_RAP, req.MDVP_PPQ, req.Jitter_DDP,
                req.MDVP_Shimmer, req.MDVP_Shimmer_dB, req.Shimmer_APQ3, req.Shimmer_APQ5,
                req.MDVP_APQ, req.Shimmer_DDA, req.NHR, req.HNR,
                req.RPDE, req.DFA, req.spread1, req.spread2, req.D2, req.PPE]
    return run_prediction("parkinsons", features, 22)