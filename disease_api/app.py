import joblib
import pandas as pd
from fastapi import FastAPI
from typing import Literal
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

app = FastAPI(
    title = 'Disease Prediction API',
    description = 'Predicts heart disease risks from clinical measurements '
                  '(Cleveland dataset, Logistic Regression).',
    version = '1.0.0'
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

saved = joblib.load('model.pkl')
model = saved['model']
Columns=saved['columns']

class PatientData(BaseModel):
    age: float = Field(..., example=63, description="Age in years")
    sex: Literal[0, 1] = Field(..., example=1, description="1 = male, 0 = female")
    cp: Literal[1, 2, 3, 4] = Field(..., example=1, description="Chest pain type")
    trestbps: float = Field(..., example=145, description="Resting blood pressure (mm Hg)")
    chol: float = Field(..., example=233, description="Serum cholesterol (mg/dl)")
    fbs: Literal[0, 1] = Field(..., example=1, description="Fasting blood sugar > 120 mg/dl")
    restecg: Literal[0, 1, 2] = Field(..., example=2, description="Resting ECG results")
    thalach: float = Field(..., example=150, description="Max heart rate achieved")
    exang: Literal[0, 1] = Field(..., example=0, description="Exercise-induced angina")
    oldpeak: float = Field(..., example=2.3, description="ST depression induced by exercise")
    slope: Literal[1, 2, 3] = Field(..., example=3, description="Slope of peak exercise ST segment")
    ca: Literal[0, 1, 2, 3] = Field(..., example=0, description="Number of major vessels colored by fluoroscopy")
    thal: Literal[3, 6, 7] = Field(..., example=6, description="3=normal, 6=fixed defect, 7=reversible defect")

class PredictionResponse(BaseModel):
    prediction: Literal['Healthy', 'Disease']
    disease_probability: float

@app.get('/')
def read_root():
    return FileResponse('index.html')

@app.post('/predict', response_model=PredictionResponse)

def predict(patient: PatientData):
    row = pd.DataFrame([patient.model_dump()], columns=Columns)
    probability = model.predict_proba(row)[0, 1]
    label = 'Disease' if probability >= 0.5 else 'Healthy'

    return PredictionResponse(prediction=label, disease_probability=probability)


