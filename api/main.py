# api/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(title="Loan Approval API")

# Allow CORS (for frontend testing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load pipeline trained on dataset columns
model = joblib.load('./models/loan_approval_pipeline_api.pkl')

# Define API input model
class LoanApplication(BaseModel):
    income_annum: float
    loan_amount: float
    cibil_score: float
    education: str
    self_employed: str

@app.post("/predict")
def predict_loan(data: LoanApplication):
    # Convert input to DataFrame
    df = pd.DataFrame([data.dict()])

    # Predict probability
    prob = model.predict_proba(df)[0][1]

    # Business threshold
    approved = bool(prob >= 0.4)

    return {
    "approved": approved,  # now a proper Python bool
    "approval_probability": float(round(prob, 3))  # make sure it's Python float
    }
