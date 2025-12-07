from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

# HF model repo id
HF_MODEL_ID = "cktai/tourism-wellness-rf-model"
MODEL_FILENAME = "model.joblib"

# Load model from Hugging Face model hub at startup
model_path = hf_hub_download(
    repo_id=HF_MODEL_ID,
    filename=MODEL_FILENAME,
    repo_type="model"
)
model = joblib.load(model_path)

# Feature schema: MUST match training features
class CustomerFeatures(BaseModel):
    Age: float
    NumberOfPersonVisiting: float
    PreferredPropertyStar: float
    NumberOfTrips: float
    NumberOfChildrenVisiting: float
    MonthlyIncome: float
    PitchSatisfactionScore: float
    NumberOfFollowups: float
    DurationOfPitch: float
    TypeofContact: str
    CityTier: float
    Occupation: str
    Gender: str
    MaritalStatus: str
    Passport: int
    OwnCar: int
    Designation: str
    ProductPitched: str

app = FastAPI(
    title="Tourism Wellness Package Prediction API",
    description="Predicts the probability that a customer will purchase the Wellness Tourism Package.",
    version="1.0.0",
)

@app.get("/")
def read_root():
    return {"message": "Tourism Wellness Package Prediction API is running."}

@app.post("/predict")
def predict(customer: CustomerFeatures):
    # Convert incoming payload to DataFrame
    data_dict = customer.dict()
    df = pd.DataFrame([data_dict])

    # Get prediction and probability
    proba = model.predict_proba(df)[:, 1][0]
    pred = int(proba >= 0.5)

    return {
        "input": data_dict,
        "predicted_class": pred,
        "predicted_probability": float(proba)
    }
