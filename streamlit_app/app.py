import os
import sys

# If we are being run with `python app.py` (Docker SDK), re-launch with
# `streamlit run app.py` so that the Streamlit server actually starts.
if os.environ.get("RUNNING_IN_STREAMLIT", "") != "1":
    os.environ["RUNNING_IN_STREAMLIT"] = "1"
    # Hugging Face Spaces uses port 7860
    os.system("streamlit run app.py --server.port 7860 --server.address 0.0.0.0")
    sys.exit(0)

import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

st.set_page_config(
    page_title="Tourism Wellness Package Predictor",
    page_icon="🧳",
    layout="centered"
)

st.title("🧳 Wellness Tourism Package Prediction")
st.write("Predict whether a customer is likely to purchase the Wellness Tourism Package.")

@st.cache_resource
def load_model():
    repo_id = "cktai/tourism-wellness-rf-model"
    filename = "model.joblib"
    model_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="model"
    )
    model = joblib.load(model_path)
    return model

model = load_model()

st.sidebar.header("Customer Input Features")

def get_user_input():
    Age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=35)
    NumberOfPersonVisiting = st.sidebar.number_input("Number Of Person Visiting", min_value=1, max_value=10, value=2)
    PreferredPropertyStar = st.sidebar.number_input("Preferred Property Star", min_value=1, max_value=5, value=3)
    NumberOfTrips = st.sidebar.number_input("Number Of Trips per year", min_value=0, max_value=50, value=2)
    NumberOfChildrenVisiting = st.sidebar.number_input("Number Of Children Visiting", min_value=0, max_value=10, value=0)
    MonthlyIncome = st.sidebar.number_input("Monthly Income", min_value=1000, max_value=1000000, value=50000)
    PitchSatisfactionScore = st.sidebar.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
    NumberOfFollowups = st.sidebar.number_input("Number Of Followups", min_value=0, max_value=20, value=2)
    DurationOfPitch = st.sidebar.number_input("Duration Of Pitch (minutes)", min_value=0, max_value=120, value=15)

    TypeofContact = st.sidebar.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    CityTier = st.sidebar.selectbox("City Tier", [1, 2, 3])
    Occupation = st.sidebar.selectbox("Occupation", ["Salaried", "Self Employed", "Business", "Free Lancer", "Other"])
    Gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    MaritalStatus = st.sidebar.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    Passport = st.sidebar.selectbox("Passport", [0, 1])
    OwnCar = st.sidebar.selectbox("Own Car", [0, 1])
    Designation = st.sidebar.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP", "Other"])
    ProductPitched = st.sidebar.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])

    data = {
        "Age": Age,
        "NumberOfPersonVisiting": NumberOfPersonVisiting,
        "PreferredPropertyStar": PreferredPropertyStar,
        "NumberOfTrips": NumberOfTrips,
        "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
        "MonthlyIncome": MonthlyIncome,
        "PitchSatisfactionScore": PitchSatisfactionScore,
        "NumberOfFollowups": NumberOfFollowups,
        "DurationOfPitch": DurationOfPitch,
        "TypeofContact": TypeofContact,
        "CityTier": CityTier,
        "Occupation": Occupation,
        "Gender": Gender,
        "MaritalStatus": MaritalStatus,
        "Passport": Passport,
        "OwnCar": OwnCar,
        "Designation": Designation,
        "ProductPitched": ProductPitched,
    }

    return pd.DataFrame([data])

input_df = get_user_input()

st.subheader("Input Features")
st.write(input_df)

if st.button("Predict Wellness Package Purchase"):
    proba = model.predict_proba(input_df)[:, 1][0]
    pred = int(proba >= 0.5)

    st.subheader("Prediction")
    st.write(f"Predicted Class (ProdTaken): **{pred}** (1 = Will Purchase, 0 = Will Not Purchase)")
    st.write(f"Predicted Probability of Purchase: **{proba:.4f}**")
