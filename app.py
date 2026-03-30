import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model (cached)
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

st.title("💼 Smart Salary Predictor")

# Inputs (customize based on your dataset)
job_domain = st.text_input("Job Domain")
experience = st.number_input("Experience (Years)", 0, 30)
qualification = st.text_input("Qualification")

# Add more fields if needed

if st.button("Predict Salary"):
    input_data = pd.DataFrame({
        "Job_Domain": [job_domain],
        "Experience_Years": [experience],
        "Qualification": [qualification]
    })

    prediction_log = model.predict(input_data)[0]
    prediction = np.expm1(prediction_log)

    st.success(f"Estimated Salary: ₹ {round(prediction, 2):,}")