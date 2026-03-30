import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Smart Salary Predictor", layout="wide")

# ==========================================
# ULTRA PREMIUM STYLISH DESIGN (FIXED SPACING)
# ==========================================

st.markdown("""
<style>
/* Import Premium Fonts */
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;800&family=Space+Grotesk:wght@400;600;700&display=swap');

/* CRITICAL: Remove Streamlit's Default Header Padding */
.block-container {
    padding-top: 1rem !important;
    padding-bottom: 0rem !important;
    padding-left: 5rem !important;
    padding-right: 5rem !important;
}

/* Animated Premium Gradient Background */
.stApp {
    background: linear-gradient(-45deg, #0a0e27, #1a1f3a, #2d1b69, #0f0c29, #302b63, #24243e);
    background-size: 400% 400%;
    animation: gradientWave 20s ease infinite;
    font-family: 'Poppins', sans-serif;
}

@keyframes gradientWave {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* Floating Particles Effect */
.stApp::before {
    content: '';
    position: fixed;
    top: -200px;
    left: -200px;
    width: 600px;
    height: 600px;
    background: radial-gradient(circle, rgba(100,200,255,0.3), transparent);
    filter: blur(120px);
    animation: float 25s ease-in-out infinite;
    z-index: 0;
}

/* Subtle Grid Pattern Overlay */
.main-container {
    position: relative;
    z-index: 1;
    padding: 0px 20px 50px 20px; /* Reduced top padding to 0 */
    background-image: 
        linear-gradient(rgba(255,255,255,0.02) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.02) 1px, transparent 1px);
    background-size: 50px 50px;
}

/* Premium Title with Gradient Text */
.main-title {
    text-align: center;
    font-size: 60px; /* Slightly reduced for better fit */
    font-weight: 800;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 25%, #f093fb 50%, #4facfe 75%, #00f2fe 100%);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: gradientShift 6s ease infinite;
    font-family: 'Space Grotesk', sans-serif;
    letter-spacing: -2px;
    margin-top: -10px; /* Pulls title upwards */
    margin-bottom: 0px;
    text-shadow: 0 0 40px rgba(102,126,234,0.5);
}

@keyframes gradientShift {
    0%, 100% { background-position: 0% center; }
    50% { background-position: 100% center; }
}

/* Premium Subtitle */
.subtitle {
    text-align: center;
    color: rgba(255, 255, 255, 0.8);
    font-size: 18px;
    font-weight: 300;
    margin-bottom: 25px;
    margin-top: -5px; /* Reduced gap between title and subtitle */
    letter-spacing: 3px;
    text-transform: uppercase;
    font-family: 'Space Grotesk', sans-serif;
}

/* Premium Glass Morphism Card */
.glass-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.12), rgba(255,255,255,0.05));
    backdrop-filter: blur(25px) saturate(180%);
    -webkit-backdrop-filter: blur(25px) saturate(180%);
    border-radius: 35px;
    padding: 40px 45px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    border: 1px solid rgba(255, 255, 255, 0.18);
    position: relative;
    overflow: hidden;
}

/* Glowing Subheader */
.glass-card h3 {
    color: white !important;
    font-size: 24px !important;
    font-weight: 700 !important;
    margin-bottom: 30px !important;
    text-align: center;
    font-family: 'Space Grotesk', sans-serif !important;
    text-shadow: 0 0 30px rgba(100,200,255,0.6);
}

/* Premium Input Fields */
.stSelectbox label, .stSlider label {
    color: rgba(255,255,255,0.95) !important;
    font-size: 14px !important;
    font-weight: 600 !important;
}

/* Premium Predict Button */
.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%) !important;
    color: white !important;
    font-size: 18px !important;
    font-weight: 700 !important;
    border-radius: 50px !important;
    height: 55px !important;
    width: 100% !important;
    border: none !important;
    box-shadow: 0 10px 30px rgba(102,126,234,0.4);
    transition: all 0.3s ease;
}

/* Premium Salary Result Box */
.salary-box {
    background: linear-gradient(135deg, rgba(0,255,200,0.15), rgba(0,200,255,0.15));
    border: 2px solid rgba(0,255,200,0.5);
    padding: 30px;
    border-radius: 25px;
    font-size: 36px;
    font-weight: 800;
    color: #00ffae;
    text-align: center;
    margin-top: 30px;
    box-shadow: 0 0 50px rgba(0,255,200,0.3);
}

/* Hide Default Streamlit Elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

</style>
""", unsafe_allow_html=True)

# Main Application Wrapper
st.markdown('<div class="main-container">', unsafe_allow_html=True)

st.markdown('<div class="main-title">💼 Smart Salary Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI Powered HR Intelligence</div>', unsafe_allow_html=True)

# ==========================================
# Load Model (Wrapped in Try-Except for Safety)
# ==========================================
try:
    model = joblib.load("hr_salary_prediction_model.pkl")
except:
    st.error("Model file not found. Please ensure 'hr_salary_prediction_model.pkl' exists.")
    st.stop()

# ==========================================
# Premium Glass Card Form
# ==========================================
st.markdown('<div class="glass-card">', unsafe_allow_html=True)

st.markdown("### 📋 Enter Employee Details")

col1, col2 = st.columns(2)

with col1:
    job_domain = st.selectbox("🎯 Job Domain", [
       'HR Executive','Account Manager','Digital Marketing Executive',
       'Customer Support','Accountant','MIS Executive','Business Analyst',
       'Sales Executive','Computer Operator','Content Writer'
    ])
    experience = st.slider("⏱️ Experience (Years)", 0, 20, 2)
    qualification = st.selectbox("🎓 Qualification", [
        '10th Pass','12th Pass','Diploma','BBA','BCA',
        'B.Com','BA','MBA','M.Com','MA'
    ])

with col2:
    joining_period = st.selectbox("📅 Joining Period", [
         'Immediately','15 Days','30 Days','45 Days',
         '2 Months','3 Months'
    ])
    employment_type = st.selectbox("💼 Employment Type", [
        'Full Time','Part Time','Contract','Intern'
    ])
    company_size = st.selectbox("🏢 Company Size", [
        "Small","Medium","Large"
    ])

# Predict Button logic
st.write("---")
if st.button("🚀 Predict Salary"):
    input_data = pd.DataFrame({
        "Home_Location": ["Raipur"],
        "Qualification": [qualification],
        "Job_Domain": [job_domain],
        "Department": ["Operations"],
        "Employment_Type": [employment_type],
        "Company_Size": [company_size],
        "Joining_Period": [joining_period],
        "Experience_Years": [experience],
        "Performance_Rating": [4],
        "Promotion_Count": [1],
        "Training_Hours_Last_Year": [40],
        "Avg_Work_Hours_Per_Week": [42]
    })

    predicted_log_salary = model.predict(input_data)[0]
    # Assuming the model predicts log-transformed salary
    predicted_salary = np.expm1(predicted_log_salary)

    st.markdown(
        f'<div class="salary-box">💰 Estimated: ₹ {round(predicted_salary, 2):,}</div>',
        unsafe_allow_html=True
    )

st.markdown("</div>", unsafe_allow_html=True) # End Glass Card
st.markdown('</div>', unsafe_allow_html=True) # End Main Container