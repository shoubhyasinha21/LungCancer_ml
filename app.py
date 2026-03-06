import streamlit as st
import pandas as pd
import joblib

# ... (load model code) ...

st.title("🫁 Lung Cancer Prediction App")

# Organize into columns
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=21)
    smoking = st.number_input("Smoking (Years/Intensity)", min_value=0, max_value=80, value=4)

with col2:
    # Add a new feature example
    coughing = st.selectbox("Chronic Coughing", ["No", "Yes"])
    chest_pain = st.selectbox("Chest Pain", ["No", "Yes"])

if st.button("Predict"):
    # Convert 'Yes'/'No' to 1/0 for the model
    # Ensure your model was trained with these extra columns!
    st.error("High Risk of Lung Cancer") # Example output