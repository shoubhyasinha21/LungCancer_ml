import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("lung_cancer_model.pkl")

st.title("🫁 Lung Cancer Prediction App")

# Organize inputs
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=21)
    smoking = st.number_input("Smoking (Years/Intensity)", min_value=0, max_value=80, value=4)

with col2:
    coughing = st.selectbox("Chronic Coughing", ["No", "Yes"])
    chest_pain = st.selectbox("Chest Pain", ["No", "Yes"])

# Convert Yes/No → 1/0
coughing = 1 if coughing == "Yes" else 0
chest_pain = 1 if chest_pain == "Yes" else 0

if st.button("Predict"):

    # Create dataframe for prediction
    input_data = pd.DataFrame({
        "Age": [age],
        "Smoking": [smoking],
        "Coughing": [coughing],
        "Chest_Pain": [chest_pain]
    })

    # Model prediction
    prediction = model.predict(input_data)

    # Show result
    if prediction[0] == 1:
        st.error("⚠️ High Risk of Lung Cancer")
    else:
        st.success("✅ Low Risk of Lung Cancer")