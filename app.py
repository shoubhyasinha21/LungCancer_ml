import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("lung_cancer_model.pkl")

st.title("🫁 Lung Cancer Prediction App")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=25)

with col2:
    smoking = st.number_input("Smoking", min_value=0, max_value=80, value=5)

if st.button("Predict"):

    input_data = pd.DataFrame({
        "age": [age],
        "smoking": [smoking]
    })

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠️ High Risk of Lung Cancer")
    else:
        st.success("✅ Low Risk of Lung Cancer")