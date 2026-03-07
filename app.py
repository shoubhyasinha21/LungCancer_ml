import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Lung Cancer Survival Predictor", layout="wide", page_icon="🫁")

@st.cache_resource
def load_data():
    if not os.path.exists('lung_cancer_model.pkl'): return None, None, None
    return joblib.load('lung_cancer_model.pkl'), joblib.load('encoders.pkl'), joblib.load('feature_names.pkl')

model, encoders, feature_names = load_data()

# Helper to find the right index for categorical defaults
def find_idx(col, target):
    classes = list(encoders[col].classes_)
    for i, c in enumerate(classes):
        if str(c).strip().lower() == str(target).lower(): return i
    return 0

st.title("🫁 Lung Cancer Survival Prediction")

if model is None:
    st.error("🛑 Model files missing. Run 'python train_model.py' first.")
    st.stop()

# Scenario Buttons
st.sidebar.header("Scenario Presets")
if st.sidebar.button("Load Healthy Profile (Low Risk)"):
    st.session_state.update({'age': 30, 'stage': 'Stage I', 'smoke': 'Never Smoked', 'bmi': 22.0, 'com': 'No'})
if st.sidebar.button("Load Critical Profile (High Risk)"):
    st.session_state.update({'age': 80, 'stage': 'Stage IV', 'smoke': 'Passive Smoker', 'bmi': 16.0, 'com': 'Yes'})

with st.form("input_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age", 1, 100, st.session_state.get('age', 50))
        gender = st.selectbox("Gender", encoders['gender'].classes_)
        stage = st.selectbox("Cancer Stage", encoders['cancer_stage'].classes_, index=find_idx('cancer_stage', st.session_state.get('stage', '')))
    
    with c2:
        smoke = st.selectbox("Smoking Status", encoders['smoking_status'].classes_, index=find_idx('smoking_status', st.session_state.get('smoke', '')))
        bmi = st.number_input("BMI", 10.0, 50.0, st.session_state.get('bmi', 24.0))
        hyp = st.selectbox("Hypertension", encoders['hypertension'].classes_, index=find_idx('hypertension', st.session_state.get('com', '')))
        asthma = st.selectbox("Asthma", encoders['asthma'].classes_, index=find_idx('asthma', st.session_state.get('com', '')))

    with c3:
        cirr = st.selectbox("Cirrhosis", encoders['cirrhosis'].classes_, index=find_idx('cirrhosis', st.session_state.get('com', '')))
        other = st.selectbox("Other Cancer", encoders['other_cancer'].classes_, index=find_idx('other_cancer', st.session_state.get('com', '')))
        dur = st.number_input("Duration (Days)", 1, 5000, 180)
        # Default fillers for less critical fields
        fam = encoders['family_history'].classes_[0]
        treat = encoders['treatment_type'].classes_[0]
        chol = 190.0

    submit = st.form_submit_button("Predict Survival Odds")

if submit:
    input_data = {
        'age': age, 'gender': gender, 'cancer_stage': stage, 'family_history': fam,
        'smoking_status': smoke, 'bmi': bmi, 'cholesterol_level': chol,
        'hypertension': hyp, 'asthma': asthma, 'cirrhosis': cirr,
        'other_cancer': other, 'treatment_type': treat, 'treatment_duration': dur
    }

    # Prepare features
    final_features = []
    for col in feature_names:
        val = input_data[col]
        if col in encoders:
            val = encoders[col].transform([str(val)])[0]
        final_features.append(val)

    # Get Probabilities
    probs = model.predict_proba([final_features])[0]
    # Class 1 is Survived (Low Risk)
    survival_prob = probs[1] if len(probs) > 1 else (probs[0] if model.predict([final_features])[0] == 1 else 0)

    st.divider()
    
    # Results Display
    kpi1, kpi2 = st.columns(2)
    with kpi1:
        st.metric("Survival Probability (Low Risk)", f"{survival_prob:.1%}")
        if survival_prob >= 0.5:
            st.success("✅ Outcome: Patient Likely to Survive")
        else:
            st.error("⚠️ Outcome: High Mortality Risk Detected")

    with kpi2:
        st.write("**Risk Visualizer**")
        st.progress(survival_prob)

    # Professional Insight
    with st.expander("See Feature Importance"):
        st.write("This chart shows which factors the model weighed most heavily for this decision.")
        importance_df = pd.DataFrame({'Feature': feature_names, 'Weight': model.feature_importances_})
        st.bar_chart(importance_df.set_index('Feature'))