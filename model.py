import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
# Backend folder path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Dataset path INSIDE backend folder
data_path = os.path.join(BASE_DIR, "dataset_med.csv")
# Load dataset
df = pd.read_csv(data_path)
# Handle missing values
df.fillna(df.median(numeric_only=True), inplace=True)
df.fillna(df.mode().iloc[0], inplace=True)
# Encode categorical values
le = LabelEncoder()
for col in df.select_dtypes(include='object'):
    df[col] = le.fit_transform(df[col])
# Features & target
X = df.drop(['id', 'survived', 'diagnosis_date', 'end_treatment_date'], axis=1)
y = df['survived']
# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)
# Save model INSIDE backend folder (IMPORTANT)
model_path = os.path.join(BASE_DIR, "model.pkl")
joblib.dump(model, model_path)
print("✅ Model trained and saved successfully!")
