import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv("dataset_med.csv")

target_col = df.columns[-1]

X = df.drop(columns=[target_col])
y = df[target_col]

X = pd.get_dummies(X)

print("Dataset shape:", df.shape)
print("Total features:", X.shape[1])

# Faster model
model = RandomForestClassifier(n_estimators=20, random_state=42)

print("Training model...")
model.fit(X, y)

joblib.dump(model, "lung_cancer_model.pkl")
joblib.dump(X.columns.tolist(), "model_features.pkl")

print("Model training completed!")