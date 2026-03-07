import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load and clean dataset
df = pd.read_csv('dataset_med.csv')
df.columns = df.columns.str.strip().str.lower()

# Feature Engineering: Treatment Duration
df['diagnosis_date'] = pd.to_datetime(df['diagnosis_date'])
df['end_treatment_date'] = pd.to_datetime(df['end_treatment_date'])
df['treatment_duration'] = (df['end_treatment_date'] - df['diagnosis_date']).dt.days

# Define features and target mapping
# 1 = Survived (Low Risk), 0 = Not Survived (High Risk)
y = df['survived'].apply(lambda x: 1 if str(x).lower().strip() == 'yes' else 0)

# Check class distribution to avoid bias
print(f"Outcome Distribution:\n{y.value_counts()}")

# Drop non-predictive and target columns
cols_to_drop = ['id', 'country', 'diagnosis_date', 'end_treatment_date', 'survived']
X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

# Professional Encoding
categorical_cols = ['gender', 'cancer_stage', 'family_history', 'smoking_status', 
                    'hypertension', 'asthma', 'cirrhosis', 'other_cancer', 'treatment_type']

encoders = {}
for col in categorical_cols:
    if col in X.columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training with Balanced Weights (Fixes the "Always High Risk" bias)
model = RandomForestClassifier(
    n_estimators=150, 
    random_state=42, 
    class_weight='balanced'
)
model.fit(X_train, y_train)

# Save Assets
joblib.dump(model, 'lung_cancer_model.pkl')
joblib.dump(encoders, 'encoders.pkl')
joblib.dump(X.columns.tolist(), 'feature_names.pkl')

print("✅ SUCCESS: Model trained with balanced weights. Run app.py now.")