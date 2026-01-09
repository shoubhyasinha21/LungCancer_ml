from flask import Flask, request, render_template
import joblib
import os
app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
@app.route('/')
def home():
    return render_template("index.html")
@app.route('/predict', methods=['POST'])
def predict():
    data = [
        float(request.form['age']),
        float(request.form['gender']),
        float(request.form['country']),
        float(request.form['cancer_stage']),
        float(request.form['family_history']),
        float(request.form['smoking_status']),
        float(request.form['bmi']),
        float(request.form['cholesterol_level']),
        float(request.form['hypertension']),
        float(request.form['asthma']),
        float(request.form['cirrhosis']),
        float(request.form['other_cancer']),
        float(request.form['treatment_type'])
    ]
    prediction = model.predict([data])[0]
    result = "Survived" if prediction == 1 else "Not Survived"
    return render_template("index.html", prediction=result)

