# ml_api.py
import joblib
import numpy as np

model = joblib.load("breast_cancer_model.pkl")
scaler = joblib.load("scaler.pkl")

def predict_cancer(features):
    if len(features) != 30:
        raise ValueError("Exactly 30 features required")

    x = np.array(features).reshape(1, -1)
    x_scaled = scaler.transform(x)
    pred = model.predict(x_scaled)[0]

    label = "Benign" if pred == 1 else "Malignant"
    return label, int(pred)
