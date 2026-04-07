import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import os
import pickle
import numpy as np

MODEL_PATH = "model.pkl"

def train_model():
    df = pd.read_csv("vancomycin_dosing_dataset_1200.csv")

    X = df[['anchor_age', 'weight_kg', 'creatinine', 'eGFR']]
    y = df['dose_mg']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = DecisionTreeRegressor(max_depth=4)
    model.fit(X_train, y_train)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    return model

def load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    else:
        return train_model()

model = load_model()

def predict_dose(age, weight, creatinine, egfr):
    features = np.array([[age, weight, creatinine, egfr]])
    predicted_dose = model.predict(features)[0]

    base_dose = 15 * weight
    adjustment_factor = predicted_dose / base_dose

    explanation = []

    if egfr < 30:
        explanation.append("Significant dose reduction due to poor kidney function")
    elif egfr < 60:
        explanation.append("Moderate dose reduction due to reduced kidney function")
    else:
        explanation.append("Normal kidney function")

    if weight > 90:
        explanation.append("Higher dose influenced by body weight")
    elif weight < 60:
        explanation.append("Lower dose influenced by body weight")

    return predicted_dose, base_dose, adjustment_factor, explanation