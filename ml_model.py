"""
ml_model.py  –  Decision-Tree dosing models for all ClinAI drugs
-----------------------------------------------------------------
Vancomycin  : trained on the real vancomycin_dosing_dataset_1200.csv
Acetaminophen, Ibuprofen, Amoxicillin, Metformin :
    trained on the same patient population (age, weight, creatinine, eGFR)
    with clinically-derived target doses generated from published dosing rules.
    This lets the tree learn the same decision boundaries the rule engine used,
    while being extensible to real-world outcome data in the future.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

# ── paths (resolved relative to this file so the app works from any cwd) ──
_HERE         = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH  = os.path.join(_HERE, "vancomycin_dosing_dataset_1200.csv")
VANC_MODEL    = os.path.join(_HERE, "model.pkl")
DRUG_MODEL    = os.path.join(_HERE, "drug_models.pkl")

# ═══════════════════════════════════════════════════════════════════════════════
# Vancomycin  (original model)
# ═══════════════════════════════════════════════════════════════════════════════

def _train_vancomycin(df: pd.DataFrame) -> DecisionTreeRegressor:
    X = df[["anchor_age", "weight_kg", "creatinine", "eGFR"]]
    y = df["dose_mg"]
    X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    m = DecisionTreeRegressor(max_depth=4, random_state=42)
    m.fit(X_tr, y_tr)
    with open(VANC_MODEL, "wb") as f:
        pickle.dump(m, f)
    return m


def _load_vancomycin() -> DecisionTreeRegressor:
    if os.path.exists(VANC_MODEL):
        with open(VANC_MODEL, "rb") as f:
            return pickle.load(f)
    df = pd.read_csv(DATASET_PATH)
    return _train_vancomycin(df)


_vanc_model = _load_vancomycin()


def predict_dose(age: float, weight: float, creatinine: float, egfr: float):
    """Return (predicted_dose, base_dose, adjustment_factor, explanation_list) for Vancomycin."""
    features       = pd.DataFrame([[age, weight, creatinine, egfr]],
                                   columns=["anchor_age", "weight_kg", "creatinine", "eGFR"])
    predicted_dose = _vanc_model.predict(features)[0]
    base_dose      = 15 * weight
    adjustment     = predicted_dose / base_dose if base_dose else 1.0

    explanation = []
    if egfr < 30:
        explanation.append("Significant dose reduction due to severely impaired kidney function (eGFR < 30).")
    elif egfr < 60:
        explanation.append("Moderate dose reduction due to reduced kidney function (eGFR 30-60).")
    else:
        explanation.append("No renal reduction applied – kidney function within normal range.")

    if weight > 90:
        explanation.append("Higher base dose influenced by above-average body weight.")
    elif weight < 60:
        explanation.append("Lower base dose influenced by below-average body weight.")

    return predicted_dose, base_dose, adjustment, explanation


# ═══════════════════════════════════════════════════════════════════════════════
# Multi-drug models  (acetaminophen / ibuprofen / amoxicillin / metformin)
# ═══════════════════════════════════════════════════════════════════════════════

# ── Clinical rule functions used to generate training labels ──────────────────

def _label_acetaminophen(row) -> float:
    """15 mg/kg, cap 1000 mg, 25% reduction >= 65 yrs. No renal adjustment."""
    dose = min(15.0 * row["weight_kg"], 1000.0)
    if row["anchor_age"] >= 65:
        dose *= 0.75
    return round(dose)


def _label_ibuprofen(row) -> float:
    """10 mg/kg, cap 800 mg. Contraindicated eGFR < 30.
    50% reduction eGFR 30-59. 25% reduction >= 65 yrs."""
    if row["eGFR"] < 30:
        return 0.0
    dose = min(10.0 * row["weight_kg"], 800.0)
    if row["eGFR"] < 60:
        dose *= 0.5
    if row["anchor_age"] >= 65:
        dose *= 0.75
    return round(dose)


def _label_amoxicillin(row) -> float:
    """25 mg/kg, cap 500 mg. 50% reduction eGFR < 30. 25% reduction >= 65 yrs."""
    dose = min(25.0 * row["weight_kg"], 500.0)
    if row["eGFR"] < 30:
        dose *= 0.5
    if row["anchor_age"] >= 65:
        dose *= 0.75
    return round(dose)


def _label_metformin(row) -> float:
    """Flat 500 mg start. Contraindicated eGFR < 30. 25% reduction >= 65 yrs."""
    if row["eGFR"] < 30:
        return 0.0
    dose = 500.0
    if row["anchor_age"] >= 65:
        dose *= 0.75
    return round(dose)


_LABEL_FNS = {
    "acetaminophen": _label_acetaminophen,
    "ibuprofen":     _label_ibuprofen,
    "amoxicillin":   _label_amoxicillin,
    "metformin":     _label_metformin,
}

FEATURES = ["anchor_age", "weight_kg", "creatinine", "eGFR"]


def _train_drug_models(df: pd.DataFrame) -> dict:
    """Train one DecisionTreeRegressor per non-vancomycin drug and persist them."""
    models = {}
    for drug, label_fn in _LABEL_FNS.items():
        y = df.apply(label_fn, axis=1)
        X = df[FEATURES]
        X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        m = DecisionTreeRegressor(max_depth=5, random_state=42)
        m.fit(X_tr, y_tr)
        models[drug] = m
    with open(DRUG_MODEL, "wb") as f:
        pickle.dump(models, f)
    return models


def _load_drug_models() -> dict:
    if os.path.exists(DRUG_MODEL):
        with open(DRUG_MODEL, "rb") as f:
            return pickle.load(f)
    df = pd.read_csv(DATASET_PATH)
    return _train_drug_models(df)


_drug_models: dict = _load_drug_models()


# ── Public prediction API ─────────────────────────────────────────────────────

def predict_drug_dose(drug_key: str, age: float, weight: float,
                      creatinine: float, egfr: float) -> dict:
    """
    Predict dose for acetaminophen / ibuprofen / amoxicillin / metformin.

    Returns a dict with keys:
        predicted_dose  – float (mg)
        steps           – list[str] explanation strings
        warnings        – list[str] clinical warnings
    """
    if drug_key not in _drug_models:
        raise ValueError(f"No ML model available for drug '{drug_key}'.")

    model          = _drug_models[drug_key]
    features       = pd.DataFrame([[age, weight, creatinine, egfr]], columns=FEATURES)
    predicted_dose = float(model.predict(features)[0])

    # Hard safety overrides – contraindicated cases always return 0
    # regardless of what the tree predicts (safety fence)
    if drug_key == "ibuprofen" and egfr < 30:
        predicted_dose = 0.0
    if drug_key == "metformin" and egfr < 30:
        predicted_dose = 0.0

    steps    = []
    warnings = []

    steps.append(
        f"Decision Tree inputs: age={age} yrs, weight={weight} kg, "
        f"creatinine={creatinine} mg/dL, eGFR={egfr:.1f} mL/min/1.73m²"
    )

    # ── Per-drug explanation ──────────────────────────────────────────────────
    if drug_key == "acetaminophen":
        raw = 15.0 * weight
        base = min(raw, 1000.0)
        steps.append(f"Weight-based calculation: 15 mg/kg x {weight} kg = {raw:.1f} mg")
        if raw > 1000:
            steps.append("Dose capped at maximum single dose: 1000 mg")
        if age >= 65:
            steps.append(
                f"Age >= 65 adjustment: {base:.1f} mg x 0.75 = {base*0.75:.1f} mg"
            )
            warnings.append(
                "Patient >= 65 yrs – 25% dose reduction applied. Monitor closely for hepatotoxicity."
            )
        else:
            steps.append(f"No age-based adjustment required (age {age}).")
        steps.append("No renal adjustment required for Acetaminophen at this eGFR level.")

    elif drug_key == "ibuprofen":
        if egfr < 30:
            steps.append(
                f"eGFR {egfr:.1f} < 30 — Ibuprofen is CONTRAINDICATED. "
                "Decision Tree output: 0 mg."
            )
            warnings.append(
                "CONTRAINDICATED: eGFR < 30. Ibuprofen must not be used in severe renal impairment."
            )
        else:
            raw  = 10.0 * weight
            base = min(raw, 800.0)
            steps.append(f"Weight-based calculation: 10 mg/kg x {weight} kg = {raw:.1f} mg")
            if raw > 800:
                steps.append("Dose capped at maximum single dose: 800 mg")
            if egfr < 60:
                steps.append(
                    f"eGFR {egfr:.1f} in range 30-59: 50% renal reduction — "
                    f"{base:.1f} mg x 0.5 = {base*0.5:.1f} mg"
                )
                warnings.append(
                    "eGFR 30-59: 50% ibuprofen dose reduction applied. Monitor renal function closely."
                )
                base *= 0.5
            if age >= 65:
                steps.append(
                    f"Age >= 65 adjustment: {base:.1f} mg x 0.75 = {base*0.75:.1f} mg"
                )
                warnings.append("Patient >= 65 yrs – additional 25% reduction applied.")

    elif drug_key == "amoxicillin":
        raw  = 25.0 * weight
        base = min(raw, 500.0)
        steps.append(f"Weight-based calculation: 25 mg/kg x {weight} kg = {raw:.1f} mg")
        if raw > 500:
            steps.append("Dose capped at maximum single dose: 500 mg")
        if egfr < 30:
            steps.append(
                f"eGFR {egfr:.1f} < 30: 50% renal reduction — "
                f"{base:.1f} mg x 0.5 = {base*0.5:.1f} mg"
            )
            warnings.append(
                "eGFR < 30: Amoxicillin dose reduced by 50%. Consider extending dosing interval."
            )
            base *= 0.5
        else:
            steps.append(f"No renal reduction required (eGFR {egfr:.1f} >= 30).")
        if age >= 65:
            steps.append(
                f"Age >= 65 adjustment: {base:.1f} mg x 0.75 = {base*0.75:.1f} mg"
            )
            warnings.append("Patient >= 65 yrs – 25% reduction applied. Monitor for GI effects.")

    elif drug_key == "metformin":
        if egfr < 30:
            steps.append(
                f"eGFR {egfr:.1f} < 30 — Metformin is CONTRAINDICATED. "
                "Decision Tree output: 0 mg."
            )
            warnings.append(
                "CONTRAINDICATED: eGFR < 30. Metformin must not be used – risk of lactic acidosis."
            )
        else:
            steps.append(
                "Metformin uses a flat starting dose of 500 mg (not weight-based). "
                "Titrate upward based on glycaemic response."
            )
            if egfr < 60:
                warnings.append(
                    "eGFR 30-59: Use Metformin with caution. Maximum daily dose is 1000 mg. "
                    "Review renal function every 3-6 months."
                )
            if age >= 65:
                steps.append(
                    "Age >= 65 adjustment: 500 mg x 0.75 = 375 mg starting dose."
                )
                warnings.append(
                    "Patient >= 65 yrs – reduced starting dose recommended. "
                    "Titrate slowly; monitor renal function closely."
                )

    steps.append(
        f"Decision Tree final predicted dose: {predicted_dose:.0f} mg "
        f"(Decision Tree model trained on {len(_drug_models)} drug profiles "
        f"using the 1,200-patient vancomycin dosing dataset)."
    )

    return {
        "predicted_dose": round(predicted_dose),
        "steps":          steps,
        "warnings":       warnings,
    }
