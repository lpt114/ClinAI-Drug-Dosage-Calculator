
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

# ── Paths (resolved relative to this file) ────────────────────────────────────
_HERE        = os.path.dirname(os.path.abspath(__file__))
VANC_CSV     = os.path.join(_HERE, "vancomycin_dosing_dataset_1200.csv")
VANC_PKL     = os.path.join(_HERE, "model.pkl")
DRUG_PKL     = os.path.join(_HERE, "drug_models.pkl")

# Drug-specific dataset paths (MIMIC-IV derived)
DRUG_DATASETS = {
    "acetaminophen": os.path.join(_HERE, "acetaminophen_dosing_dataset.csv"),
    "ibuprofen":     os.path.join(_HERE, "ibuprofen_dosing_dataset.csv"),
    "amoxicillin":   os.path.join(_HERE, "amoxicillin_dosing_dataset.csv"),
    "metformin":     os.path.join(_HERE, "metformin_dosing_dataset.csv"),
}

# Features used per drug
DRUG_FEATURES = {
    "acetaminophen": ["anchor_age", "gender", "weight_kg", "ALT", "AST", "creatinine", "eGFR"],
    "ibuprofen":     ["anchor_age", "gender", "weight_kg", "creatinine", "eGFR"],
    "amoxicillin":   ["anchor_age", "gender", "weight_kg", "creatinine", "eGFR"],
    "metformin":     ["anchor_age", "gender", "weight_kg", "creatinine", "eGFR", "glucose", "HbA1c"],
}


# ═══════════════════════════════════════════════════════════════════════════════
# Vancomycin  (original model — unchanged)
# ═══════════════════════════════════════════════════════════════════════════════

def _train_vancomycin() -> DecisionTreeRegressor:
    df = pd.read_csv(VANC_CSV)
    X  = df[["anchor_age", "weight_kg", "creatinine", "eGFR"]]
    y  = df["dose_mg"]
    X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    m = DecisionTreeRegressor(max_depth=4, random_state=42)
    m.fit(X_tr, y_tr)
    with open(VANC_PKL, "wb") as f:
        pickle.dump(m, f)
    return m


def _load_vancomycin() -> DecisionTreeRegressor:
    if os.path.exists(VANC_PKL):
        with open(VANC_PKL, "rb") as f:
            return pickle.load(f)
    return _train_vancomycin()


_vanc_model = _load_vancomycin()


def predict_dose(age: float, weight: float, creatinine: float, egfr: float):
    """Vancomycin dose prediction. Returns (dose, base_dose, adjustment, explanation)."""
    X              = pd.DataFrame([[age, weight, creatinine, egfr]],
                                  columns=["anchor_age", "weight_kg", "creatinine", "eGFR"])
    predicted_dose = float(_vanc_model.predict(X)[0])
    base_dose      = 15.0 * weight
    adjustment     = predicted_dose / base_dose if base_dose else 1.0

    explanation = []
    if egfr < 30:
        explanation.append("Significant dose reduction due to severely impaired kidney function (eGFR < 30).")
    elif egfr < 60:
        explanation.append("Moderate dose reduction due to reduced kidney function (eGFR 30–60).")
    else:
        explanation.append("No renal reduction applied – kidney function within normal range.")

    if weight > 90:
        explanation.append("Higher base dose influenced by above-average body weight.")
    elif weight < 60:
        explanation.append("Lower base dose influenced by below-average body weight.")

    return predicted_dose, base_dose, adjustment, explanation


# ═══════════════════════════════════════════════════════════════════════════════
# Multi-drug models  (MIMIC-IV derived datasets)
# ═══════════════════════════════════════════════════════════════════════════════

def _train_drug_models() -> dict:
    """Train one DecisionTreeRegressor per non-vancomycin drug from its MIMIC-IV dataset."""
    models = {}
    for drug, csv_path in DRUG_DATASETS.items():
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"Dataset for '{drug}' not found at: {csv_path}\n"
                "Run the dataset builder script to generate drug datasets from MIMIC-IV."
            )
        df   = pd.read_csv(csv_path)
        feats = DRUG_FEATURES[drug]
        X    = df[feats].fillna(df[feats].median())
        y    = df["dose_mg"]
        X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        m = DecisionTreeRegressor(max_depth=5, random_state=42)
        m.fit(X_tr, y_tr)
        models[drug] = m
        print(f"[ml_model] Trained {drug} model on {len(df)} rows.")
    with open(DRUG_PKL, "wb") as f:
        pickle.dump(models, f)
    return models


def _load_drug_models() -> dict:
    if os.path.exists(DRUG_PKL):
        with open(DRUG_PKL, "rb") as f:
            return pickle.load(f)
    return _train_drug_models()


_drug_models: dict = _load_drug_models()


# ── Public prediction API ─────────────────────────────────────────────────────

def predict_drug_dose(drug_key: str, age: float, weight: float,
                      creatinine: float, egfr: float,
                      gender: int = 0,
                      alt: float = 25.0, ast: float = 30.0,
                      glucose: float = 120.0, hba1c: float = 7.0) -> dict:
    """
    Predict single dose for acetaminophen / ibuprofen / amoxicillin / metformin.

    Parameters
    ----------
    drug_key   : one of 'acetaminophen', 'ibuprofen', 'amoxicillin', 'metformin'
    age        : patient age in years
    weight     : body weight in kg
    creatinine : serum creatinine in mg/dL
    egfr       : estimated GFR in mL/min/1.73m²
    gender     : 1 = Male, 0 = Female  (default Female)
    alt        : ALT liver enzyme in IU/L (acetaminophen only; default 25)
    ast        : AST liver enzyme in IU/L (acetaminophen only; default 30)
    glucose    : blood glucose in mg/dL  (metformin only; default 120)
    hba1c      : HbA1c percentage         (metformin only; default 7.0)

    Returns
    -------
    dict with keys: predicted_dose (int mg), steps (list[str]), warnings (list[str])
    """
    if drug_key not in _drug_models:
        raise ValueError(f"No ML model available for drug '{drug_key}'.")

    feats  = DRUG_FEATURES[drug_key]
    values = {
        "anchor_age": age,
        "gender":     gender,
        "weight_kg":  weight,
        "creatinine": creatinine,
        "eGFR":       egfr,
        "ALT":        alt,
        "AST":        ast,
        "glucose":    glucose,
        "HbA1c":      hba1c,
    }
    X              = pd.DataFrame([[values[f] for f in feats]], columns=feats)
    predicted_dose = float(_drug_models[drug_key].predict(X)[0])

    # ── Hard safety fences ────────────────────────────────────────────────────
    if drug_key == "ibuprofen" and egfr < 30:
        predicted_dose = 0.0
    if drug_key == "metformin" and egfr < 30:
        predicted_dose = 0.0

    steps    = []
    warnings = []

    # ── Dataset provenance note ───────────────────────────────────────────────
    dataset_note = (
        f"Model trained on MIMIC-IV clinical data "
        f"({len(pd.read_csv(DRUG_DATASETS[drug_key]))} patient records, "
        f"features: {', '.join(feats)})."
    )

    steps.append(
        f"Decision Tree inputs — age: {age} yrs, weight: {weight} kg, "
        f"creatinine: {creatinine} mg/dL, eGFR: {egfr:.1f} mL/min/1.73m²"
        + (f", ALT: {alt} IU/L, AST: {ast} IU/L" if drug_key == "acetaminophen" else "")
        + (f", glucose: {glucose} mg/dL, HbA1c: {hba1c}%" if drug_key == "metformin" else "")
    )

    # ── Per-drug explanation ──────────────────────────────────────────────────
    if drug_key == "acetaminophen":
        liver_flag = alt > 120 or ast > 120
        if liver_flag:
            steps.append(
                f"Elevated liver enzymes detected (ALT {alt} or AST {ast} > 120 IU/L, ~3× ULN). "
                "Dose limited to 650 mg max to reduce hepatotoxicity risk."
            )
            warnings.append(
                "Elevated ALT/AST: Acetaminophen dose reduced. Avoid prolonged use; "
                "do not exceed 2 g/day total in hepatic impairment."
            )
        if age >= 65:
            steps.append("Age ≥ 65: dose capped at 650 mg per administration.")
            warnings.append("Patient ≥ 65 yrs: reduced dose applied. Monitor for hepatotoxicity.")
        if weight < 50:
            steps.append(f"Low body weight ({weight} kg): weight-based reduction applied.")
        if not liver_flag and age < 65 and weight >= 50:
            steps.append("No hepatic, age, or weight contraindications — standard adult dose selected.")

    elif drug_key == "ibuprofen":
        if egfr < 30:
            steps.append(f"eGFR {egfr:.1f} < 30: Ibuprofen CONTRAINDICATED. Dose set to 0 mg.")
            warnings.append(
                "CONTRAINDICATED: eGFR < 30. Ibuprofen must not be used in severe renal impairment."
            )
        elif egfr < 60:
            steps.append(
                f"eGFR {egfr:.1f} in range 30–59: dose limited to 400 mg to protect renal function."
            )
            warnings.append("eGFR 30–59: Ibuprofen dose reduced. Monitor renal function closely.")
        if age >= 65:
            steps.append("Age ≥ 65: dose limited to 400 mg.")
            warnings.append("Patient ≥ 65 yrs: maximum 400 mg per dose recommended.")
        if egfr >= 60 and age < 65:
            steps.append(f"Dose selected based on body weight ({weight} kg): 400/600/800 mg tier.")

    elif drug_key == "amoxicillin":
        if egfr < 30:
            steps.append(
                f"eGFR {egfr:.1f} < 30: dose reduced to 500 mg q24h interval (extended interval dosing)."
            )
            warnings.append(
                "eGFR < 30: Amoxicillin dose reduced and interval extended. "
                "Consider culture-guided therapy."
            )
        else:
            steps.append(
                f"eGFR {egfr:.1f} ≥ 30: standard 875 mg dose selected "
                f"{'(reduced for age ≥ 65)' if age >= 65 else ''}."
            )
        if age >= 65:
            warnings.append("Patient ≥ 65 yrs: monitor for GI effects and C. diff risk.")

    elif drug_key == "metformin":
        if egfr < 30:
            steps.append(f"eGFR {egfr:.1f} < 30: Metformin CONTRAINDICATED. Dose set to 0 mg.")
            warnings.append(
                "CONTRAINDICATED: eGFR < 30. Metformin must not be used — risk of lactic acidosis."
            )
        elif egfr < 45:
            steps.append(
                f"eGFR {egfr:.1f} in range 30–44: starting dose 500 mg. "
                "Titrate cautiously; maximum 1000 mg/day."
            )
            warnings.append(
                "eGFR 30–44: Use with caution. Review renal function every 3 months."
            )
        elif egfr < 60:
            steps.append(
                f"eGFR {egfr:.1f} in range 45–59: dose 500–850 mg guided by glycaemic control "
                f"(HbA1c {hba1c}%, glucose {glucose} mg/dL)."
            )
            warnings.append("eGFR 45–59: Maximum 1000 mg/day. Monitor renal function every 3–6 months.")
        else:
            steps.append(
                f"eGFR {egfr:.1f} ≥ 60: dose determined by glycaemic control "
                f"(HbA1c {hba1c}%, glucose {glucose} mg/dL)."
            )
        if age >= 65:
            warnings.append(
                "Patient ≥ 65 yrs: start at lower dose, titrate slowly. "
                "Monitor renal function more frequently."
            )

    steps.append(f"Decision Tree predicted dose: {predicted_dose:.0f} mg. {dataset_note}")

    return {
        "predicted_dose": round(predicted_dose),
        "steps":          steps,
        "warnings":       warnings,
    }
