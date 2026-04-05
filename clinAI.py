from flask import Flask, render_template, request, redirect
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

app = Flask(__name__)

# Database config
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///clinai.db"
db = SQLAlchemy(app)

# -----------------------------
# Database Models
# -----------------------------
class Patient(db.Model):
    id              = db.Column(db.Integer, primary_key=True)
    sex             = db.Column(db.String(10))
    age             = db.Column(db.Integer)
    diagnosis       = db.Column(db.String(100))
    allergies       = db.Column(db.String(200))
    weight          = db.Column(db.Float)
    height          = db.Column(db.Float)
    medical_history = db.Column(db.Text)
    dose_logs       = db.relationship("DoseLog", backref="patient", lazy=True, order_by="DoseLog.timestamp.desc()")


class DoseLog(db.Model):
    id         = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey("patient.id"), nullable=False)
    drug_key   = db.Column(db.String(50))
    drug_name  = db.Column(db.String(100))
    dose_mg    = db.Column(db.Float)
    frequency  = db.Column(db.String(100))
    notes      = db.Column(db.Text)
    warnings   = db.Column(db.Text)         # pipe-separated
    steps      = db.Column(db.Text)         # pipe-separated explanation steps
    timestamp  = db.Column(db.DateTime, default=datetime.utcnow)


# -----------------------------
# Drug Profiles
# -----------------------------
# Each drug may define:
#   base_dose_per_kg   – mg per kg of body weight
#   max_single_dose    – hard ceiling per dose (mg)
#   frequency          – dosing interval string
#   notes              – general clinical note
#   age_max_dose       – {age_threshold: reduced_max} pairs (oldest threshold wins)
#   renal_adjustment   – True/False (flag for future creatinine-based tuning)
#   contraindications  – list of allergy/condition strings that trigger a warning
#   tuning             – configurable parameters exposed for model flexibility
DRUG_PROFILES = {
    "acetaminophen": {
        "name": "Acetaminophen (Tylenol)",
        "base_dose_per_kg": 15,
        "max_single_dose": 1000,
        "frequency": "Every 4–6 hours (max 4 doses/day)",
        "notes": "Do not exceed 4 g/day total. Use with caution in hepatic impairment.",
        "contraindications": ["acetaminophen", "tylenol", "paracetamol", "liver disease", "hepatic"],
        "renal_adjustment": False,
        "tuning": {
            "dose_per_kg": 15,
            "max_single_dose": 1000,
            "daily_max": 4000
        }
    },
    "ibuprofen": {
        "name": "Ibuprofen (Advil / Motrin)",
        "base_dose_per_kg": 10,
        "max_single_dose": 800,
        "frequency": "Every 6–8 hours with food (max 3200 mg/day)",
        "notes": "Avoid in renal impairment, peptic ulcer disease, and late pregnancy.",
        "contraindications": ["ibuprofen", "nsaid", "aspirin", "renal", "kidney", "peptic", "ulcer"],
        "renal_adjustment": True,
        "tuning": {
            "dose_per_kg": 10,
            "max_single_dose": 800,
            "daily_max": 3200
        }
    },
    "vancomycin": {
        "name": "Vancomycin",
        "base_dose_per_kg": 15,
        "max_single_dose": 3000,
        "frequency": "Every 8–12 hours IV (adjust for renal function)",
        "notes": "Requires therapeutic drug monitoring. Dose heavily adjusted by renal function (eGFR/CrCl).",
        "contraindications": ["vancomycin"],
        "renal_adjustment": True,
        "tuning": {
            "dose_per_kg": 15,
            "max_single_dose": 3000,
            "egfr_severe_threshold": 30,
            "egfr_moderate_threshold": 60,
            "severe_reduction_factor": 0.5,
            "moderate_reduction_factor": 0.75
        }
    },
    "amoxicillin": {
        "name": "Amoxicillin",
        "base_dose_per_kg": 25,
        "max_single_dose": 500,
        "frequency": "Every 8 hours (standard) or every 12 hours (high-dose)",
        "notes": "First-line for many bacterial infections. Reduce dose in severe renal impairment.",
        "contraindications": ["penicillin", "amoxicillin", "ampicillin", "cephalosporin"],
        "renal_adjustment": True,
        "tuning": {
            "dose_per_kg": 25,
            "max_single_dose": 500,
            "daily_max": 1500
        }
    },
    "metformin": {
        "name": "Metformin (Glucophage)",
        "base_dose_per_kg": 0,          # flat dosing, not weight-based
        "flat_dose": 500,
        "max_single_dose": 1000,
        "frequency": "Twice daily with meals (titrate up as tolerated)",
        "notes": "Contraindicated in eGFR < 30. Hold before contrast procedures.",
        "contraindications": ["metformin", "renal", "kidney", "contrast"],
        "renal_adjustment": True,
        "tuning": {
            "starting_dose": 500,
            "max_single_dose": 1000,
            "daily_max": 2000,
            "egfr_contraindication_threshold": 30
        }
    }
}


# -----------------------------
# Dosage Engine  (Sprint 3 — explainability + tuning)
# -----------------------------
def calculate_dosage(drug_key, patient):
    profile  = DRUG_PROFILES[drug_key]
    tuning   = profile.get("tuning", {})
    warnings = []
    steps    = []      # ordered explanation steps

    # ── Step 1: identify base calculation method ──
    if profile.get("flat_dose"):
        base_dose = profile["flat_dose"]
        steps.append(f"Flat starting dose selected (not weight-based): {base_dose} mg")
    else:
        dose_per_kg = tuning.get("dose_per_kg", profile.get("base_dose_per_kg", 0))
        base_dose   = dose_per_kg * patient.weight
        steps.append(
            f"Base dose: {dose_per_kg} mg/kg × {patient.weight} kg = {base_dose:.1f} mg"
        )

    dose = base_dose

    # ── Step 2: apply max single-dose cap ──
    max_dose = tuning.get("max_single_dose", profile.get("max_single_dose", dose))
    if dose > max_dose:
        steps.append(
            f"Dose capped at maximum single dose: {dose:.1f} mg → {max_dose} mg"
        )
        dose = max_dose
    else:
        steps.append(f"Dose is within maximum single-dose limit ({max_dose} mg). No cap applied.")

    # ── Step 3: age adjustment ──
    if patient.age >= 65:
        pre_age = dose
        dose    = dose * 0.75
        steps.append(
            f"Age adjustment (≥65 years): dose reduced by 25% — {pre_age:.1f} mg → {dose:.1f} mg"
        )
        warnings.append("Patient is ≥65 years old. Reduced dose applied; monitor closely for adverse effects.")
    elif patient.age < 18:
        steps.append(f"Patient is a minor (age {patient.age}). Paediatric weight-based dosing used.")
        warnings.append("Paediatric patient — verify dose with paediatric formulary before use.")
    else:
        steps.append(f"No age-based adjustment required (age {patient.age}).")

    # ── Step 4: renal adjustment (vancomycin / ibuprofen / amoxicillin) ──
    if profile.get("renal_adjustment"):
        # In a real system this would use creatinine/eGFR from the patient record.
        # We surface a visible warning so the clinician knows to act on it.
        steps.append("Renal adjustment flag is active for this drug.")
        warnings.append(
            f"{profile['name']} requires renal dose adjustment. Review current eGFR/CrCl and "
            "adjust frequency or dose accordingly before administering."
        )

    # ── Step 5: contraindication / allergy check ──
    allergies    = (patient.allergies or "").lower()
    history      = (patient.medical_history or "").lower()
    combined     = allergies + " " + history
    contras      = profile.get("contraindications", [])
    triggered    = [c for c in contras if c in combined]
    if triggered:
        warnings.append(
            f"Possible contraindication detected in patient record ({', '.join(triggered)}). "
            "Verify allergy and medication history before proceeding."
        )
        steps.append(
            f"Contraindication check: match found for [{', '.join(triggered)}] in patient record."
        )
    else:
        steps.append("Contraindication check: no conflicts detected in documented allergies or history.")

    # ── Step 6: comparative context ──
    comparative = _comparative_context(drug_key, dose, patient)
    steps.append(comparative)

    # ── Final dose ──
    final_dose = round(dose)

    return {
        "drug_name":        profile["name"],
        "recommended_dose": final_dose,
        "frequency":        profile.get("frequency", "As directed"),
        "notes":            profile.get("notes", ""),
        "warnings":         warnings,
        "steps":            steps,
    }


def _comparative_context(drug_key, dose, patient):
    """Generate a plain-language comparative sentence for explainability."""
    profile     = DRUG_PROFILES[drug_key]
    max_dose    = profile.get("tuning", {}).get("max_single_dose", profile.get("max_single_dose", 1))
    pct_of_max  = (dose / max_dose) * 100 if max_dose else 0

    if pct_of_max >= 100:
        level = "at the maximum permitted single dose"
    elif pct_of_max >= 75:
        level = f"in the upper range ({pct_of_max:.0f}% of the single-dose ceiling)"
    elif pct_of_max >= 40:
        level = f"in the mid range ({pct_of_max:.0f}% of the single-dose ceiling)"
    else:
        level = f"in the lower range ({pct_of_max:.0f}% of the single-dose ceiling)"

    return (
        f"Final recommended dose of {round(dose)} mg is {level} for {profile['name']}. "
        f"Patient weight ({patient.weight} kg) and age ({patient.age} yrs) are the primary drivers."
    )


# -----------------------------
# Routes
# -----------------------------

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/patients")
def patient_list():
    patients = Patient.query.order_by(Patient.id.desc()).all()
    return render_template("patients.html", patients=patients)


@app.route("/add_patient", methods=["POST"])
def add_patient():
    new_patient = Patient(
        sex             = request.form["sex"],
        age             = int(request.form["age"]),
        diagnosis       = request.form["diagnosis"],
        allergies       = request.form.get("allergies", ""),
        weight          = float(request.form["weight"]),
        height          = float(request.form["height"]),
        medical_history = request.form.get("medical_history", "")
    )
    db.session.add(new_patient)
    db.session.commit()
    return redirect(f"/patient/{new_patient.id}")


@app.route("/search_patient", methods=["POST"])
def search_patient():
    patient_id = request.form["patient_id"]
    patient = db.session.get(Patient, int(patient_id))
    if not patient:
        return render_template("error.html", message=f"No patient found with ID {patient_id}.")
    return redirect(f"/patient/{patient_id}")


@app.route("/patient/<int:patient_id>")
def view_patient(patient_id):
    patient = db.session.get(Patient, patient_id)
    if not patient:
        return render_template("error.html", message=f"No patient found with ID {patient_id}.")
    return render_template("patient.html", patient=patient, drugs=DRUG_PROFILES)


@app.route("/patient/<int:patient_id>/delete", methods=["POST"])
def delete_patient(patient_id):
    patient = db.session.get(Patient, patient_id)
    if patient:
        DoseLog.query.filter_by(patient_id=patient_id).delete()
        db.session.delete(patient)
        db.session.commit()
    return redirect("/patients")


@app.route("/patient/<int:patient_id>/edit", methods=["GET", "POST"])
def edit_patient(patient_id):
    patient = db.session.get(Patient, patient_id)
    if not patient:
        return render_template("error.html", message=f"No patient found with ID {patient_id}.")
    if request.method == "POST":
        patient.sex             = request.form["sex"]
        patient.age             = int(request.form["age"])
        patient.diagnosis       = request.form["diagnosis"]
        patient.allergies       = request.form.get("allergies", "")
        patient.weight          = float(request.form["weight"])
        patient.height          = float(request.form["height"])
        patient.medical_history = request.form.get("medical_history", "")
        db.session.commit()
        return redirect(f"/patient/{patient_id}")
    return render_template("edit_patient.html", patient=patient)


@app.route("/calculate_dose", methods=["POST"])
def calculate_dose():
    patient_id = int(request.form["patient_id"])
    drug_type  = request.form["drug_type"]
    patient    = db.session.get(Patient, patient_id)

    if not patient:
        return render_template("error.html", message=f"No patient found with ID {patient_id}.")
    if drug_type not in DRUG_PROFILES:
        return render_template("error.html", message=f"Unknown drug '{drug_type}'.")

    result = calculate_dosage(drug_type, patient)

    # Persist dose log
    log = DoseLog(
        patient_id = patient_id,
        drug_key   = drug_type,
        drug_name  = result["drug_name"],
        dose_mg    = result["recommended_dose"],
        frequency  = result["frequency"],
        notes      = result["notes"],
        warnings   = "|".join(result["warnings"]),
        steps      = "|".join(result["steps"]),
    )
    db.session.add(log)
    db.session.commit()

    return render_template(
        "patient.html",
        patient      = patient,
        drugs        = DRUG_PROFILES,
        dosage_result= result
    )


@app.route("/drug/<drug_key>")
def drug_info(drug_key):
    profile = DRUG_PROFILES.get(drug_key)
    if not profile:
        return render_template("error.html", message=f"No drug profile found for '{drug_key}'.")
    return render_template("drug.html", drug_key=drug_key, profile=profile)


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)
