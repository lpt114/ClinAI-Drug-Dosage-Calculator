# imports
from flask import Flask, render_template, request, redirect
from flask_scss import Scss
from flask_sqlalchemy import SQLAlchemy

# clinAI
ClinAi = Flask(__name__)
Scss(ClinAi)
ClinAi.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///clinai.db"
db = SQLAlchemy(ClinAi)


class patient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    sex = db.Column(db.String(10), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    diagnosis = db.Column(db.String(100), nullable=False)
    allergies = db.Column(db.String(200), nullable=True)
    weight = db.Column(db.Float, nullable=False)
    height = db.Column(db.Float, nullable=False)
    medical_history = db.Column(db.Text, nullable=True)

    def __repr__(self):
        return f"<Patient {self.id}>"


# ---------------------------------------------------------------------------
# Dosage calculation engine
# Each drug entry has:
#   base_dose_per_kg  — mg per kg of body weight per dose
#   max_single_dose   — hard ceiling for a single dose in mg
#   frequency         — how often to take it
#   age_adjustments   — list of (max_age, dose_cap) tuples applied in order
#   contraindications — substrings to look for in allergies / medical history
# ---------------------------------------------------------------------------
DRUG_PROFILES = {
    "acetaminophen": {
        "name": "Acetaminophen (Tylenol)",
        "base_dose_per_kg": 15,       # mg/kg
        "max_single_dose": 1000,      # mg
        "frequency": "every 4–6 hours (max 4 doses/day)",
        "age_adjustments": [
            (2,   125),   # under 2 years → max 125 mg
            (6,   250),   # 2–6 years     → max 250 mg
            (12,  500),   # 6–12 years    → max 500 mg
        ],
        "contraindications": ["acetaminophen", "paracetamol", "liver disease", "hepatic"],
        "notes": "Take with or without food. Do not exceed 4 g/day total.",
    }
    
}


def calculate_dosage(drug_key, p):
    """
    Given a drug key and a patient object, return a dict with:
      - recommended_dose (mg)
      - frequency
      - warnings  (list of strings)
      - notes
    """
    profile = DRUG_PROFILES[drug_key]
    warnings = []

    # Check contraindications against allergies + medical history
    combined_text = " ".join([
        (p.allergies or "").lower(),
        (p.medical_history or "").lower(),
        (p.diagnosis or "").lower(),
    ])
    for contra in profile["contraindications"]:
        if contra in combined_text:
            warnings.append(
                f"⚠️ Possible contraindication detected: '{contra}' found in patient record. "
                "Not recommended to use this drug without further medical review."
            )

    # Elderly caution
    if p.age >= 65:
        warnings.append("⚠️ Patient is 65+: consider starting at lower end of dose range and monitor closely.")

    # Paediatric / underweight caution
    if p.weight < 40 and p.age >= 18:
        warnings.append("⚠️ Low body weight for adult: dose has been adjusted by weight.")

    # Calculate dose
    if profile.get("base_dose_per_kg") is not None:
        dose = profile["base_dose_per_kg"] * p.weight
    else:
        dose = profile.get("fixed_dose", 0)

    # Apply age-based caps
    for max_age, cap in profile.get("age_adjustments", []):
        if p.age <= max_age:
            dose = min(dose, cap)
            break

    # Apply absolute max single dose
    dose = min(dose, profile["max_single_dose"])

    # Elderly reduction — shave 25 % off weight-based drugs
    if p.age >= 65 and profile.get("base_dose_per_kg") is not None:
        dose = dose * 0.75

    dose = round(dose)

    return {
        "drug_name": profile["name"],
        "recommended_dose": dose,
        "frequency": profile["frequency"],
        "warnings": warnings,
        "notes": profile["notes"],
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@ClinAi.route("/", methods=["GET"])
def index():
    return render_template("index.html", drugs=DRUG_PROFILES)


@ClinAi.route("/add_patient", methods=["POST"])
def add_patient():
    """Save new patient data to the database."""
    try:
        new_patient = patient(
            sex=request.form['sex'],
            age=int(request.form['age']),
            diagnosis=request.form['diagnosis'],
            allergies=request.form.get('allergies', ''),
            weight=float(request.form['weight']),
            height=float(request.form['height']),
            medical_history=request.form.get('medical_history', '')
        )
        db.session.add(new_patient)
        db.session.commit()
        return redirect("/")
    except Exception as e:
        print(f"Error adding patient: {e}")
        return render_template("index.html", drugs=DRUG_PROFILES, error=f"Error adding patient: {e}")


@ClinAi.route("/search_patient", methods=["POST"])
def search_patient():
    """Search for a patient by ID and redirect to their page."""
    patient_id = request.form.get('patient_id', '').strip()
    if not patient_id:
        return render_template("index.html", drugs=DRUG_PROFILES, error="Please enter a patient ID.")
    return redirect(f"/patient/{patient_id}")


@ClinAi.route("/patient/<int:patient_id>", methods=["GET"])
def view_patient(patient_id):
    """Display patient information fetched from the database."""
    found_patient = db.session.get(patient, patient_id)
    if found_patient is None:
        return render_template("index.html", drugs=DRUG_PROFILES, error=f"No patient found with ID {patient_id}.")
    return render_template("index.html", drugs=DRUG_PROFILES, found_patient=found_patient)


@ClinAi.route("/calculate_dose", methods=["POST"])
def calculate_dose():
    """Calculate optimal dosage for a patient + drug combination."""
    patient_id = request.form.get('patient_id', '').strip()
    drug_key = request.form.get('drug_type', '').strip()

    if not patient_id:
        return render_template("index.html", drugs=DRUG_PROFILES, error="Please enter a patient ID for dosage calculation.")

    if drug_key not in DRUG_PROFILES:
        return render_template("index.html", drugs=DRUG_PROFILES, error="Unknown drug selected.")

    found_patient = db.session.get(patient, int(patient_id))
    if found_patient is None:
        return render_template("index.html", drugs=DRUG_PROFILES, error=f"No patient found with ID {patient_id}.")

    dosage_result = calculate_dosage(drug_key, found_patient)

    return render_template(
        "index.html",
        drugs=DRUG_PROFILES,
        found_patient=found_patient,
        dosage_result=dosage_result,
    )


if __name__ == "__main__":
    with ClinAi.app_context():
        db.create_all()
    ClinAi.run(debug=True)