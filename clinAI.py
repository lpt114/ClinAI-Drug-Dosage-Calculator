from flask import Flask, render_template, request, redirect
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)

# Database config
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///clinai.db"
db = SQLAlchemy(app)

# -----------------------------
# Database Model
# -----------------------------
class Patient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    sex = db.Column(db.String(10))
    age = db.Column(db.Integer)
    diagnosis = db.Column(db.String(100))
    allergies = db.Column(db.String(200))
    weight = db.Column(db.Float)
    height = db.Column(db.Float)
    medical_history = db.Column(db.Text)

# -----------------------------
# Drug Profiles
# -----------------------------
DRUG_PROFILES = {
    "acetaminophen": {
        "name": "Acetaminophen (Tylenol)",
        "base_dose_per_kg": 15,
        "max_single_dose": 1000,
        "frequency": "every 4–6 hours (max 4 doses/day)",
        "notes": "Do not exceed 4 g/day"
    },
    "drugA": {"name": "Drug A"},
    "drugB": {"name": "Drug B"}
}

# -----------------------------
# Dosage Engine
# -----------------------------
def calculate_dosage(drug_key, patient):

    profile = DRUG_PROFILES[drug_key]

    if "base_dose_per_kg" in profile:
        dose = profile["base_dose_per_kg"] * patient.weight
        dose = min(dose, profile["max_single_dose"])
    else:
        dose = 100

    return {
        "drug_name": profile["name"],
        "recommended_dose": round(dose),
        "frequency": profile.get("frequency", "Twice daily"),
        "notes": profile.get("notes", "Standard dose")
    }

# -----------------------------
# Routes
# -----------------------------

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/add_patient", methods=["POST"])
def add_patient():

    new_patient = Patient(
        sex=request.form["sex"],
        age=int(request.form["age"]),
        diagnosis=request.form["diagnosis"],
        allergies=request.form["allergies"],
        weight=float(request.form["weight"]),
        height=float(request.form["height"]),
        medical_history=request.form["medical_history"]
    )

    db.session.add(new_patient)
    db.session.commit()

    return redirect(f"/patient/{new_patient.id}")


@app.route("/search_patient", methods=["POST"])
def search_patient():

    patient_id = request.form["patient_id"]

    return redirect(f"/patient/{patient_id}")


@app.route("/patient/<int:patient_id>")
def view_patient(patient_id):

    patient = db.session.get(Patient, patient_id)

    if not patient:
        return "Patient not found"

    return render_template(
        "patient.html",
        patient=patient,
        drugs=DRUG_PROFILES
    )


@app.route("/calculate_dose", methods=["POST"])
def calculate_dose():

    patient_id = int(request.form["patient_id"])
    drug_type = request.form["drug_type"]

    patient = db.session.get(Patient, patient_id)

    dosage_result = calculate_dosage(drug_type, patient)

    return render_template(
        "patient.html",
        patient=patient,
        drugs=DRUG_PROFILES,
        dosage_result=dosage_result
    )


if __name__ == "__main__":
    with app.app_context():
        db.create_all()

    app.run(debug=True)
