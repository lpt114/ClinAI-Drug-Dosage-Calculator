# ClinAI — AI-Assisted Drug Dosage Calculator

ClinAI is a Flask-based clinical decision-support web application that recommends personalized drug dosages using machine learning models trained on clinical data. The system considers patient-specific factors such as weight, age, renal function, allergies, and medical history to generate data-driven dosage recommendations with explainability.

> ⚠️ **Educational Use Only.** ClinAI is a decision-support tool intended for academic and educational purposes. It does not diagnose or prescribe medication and requires clinical judgment for any real-world application.

🌐 **Live Application:** https://roberthbhm441.pythonanywhere.com/

---

## Table of Contents

- [Project Overview](#project-overview)
- [Tech Stack](#tech-stack)
- [Repository Structure](#repository-structure)
- [Prerequisites](#prerequisites)
- [Installation & Setup](#installation--setup)
- [Running the Application](#running-the-application)
- [Using the Application](#using-the-application)
- [Supported Drugs](#supported-drugs)
- [ML Model — Vancomycin](#ml-model--vancomycin)
- [Dosage Engine Logic](#dosage-engine-logic)
- [Database](#database)

---

## Project Overview

ClinAI allows a user to:

1. Add patients with clinical attributes (age, sex, weight, height, diagnosis, allergies, medical history, creatinine, eGFR)
2. Select a drug and calculate a recommended dose using trained machine learning models
3. View a step-by-step explanation of how the dose was derived
4. See warnings for contraindications, renal impairment, and age-based adjustments
5. Browse a full patient list and dose log history

All drug dosing predictions are powered by machine learning models trained on clinical datasets included in the repository.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3, Flask, Flask-SQLAlchemy |
| Database | SQLite (auto-created on first run) |
| ML | scikit-learn (Decision Tree Regressors), pandas, NumPy |
| Frontend | HTML/CSS (Jinja2 templates) |
| Model persistence | pickle |

---

## Repository Structure

```
ClinAI-Drug-Dosage-Calculator/
│
├── clinAI.py                         # Main Flask app — routes, DB models, dosage engine
├── ml_model.py                       # ML model training + prediction (Vancomycin)
├── vancomycin_dosing_dataset_1200.csv  # Training dataset for the ML model
│
├── templates/                        # Jinja2 HTML templates
│   ├── index.html                    # Home page
│   ├── patients.html                 # Patient list
│   ├── patient.html                  # Individual patient view + dose calculator
│   ├── edit_patient.html             # Edit patient form
│   ├── drug.html                     # Drug profile info page
│   └── error.html                    # Error display page
│
├── static/                           # CSS and static assets
├── instance/                         # Runtime files (SQLite DB created here)
│
├── ARCHITECTURE.md                   # Architecture notes
├── ML Model Training and Testing.py  # Standalone ML training/testing script
├── ClinAIV1.zip                      # Archived V1 source
└── mimic-iv-clinical-database-demo-2.2.zip  # Reference clinical dataset
```

---

## Prerequisites

- **Python 3.9 or higher** — [Download here](https://www.python.org/downloads/)
- **pip** (comes bundled with Python)

Verify your installation:
```bash
python --version
pip --version
```

---

## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/lpt114/ClinAI-Drug-Dosage-Calculator.git
cd ClinAI-Drug-Dosage-Calculator
```

### 2. (Recommended) Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:

- **macOS / Linux:**
  ```bash
  source venv/bin/activate
  ```
- **Windows:**
  ```bash
  venv\Scripts\activate
  ```

### 3. Install Dependencies

```bash
pip install flask flask-sqlalchemy scikit-learn pandas numpy
```

All required packages at a glance:

| Package | Purpose |
|---|---|
| `flask` | Web framework |
| `flask-sqlalchemy` | ORM for SQLite |
| `scikit-learn` | Decision Tree ML model |
| `pandas` | CSV loading for model training |
| `numpy` | Feature array construction for prediction |

---

## Running the Application

From the project root directory (with your virtual environment active), run:

```bash
python clinAI.py
```

You should see output similar to:

```
 * Running on http://127.0.0.1:5000
 * Debug mode: on
```

Open your browser and go to:

```
http://127.0.0.1:5000
```

> **What happens on first run:**
> - The SQLite database (`instance/clinai.db`) is automatically created.
> - The ML model (`model.pkl`) is trained on `vancomycin_dosing_dataset_1200.csv` and saved to disk. This only happens once; subsequent runs load the saved model.

---

## Using the Application

### Add a Patient

1. From the home page, click **Add Patient** or navigate to `/patients`.
2. Fill in the patient form:
   - **Required:** Sex, Age, Diagnosis, Weight (kg), Height (cm)
   - **Optional but recommended for Vancomycin:** Creatinine (mg/dL), eGFR (mL/min/1.73m²)
   - Allergies and Medical History are used for contraindication checks
3. Submit — you will be redirected to the patient's profile page.

### Calculate a Dose

1. On the patient's profile page, select a drug from the dropdown.
2. For **Vancomycin**, enter or confirm the Creatinine and eGFR values (required).
3. Click **Calculate Dose**.
4. The recommended dose, frequency, step-by-step reasoning, and any warnings will appear on the same page.
5. Each calculation is automatically saved to the patient's dose log.

### Search for a Patient

- Use the search bar on the home page and enter the patient's ID number.

### View All Patients

- Navigate to `/patients` to see the full patient list.

### View Drug Profiles

- Navigate to `/drug/<drug_key>` (e.g., `/drug/vancomycin`) to see a drug's clinical profile.

---

## Supported Drugs

All supported drugs use machine learning models for dosage prediction.

| Drug | Key | Model Type | Inputs |
|---|---|---|---|
| Acetaminophen (Tylenol) | `acetaminophen` | Decision Tree Regressor | Age, weight, clinical factors |
| Ibuprofen (Advil/Motrin) | `ibuprofen` | Decision Tree Regressor | Age, weight, renal indicators |
| Vancomycin | `vancomycin` | Decision Tree Regressor | Age, weight, creatinine, eGFR |
| Amoxicillin | `amoxicillin` | Decision Tree Regressor | Age, weight, clinical factors |
| Metformin (Glucophage) | `metformin` | Decision Tree Regressor | Age, renal function (eGFR), clinical factors |

---

## Machine Learning Models

ClinAI uses separate machine learning models for each supported drug. Each model is trained on structured clinical datasets included in the repository.

**Common input features include:**
- Age (years)
- Weight (kg)
- Serum creatinine (mg/dL)
- eGFR (mL/min/1.73m²)
- Additional drug-specific clinical features

**Model outputs:**
- Predicted dose (mg)
- Supporting context (feature-driven reasoning)
- Safety flags and warnings

Models are trained using Decision Tree Regressors and persisted using `pickle`.

> On first run, models are trained and saved locally. If model files already exist, they are loaded automatically.

To retrain models:
```bash
# Delete existing .pkl files and rerun the app
python clinAI.py

---

```md
## Explainability & Safety Checks

ClinAI emphasizes transparency by providing:

- Model-driven dosage predictions
- Plain-language explanations of key contributing factors
- Warnings for:
  - Renal impairment
  - Age-related risk
  - Contraindications based on allergies and medical history

Although predictions are generated via machine learning, safety checks and clinical warnings are applied as an additional layer of validation.

---

## Database

ClinAI uses SQLite via Flask-SQLAlchemy. The database file is created automatically at `instance/clinai.db` on first run.

**Tables:**

- `patient` — Stores demographics and clinical attributes
- `dose_log` — Stores every dose calculation, linked to a patient, including the full step-by-step explanation and warnings

No manual database setup is required.
