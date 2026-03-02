ClinAI Architecture Notes
ClinAI is a Flask application that stores patient information in a SQLite database and generates a rule-based dosage recommendation for a selected drug. The user interacts through a single rendered UI page using templates.

System Flow
The home page lists available drugs. A patient can be added to the database using the add patient route. A patient can be loaded back into the UI by searching their ID. When calculating dosage, the system fetches the patient from the database, runs the dosage engine for the selected drug, then renders the result back on the same page.

Key Components
clinAI.py contains the Flask app setup, database model, drug profiles, dosage calculation function, and routes. Templates contain the HTML used to render the UI. static contains styling and assets. An instance contains runtime app files such as the SQLite database.

Dosage Engine
Drug rules are stored in DRUG_PROFILES. Dosage is calculated deterministically using explicit rule logic such as weight-based dosing, age caps, max single dose limits, and warnings based on contraindications and patient factors.

Scope Guardrails
ClinAI is decision support only and intended for educational use. It does not diagnose or prescribe and requires clinical judgment for real-world decisions.
