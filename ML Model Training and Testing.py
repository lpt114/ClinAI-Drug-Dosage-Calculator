import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# -------------------------
# LOAD DATA
# -------------------------
df = pd.read_csv("C:/Users/brady/Downloads/vancomycin_dosing_dataset_1200.csv")

# Features and target
X = df[['anchor_age', 'weight_kg', 'creatinine', 'eGFR']]
y = df['dose_mg']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeRegressor(max_depth=4)
model.fit(X_train, y_train)

# -------------------------
# FUNCTION: GET RECOMMENDATION
# -------------------------
def get_recommendation(patient_id):
    
    # Find patient
    patient = df[df['subject_id'] == patient_id]
    
    if patient.empty:
        print("Patient not found.")
        return
    
    # Extract features
    features = patient[['anchor_age', 'weight_kg', 'creatinine', 'eGFR']]
    
    age = float(patient['anchor_age'].iloc[0])
    weight = float(patient['weight_kg'].iloc[0])
    creatinine = float(patient['creatinine'].iloc[0])
    egfr = float(patient['eGFR'].iloc[0])
    
    # Predict dose
    predicted_dose = model.predict(features)[0]
    
    # Calculate base dose
    base_dose = 15 * weight
    
    # Adjustment factor
    adjustment_factor = predicted_dose / base_dose
    
    # -------------------------
    # OUTPUT
    # -------------------------
    print("\n--- ClinAI Assist Recommendation ---")
    print(f"Patient ID: {patient_id}")
    
    print("\nPatient Factors:")
    print(f"Age: {age}")
    print(f"Weight: {weight:.2f} kg")
    print(f"Creatinine: {creatinine:.2f}")
    print(f"eGFR: {egfr:.2f}")
    
    print("\nDosing Calculation:")
    print(f"Base Dose (15 mg/kg): {base_dose:.2f} mg")
    print(f"Adjustment Factor: {adjustment_factor:.2f}")
    
    print("\nRecommended Dose:")
    print(f"{predicted_dose:.2f} mg")
    
    # -------------------------
    # SIMPLE EXPLANATION
    # -------------------------
    print("\nExplanation:")
    
    if egfr < 30:
        print("- Significant dose reduction due to poor kidney function")
    elif egfr < 60:
        print("- Moderate dose reduction due to reduced kidney function")
    else:
        print("- Normal kidney function, standard dosing applied")
    
    if weight > 90:
        print("- Higher dose influenced by above-average body weight")
    elif weight < 60:
        print("- Lower dose influenced by lower body weight")

# -------------------------
# USER INPUT LOOP
# -------------------------
while True:
    user_input = input("\nEnter Patient ID (or type 'exit' to quit): ")
    
    if user_input.lower() == 'exit':
        print("Exiting ClinAI Assist.")
        break
    
    try:
        patient_id = int(user_input)
        get_recommendation(patient_id)
    except ValueError:
        print("Invalid input. Please enter a numeric Patient ID.")