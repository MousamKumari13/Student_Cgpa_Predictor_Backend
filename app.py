from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load model and scaler
model = pickle.load(open("svr_model.pkl", "rb"))  # Or use xgb_model.pkl
scaler = pickle.load(open("scaler.pkl", "rb"))

# Mapping functions
def yes_no_to_int(value):
    return 1 if value.lower() == "yes" else 0

def motivation_to_int(value):
    return {"low": 0, "medium": 1, "high": 2}.get(value.lower(), 1)

def health_to_int(value):
    return {"low": 0, "medium": 1, "high": 2}.get(value.lower(), 1)

def stress_to_int(value):
    return {"high": 0, "medium": 1, "low": 2}.get(value.lower(), 1)

@app.route("/predict", methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print("Incoming JSON data:", data)

        # Parse and convert values
        Sem1 = float(data['Sem1'])
        Study_Hr = float(data['Study_Hr'])
        Sleep_Hr = float(data['Sleep_Hr'])
        Backlog = yes_no_to_int(data['Active_backlog'])
        Extra = float(data['ExtraCurriculum'])
        Screen = float(data['Screen_time'])
        Health = health_to_int(data['Health'])
        Stress = stress_to_int(data['stress_level'])
        Motivation = motivation_to_int(data['Motivation'])

        # New engineered feature
        StudySleepCombo = Study_Hr * np.exp(-(Sleep_Hr - 7) ** 2 / 4)

        # Create raw DataFrame
        input_df = pd.DataFrame([[
            Sem1, Study_Hr, Sleep_Hr, Backlog, Extra, Screen,
            Health, Stress, Motivation, StudySleepCombo
        ]], columns=[
            "Sem1", "Study Hours", "Sleeping Hours", "Active Backlog",
            "Extra Curricular", "Screen Time", "Health Condition",
            "Stress Level", "Motivation Level", "StudySleepCombo"
        ])

        # Apply same weights as training
        input_df["Sem1"] *= 1.0
        input_df["Study Hours"] *= 15.0
        input_df["Sleeping Hours"] *= .1
        input_df["Screen Time"] *= 3.5
        input_df["Extra Curricular"] *= 1.5
        input_df["Health Condition"] *= 1.1
        input_df["Stress Level"] *= 1.1
        input_df["Motivation Level"] *= 1.2
        input_df["StudySleepCombo"] *= 10.0

        # Debug: Print final input before scaling
        print("Input to model (after weights, before scaling):\n", input_df)

        # Apply scaler
        input_scaled = scaler.transform(input_df)

        # Convert the scaled input into DataFrame with correct feature names
        input_scaled_df = pd.DataFrame(input_scaled, columns=input_df.columns)


        # Predict
        prediction = model.predict(input_scaled_df)[0]
        prediction = np.clip(prediction, 1, 10)

        return jsonify({'prediction': round(prediction, 2)})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, port=5001)