from flask import Flask, request, jsonify
import pickle
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load trained models
#with open("Lr_model.pkl", "rb") as f:
 #   Lr_Model = pickle.load(f)
#with open("DT_model.pkl", "rb") as f:
 #   DT_Model = pickle.load(f)
#with open("Rf_model.pkl", "rb") as f:
 #   Rf_Model = pickle.load(f)
with open("gb_model.pkl", "rb") as f:
     gb_model = pickle.load(f)
#with open("xgb_model.pkl", "rb") as f:
 #   xgb_model = pickle.load(f)
#with open("knn_model.pkl", "rb") as f:
 #   knn_model = pickle.load(f)
#with open("svr_model.pkl", "rb") as f:
 #   svr_model = pickle.load(f)

# ✅ Use best performing model (change if needed)
best_model = gb_model  # You can change to xgb_model if that performs better

# Mapping functions
def yes_no_to_int(value):
    return 1 if value.lower() == "yes" else 0

def motivation_to_int(value):
    mapping = {"low": 0, "medium": 1, "high": 2}
    return mapping.get(value.lower(), 1)

def health_to_int(value):
    mapping = {"low": 0, "medium": 1, "high": 2}
    return mapping.get(value.lower(), 1)

def stress_to_int(value):
    mapping = {"high": 0, "medium": 1, "low": 2}
    return mapping.get(value.lower(), 1)

@app.route("/predict", methods=['POST'])
def predict():
    data = request.get_json()

    try:
        Sem1 = float(data['Sem1'])
        Study_Hr = float(data['Study_Hr'])
        Sleep_Hr = float(data['Sleep_Hr'])
        Active_backlog = yes_no_to_int(data['Active_backlog'])
        ExtraCurriculum = float(data['ExtraCurriculum'])
        Screen_time = float(data['Screen_time'])
        Health = health_to_int(data['Health'])
        stress_level = stress_to_int(data['stress_level'])
        Motivation = motivation_to_int(data['Motivation'])

        input_features = np.array([[Sem1, Study_Hr, Sleep_Hr, Active_backlog,
                                    ExtraCurriculum, Screen_time, Health,
                                    stress_level, Motivation]])

        prediction = best_model.predict(input_features)[0]
        prediction = np.clip(prediction, 1, 10)

        return jsonify({'prediction': round(prediction, 2)})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True , port=5001)