from flask import Flask, request, jsonify
import joblib
import json
import numpy as np
import os
from datetime import datetime
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)  
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

# Paths to model files
model_path = os.path.join("..", "Model", "ensemble_model.joblib")
scaler_path = os.path.join("..", "Model", "scaler.joblib")
label_encoders_path = os.path.join("..", "Model", "label_encoders.json")

# Load model and preprocessing tools
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

with open(label_encoders_path, "r") as f:
    label_encoders = json.load(f)

@app.route("/")
def index():
    return "🚀 Bike Price Prediction API is Live!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Parse request JSON
        data = request.json
        required_fields = ["Bike Type", "Brand", "Edition", "Model", "Year", "Mileage", "Capacity"]

        # Check for missing fields
        missing = [f for f in required_fields if f not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

        # Handle encodings
        def safe_encode(category, field):
            return label_encoders.get(field, {}).get(str(category), 0)

        encoded_input = [
            safe_encode(data["Bike Type"], "Bike Type"),
            safe_encode(data["Brand"], "Brand"),
            safe_encode(data["Edition"], "Edition"),
            safe_encode(data["Model"], "Model")
        ]

        # Compute features
        current_year = datetime.now().year
        year_gap = current_year - int(data["Year"])
        age = np.log1p(year_gap)
        mileage = np.log1p(float(data["Mileage"]))
        capacity = float(data["Capacity"])

        # Construct feature vector
        features = encoded_input + [age, mileage, capacity]

        # Scale
        scaled = scaler.transform([features])

        # Predict
        prediction = model.predict(scaled)[0]

        return jsonify({
            "predicted_price": round(float(prediction), 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
