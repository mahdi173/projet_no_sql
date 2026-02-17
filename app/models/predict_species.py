from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)


# ---------------------------------------------------------
# Load model once at startup
# ---------------------------------------------------------
def load_model(model_path="species_model.pkl"):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


model = load_model()


# ---------------------------------------------------------
# Prediction route
# ---------------------------------------------------------
@app.route("/predict_species", methods=["POST"])
def predict_species():

    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()

    required_top = ["island", "sex", "features"]
    required_features = [
        "bill_length_mm",
        "bill_depth_mm",
        "flipper_length_mm",
        "body_mass_g"
    ]

    # Validate fields
    for key in required_top:
        if key not in data:
            return jsonify({"error": f"Missing field '{key}'"}), 400

    for key in required_features:
        if key not in data["features"]:
            return jsonify({"error": f"Missing field 'features.{key}'"}), 400

    # Flatten to fit the model input
    flat = {
        "island": data["island"],
        "sex": data["sex"],
        "features.bill_length_mm": data["features"]["bill_length_mm"],
        "features.bill_depth_mm": data["features"]["bill_depth_mm"],
        "features.flipper_length_mm": data["features"]["flipper_length_mm"],
        "features.body_mass_g": data["features"]["body_mass_g"],
    }

    # Convert to array for sklearn
    X = np.array([list(flat.values())], dtype=object)

    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]

    response = {
        "input_received": data,
        "predicted_species": prediction,
        "probabilities": dict(zip(model.classes_, map(float, probabilities)))
    }

    return jsonify(response), 200


# ---------------------------------------------------------
# Run server
# ---------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)