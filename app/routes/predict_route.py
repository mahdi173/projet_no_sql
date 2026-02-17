# routes/predict_route.py

from flask import Blueprint, request, jsonify
import numpy as np
import pickle
import os
import pandas as pd

predict_bp = Blueprint("predict_bp", __name__)


def load_model(model_path="species_model.pkl"):
    if not os.path.exists(model_path):
        return None

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


model = load_model()


@predict_bp.route("/predict_species", methods=["POST"])
def predict_species():

    if model is None:
        return jsonify({
            "error": "Model not found. Train the model first.",
        }), 503

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

    for key in required_top:
        if key not in data:
            return jsonify({"error": f"Missing field '{key}'"}), 400

    for key in required_features:
        if key not in data["features"]:
            return jsonify({"error": f"Missing field 'features.{key}'"}), 400

    flat = {
        "island": data["island"],
        "sex": data["sex"],
        "features.bill_length_mm": data["features"]["bill_length_mm"],
        "features.bill_depth_mm": data["features"]["bill_depth_mm"],
        "features.flipper_length_mm": data["features"]["flipper_length_mm"],
        "features.body_mass_g": data["features"]["body_mass_g"]
    }

   
    X = pd.DataFrame([flat])
    pred = model.predict(X)[0]

    proba = model.predict_proba(X)[0]

    response = {
        "input_received": data,
        "predicted_species": pred,
        "probabilities": dict(zip(model.classes_, map(float, proba)))
    }

    return jsonify(response), 200