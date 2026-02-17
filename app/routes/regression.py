from flask import Blueprint, jsonify, send_file
from models.regression import regression_simple, regression_multiple
import os

reg = Blueprint("regression", __name__)

@reg.route("/simple")
def reg_simple_route():
    regression_simple()
    
    filepath = "/app/plots/reg_simple.png"
    return send_file(filepath, mimetype="image/png")


@reg.route("/simple/predict/<int:value>")
def reg_simple_predict(value):
    model, prediction = regression_simple(value)

    return {
        "flipper_length_mm": value,
        "predicted_body_mass_g": prediction
    }

@reg.route("/multiple")
def reg_multi_route():
    regression_multiple()

    filepath = "/app/plots/reg_multi.png"
    return send_file(filepath, mimetype="image/png")

@reg.route("/multiple/predict/<int:flipper>/<int:bill_length>/<int:bill_depth>")
def reg_multiple_predict(flipper, bill_length, bill_depth):
    model, prediction = regression_multiple(flipper, bill_length, bill_depth)

    return {
        "flipper_length_mm": flipper,
        "bill_length_mm": bill_length,
        "bill_depth_mm": bill_depth,
        "predicted_body_mass_g": prediction
    }