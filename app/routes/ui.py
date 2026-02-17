from flask import Blueprint, render_template, request, send_file, jsonify
import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os
import json
import base64

from routes.predict_route import load_model, model as loaded_model
from db.get_db import get_database, set_db_type, get_db_type

# Sklearn imports for evaluation pipeline reconstruction
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

ui_bp = Blueprint("ui_bp", __name__, template_folder="../templates")

# ------------------------------------------------------------------------------
# DB Switching
# ------------------------------------------------------------------------------

@ui_bp.route("/set_db/<db_type>", methods=["POST"])
def switch_db(db_type):
    try:
        new_type = set_db_type(db_type)
        return jsonify({"status": "success", "current_db": new_type})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@ui_bp.route("/current_db", methods=["GET"])
def current_db_status():
    return jsonify({"current_db": get_db_type()})

# ------------------------------------------------------------------------------
# Helper: Load Dataframe from Current DB
# ------------------------------------------------------------------------------

def _load_df_from_current_db():
    db = get_database()
    # All DB implementations should support find({}) -> list of dicts
    # However, structure might vary slightly (e.g. Cassandra objects vs dicts)
    
    raw_data = db.find({})
    
    # helper to normalize dicts
    docs = []
    for d in raw_data:
        # If it's a Cassandra Row object, convert to dict
        if not isinstance(d, dict):
             # assuming it has _fields or __dict__ or we map manually
            docs.append({
                "species": getattr(d, "species", None),
                "island": getattr(d, "island", None),
                "sex": getattr(d, "sex", None),
                "features.bill_length_mm": getattr(d, "bill_length_mm", None),
                "features.bill_depth_mm": getattr(d, "bill_depth_mm", None),
                "features.flipper_length_mm": getattr(d, "flipper_length_mm", None),
                "features.body_mass_g": getattr(d, "body_mass_g", None),
            })
        else:
             docs.append(d)

    df = pd.json_normalize(docs)
    
    # Rename features.x -> x if needed for consistency, or keep as is?
    # The training script expects 'features.bill_length_mm' etc.
    # But for visualization we might want short names.
    # Let's keep them as is and handle in visualization.
    return df

# ------------------------------------------------------------------------------
# Visualization
# ------------------------------------------------------------------------------

@ui_bp.route("/visualize_fragment", methods=["GET"])
def visualize_fragment():
    return render_template("visualize.html", variables=[
        "features.bill_length_mm", "features.bill_depth_mm", 
        "features.flipper_length_mm", "features.body_mass_g", 
        "island", "sex", "species"
    ])

@ui_bp.route("/plot_distribution", methods=["POST"])
def plot_distribution():
    var = request.form.get("variable")
    plot_type = request.form.get("plot_type")
    
    df = _load_df_from_current_db()
    if df.empty:
        return send_file(io.BytesIO(), mimetype="image/png") # Return empty image

    buf = io.BytesIO()
    plt.figure(figsize=(6, 4))
    
    try:
        if plot_type == "boxplot":
            sns.boxplot(x=df[var])
        else:
            sns.histplot(df[var].dropna(), kde=False)
        plt.title(f"{plot_type.capitalize()} of {var}")
    except Exception as e:
        plt.text(0.5, 0.5, f"Error: {e}", ha='center')
        
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    return send_file(buf, mimetype="image/png")

@ui_bp.route("/correlation", methods=["POST"])
def correlation():
    df = _load_df_from_current_db()
    
    # Filter for numeric columns
    numeric_cols = [c for c in df.columns if "features." in c or "mm" in c or "g" in c]
    numeric_df = df[numeric_cols].dropna()
    
    buf = io.BytesIO()
    plt.figure(figsize=(6, 5))
    
    if not numeric_df.empty:
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
        plt.title("Correlation matrix")
    else:
        plt.text(0.5, 0.5, "No numeric data found", ha='center')
        
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    return send_file(buf, mimetype="image/png")


# ------------------------------------------------------------------------------
# Prediction
# ------------------------------------------------------------------------------

@ui_bp.route("/predict_fragment", methods=["GET"])
def predict_fragment():
    return render_template("predict.html")

# Existing /predict_submit can remain, but let's make it return JSON for AJAX
@ui_bp.route("/predict_json", methods=["POST"])
def predict_json():
    data = request.json or request.form.to_dict()
    
    try:
        features = {
            "bill_length_mm": float(data.get("bill_length_mm")),
            "bill_depth_mm": float(data.get("bill_depth_mm")),
            "flipper_length_mm": float(data.get("flipper_length_mm")),
            "body_mass_g": float(data.get("body_mass_g")),
        }
    except ValueError:
        return jsonify({"error": "Invalid numeric inputs"}), 400

    params = {
        "island": data.get("island"),
        "sex": data.get("sex"),
        "features": features
    }
    
    if loaded_model is None:
        return jsonify({"error": "Model not loaded"}), 503
        
    # Prepare DF for model
    flat = {
        "island": params["island"],
        "sex": params["sex"],
        "features.bill_length_mm": params["features"]["bill_length_mm"],
        "features.bill_depth_mm": params["features"]["bill_depth_mm"],
        "features.flipper_length_mm": params["features"]["flipper_length_mm"],
        "features.body_mass_g": params["features"]["body_mass_g"],
    }
    X = pd.DataFrame([flat])
    
    try:
        pred = loaded_model.predict(X)[0]
        proba = loaded_model.predict_proba(X)[0]
        probs = dict(zip(loaded_model.classes_, map(float, proba)))
        
        return jsonify({
            "predicted_species": pred,
            "probabilities": probs
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ------------------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------------------

@ui_bp.route("/evaluate_fragment", methods=["GET"])
def evaluate_fragment():
    return render_template("evaluate.html")

@ui_bp.route("/run_evaluation", methods=["POST"])
def run_evaluation():
    # 1. Load data from CURRENT DB
    df = _load_df_from_current_db()
    
    target = "species"
    num_features = [
        "features.bill_length_mm",
        "features.bill_depth_mm",
        "features.flipper_length_mm",
        "features.body_mass_g"
    ]
    # Handle column name variations (some might not have 'features.' prefix if coming from flattened dicts)
    # But _load_df_from_current_db should handle normalization if we enforced it. 
    # Let's double check columns exist.
    
    missing_cols = [c for c in num_features if c not in df.columns]
    if missing_cols:
         # try without 'features.' prefix
         pass 

    cat_features = ["island", "sex"]
    
    if target not in df.columns:
        return jsonify({"error": f"Target '{target}' not found in data (DB might be empty)"}), 400
        
    df = df.dropna(subset=[target])
    
    if len(df) < 10:
         return jsonify({"error": "Not enough data (n<10) to evaluate"}), 400

    X = df[num_features + cat_features]
    y = df[target]

    # 2. Logic to Train/Evaluate on the fly
    # We reconstruct the pipeline here just solely for evaluation metric generation 
    # (Train-Test split approach as per original)
    
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_features),
            ("cat", categorical_transformer, cat_features)
        ]
    )
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("rf", rf_model)])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    # CM Image
    buf = io.BytesIO()
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix (Validation Set)")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")

    return jsonify({
        "accuracy": round(acc, 4),
        "report": report,
        "cm_image": img_b64,
        "n_samples": len(df)
    })
