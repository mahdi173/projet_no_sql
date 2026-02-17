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
from models.train_species_model import load_data_from_mongo
from routes.predict_route import load_model
from db.cassandra_db import CassandraDB
import redis as redis_lib
import json
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

ui_bp = Blueprint("ui_bp", __name__, template_folder="../templates")


@ui_bp.route("/visualize", methods=["GET"])
def visualize_page():
    variables = [
        "bill_length_mm",
        "bill_depth_mm",
        "flipper_length_mm",
        "body_mass_g",
        "island",
        "sex",
        "species",
    ]
    return render_template("visualize.html", variables=variables)


def _load_df():
    df = load_data_from_mongo()
    df = df.rename(columns=lambda c: c.replace("features.", ""))
    return df


@ui_bp.route("/plot_distribution", methods=["POST"])
def plot_distribution():
    var = request.form.get("variable") or request.json.get("variable")
    plot_type = request.form.get("plot_type") or request.json.get("plot_type")

    df = _load_df()
    buf = io.BytesIO()

    plt.figure()
    if plot_type == "boxplot":
        sns.boxplot(x=df[var])
    else:
        # histogram by default
        sns.histplot(df[var].dropna(), kde=False)

    plt.title(f"{plot_type.capitalize()} of {var}")
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    return send_file(buf, mimetype="image/png")


@ui_bp.route("/correlation", methods=["POST"])
def correlation():
    var = request.form.get("variable") or request.json.get("variable")
    df = _load_df()

    numeric = df[["body_mass_g", "flipper_length_mm", "bill_length_mm", "bill_depth_mm"]].dropna()
    buf = io.BytesIO()
    plt.figure(figsize=(6, 5))
    sns.heatmap(numeric.corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation matrix")
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    return send_file(buf, mimetype="image/png")


@ui_bp.route("/predict", methods=["GET"]) 
def predict_page():
    return render_template("predict.html")


@ui_bp.route("/predict_submit", methods=["POST"]) 
def predict_submit():
    data = request.form.to_dict()
    # build payload expected by model
    try:
        features = {
            "bill_length_mm": float(data.get("bill_length_mm")),
            "bill_depth_mm": float(data.get("bill_depth_mm")),
            "flipper_length_mm": float(data.get("flipper_length_mm")),
            "body_mass_g": float(data.get("body_mass_g")),
        }
    except Exception:
        return jsonify({"error": "Invalid numeric inputs"}), 400

    payload = {
        "island": data.get("island"),
        "sex": data.get("sex"),
        "features": features
    }

    model = load_model()
    if model is None:
        return jsonify({"error": "Model not found. Train the model first."}), 503

    flat = {
        "island": payload["island"],
        "sex": payload["sex"],
        "features.bill_length_mm": payload["features"]["bill_length_mm"],
        "features.bill_depth_mm": payload["features"]["bill_depth_mm"],
        "features.flipper_length_mm": payload["features"]["flipper_length_mm"],
        "features.body_mass_g": payload["features"]["body_mass_g"],
    }

    X = pd.DataFrame([flat])
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]

    return render_template("predict.html", input=payload, predicted=pred, probs=dict(zip(model.classes_, map(float, proba))))


@ui_bp.route("/evaluate", methods=["GET"])
def evaluate_page():
    # Load data and model, compute metrics
    df = load_data_from_mongo()
    target = "species"
    num_features = [
        "features.bill_length_mm",
        "features.bill_depth_mm",
        "features.flipper_length_mm",
        "features.body_mass_g"
    ]
    cat_features = ["island", "sex"]

    df = df.dropna(subset=[target])
    X = df[num_features + cat_features]
    y = df[target]

    # Load pipeline
    model = load_model()
    if model is None:
        return render_template("evaluate.html", error="Model not found. Train model first." )

    # If model is a pipeline, use it directly
    try:
        y_pred = model.predict(X)
    except Exception:
        # try transforming then predicting
        X_trans = model.named_steps["preprocess"].transform(X)
        y_pred = model.named_steps["rf"].predict(X_trans)

    acc = accuracy_score(y, y_pred)
    cm = confusion_matrix(y, y_pred)
    report = classification_report(y, y_pred, output_dict=True)

    # confusion matrix image
    buf = io.BytesIO()
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)

    # send page with embedded image as data URI
    import base64
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")

    return render_template("evaluate.html", accuracy=round(float(acc), 4), report=report, cm_image=img_b64)


def _build_and_evaluate_from_df(df):
    # Prepare data like train_species_model
    target = "species"
    num_features = [
        "features.bill_length_mm",
        "features.bill_depth_mm",
        "features.flipper_length_mm",
        "features.body_mass_g"
    ]
    cat_features = ["island", "sex"]

    df = df.dropna(subset=[target])
    if df.shape[0] < 10:
        return {"error": "Not enough data to evaluate."}

    X = df[num_features + cat_features]
    y = df[target]

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

    rf_model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)

    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("rf", rf_model)])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # confusion matrix image
    buf = io.BytesIO()
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)

    import base64
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")

    return {"accuracy": round(float(acc), 4), "report": report, "cm_image": img_b64}


@ui_bp.route("/evaluate/cassandra", methods=["GET"])
def evaluate_cassandra():
    host = os.getenv("CASSANDRA_HOST")
    if not host:
        return render_template("evaluate.html", error="CASSANDRA_HOST env not set")

    cass = CassandraDB()
    cass.connect()
    rows = cass.find({})
    cass.close()

    docs = []
    for r in rows:
        docs.append({
            "species": r.species,
            "island": r.island,
            "sex": r.sex,
            "features.bill_length_mm": r.bill_length_mm,
            "features.bill_depth_mm": r.bill_depth_mm,
            "features.flipper_length_mm": r.flipper_length_mm,
            "features.body_mass_g": r.body_mass_g,
        })

    df = pd.DataFrame(docs)

    res = _build_and_evaluate_from_df(df)
    if "error" in res:
        return render_template("evaluate.html", error=res["error"]) 

    return render_template("evaluate.html", accuracy=res["accuracy"], report=res["report"], cm_image=res["cm_image"])


@ui_bp.route("/evaluate/redis", methods=["GET"])
def evaluate_redis():
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    try:
        r = redis_lib.from_url(redis_url)
        keys = r.keys("penguin:*")
        docs = []
        for k in keys:
            raw = r.get(k)
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except Exception:
                continue
            docs.append(obj)

        if not docs:
            return render_template("evaluate.html", error="No penguin data found in Redis (keys penguin:*)")

        # Normalize to dataframe similar to Mongo shape
        rows = []
        for d in docs:
            rows.append({
                "species": d.get("species"),
                "island": d.get("island"),
                "sex": d.get("sex"),
                "features.bill_length_mm": d.get("features", {}).get("bill_length_mm"),
                "features.bill_depth_mm": d.get("features", {}).get("bill_depth_mm"),
                "features.flipper_length_mm": d.get("features", {}).get("flipper_length_mm"),
                "features.body_mass_g": d.get("features", {}).get("body_mass_g"),
            })

        df = pd.DataFrame(rows)
        res = _build_and_evaluate_from_df(df)
        if "error" in res:
            return render_template("evaluate.html", error=res["error"]) 

        return render_template("evaluate.html", accuracy=res["accuracy"], report=res["report"], cm_image=res["cm_image"])
    except Exception as e:
        return render_template("evaluate.html", error=f"Redis error: {e}")
