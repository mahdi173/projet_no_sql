import pandas as pd
import numpy as np
import pickle
from pymongo import MongoClient

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.inspection import permutation_importance

from db.mongo_db import get_collection

# --------------------------------------------------------
# 1. Load data from MongoDB
# --------------------------------------------------------

def load_data_from_mongo():
    collection = get_collection()

    data = list(collection.find({}))
    df = pd.json_normalize(data)

    return df


# --------------------------------------------------------
# 2. Train Model
# --------------------------------------------------------

def train_and_save_model(model_path="species_model.pkl"):

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

    rf_model = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("rf", rf_model)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", round(accuracy, 4))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=3))

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    print("\nCross-validation mean:", cv_scores.mean())
    print("Cross-validation std:", cv_scores.std())


    X_test_transformed = pipeline.named_steps["preprocess"].transform(X_test)

    actual_feature_names = (
        pipeline.named_steps["preprocess"]
        .get_feature_names_out()
    )

    perm = permutation_importance(
        pipeline,
        X_test,
        y_test,
        n_repeats=10,
        random_state=42,
        n_jobs=-1
    )

    values = perm.importances_mean
    names = actual_feature_names

    print("DEBUG:", len(values), "importance values")
    print("DEBUG:", len(names), "feature names")

    names = names[:len(values)]

    importances = pd.Series(values, index=names).sort_values(ascending=False)

    print("\nPermutation Importances:\n", importances)

    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)

    print(f"\nModel saved to: {model_path}")


if __name__ == "__main__":
    train_and_save_model()