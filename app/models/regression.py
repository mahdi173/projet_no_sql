
import matplotlib
matplotlib.use("Agg")
from db.mongo_db import get_collection
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
import os

PLOT_DIR = "app/plots"

def load_penguins():
    collection = get_collection()

    docs = list(collection.find({}))
    df = pd.DataFrame(docs)

    df = df.drop(columns=["_id"], errors="ignore")

    if "features" in df.columns:
        features_df = pd.json_normalize(df["features"])
        df = pd.concat([df.drop(columns=["features"]), features_df], axis=1)

    return df


def regression_simple(flipper_length_value=None):
    df = load_penguins().dropna()
    
    X = df["flipper_length_mm"]
    y = df["body_mass_g"]

    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    
    prediction = None
    if flipper_length_value is not None:
        prediction_input = pd.DataFrame(
            {"const": [1], "flipper_length_mm": [flipper_length_value]}
        )

        prediction = float(model.predict(prediction_input)[0])

    sns.regplot(
        x="flipper_length_mm",
        y="body_mass_g",
        data=df,
        line_kws={"color": "red"}
    )
    plt.title("Régression simple")
    plt.savefig(f"plots/reg_simple.png")
    plt.close()

    return model, prediction

def regression_multiple(flipper_length_value=None, bill_length_value=None,  bill_depth_value=None):
    df = load_penguins().dropna()

    X = df[["flipper_length_mm", "bill_length_mm", "bill_depth_mm"]]
    y = df["body_mass_g"]

    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()

    prediction = None
    # Prediction only if all values are provided
    if (
        flipper_length_value is not None and
        bill_length_value is not None and
        bill_depth_value is not None
    ):
        prediction_input = pd.DataFrame([{
            "const": 1,
            "flipper_length_mm": flipper_length_value,
            "bill_length_mm": bill_length_value,
            "bill_depth_mm": bill_depth_value
        }])

        prediction = float(model.predict(prediction_input)[0])

    # Plot with pairplot or correlation for multiple model
    sns.pairplot(df[["body_mass_g", "flipper_length_mm", "bill_length_mm", "bill_depth_mm"]])
    plt.savefig(f"plots/reg_multi.png")
    plt.close()

    return model, prediction