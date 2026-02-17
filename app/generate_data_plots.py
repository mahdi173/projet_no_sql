
import matplotlib
matplotlib.use("Agg")
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from db.mongo_db import get_collection


def load_dataframe():
    collection = get_collection()
    data = list(collection.find({}, {
        "_id": 0,
        "species": 1,
        "sex": 1,
        "features.bill_length_mm": 1,
        "features.bill_depth_mm": 1,
        "features.flipper_length_mm": 1,
        "features.body_mass_g": 1
    }))
    return pd.json_normalize(data)


def plot_distributions(df):
    numeric_cols = [
        "features.bill_length_mm",
        "features.bill_depth_mm",
        "features.flipper_length_mm",
        "features.body_mass_g"
    ]

    for col in numeric_cols:
        plt.figure()
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f"Distribution of {col}")
        plt.savefig(f"plots/{col}_hist.png")
        plt.close()

        plt.figure()
        sns.boxplot(x=df[col].dropna())
        plt.title(f"Boxplot of {col}")
        plt.savefig(f"plots/{col}_box.png")
        plt.close()

        plt.figure()
        sns.kdeplot(df[col].dropna(), fill=True)
        plt.title(f"Density plot of {col}")
        plt.savefig(f"plots/{col}_density.png")
        plt.close()




def scatter_bill_by_species(df):
    plt.figure()
    sns.scatterplot(
        data=df,
        x="features.bill_length_mm",
        y="features.bill_depth_mm",
        hue="species"
    )
    plt.title("Bill length vs bill depth by species")
    plt.savefig("plots/bill_scatter.png")
    plt.close()



def scatter_flipper_mass_by_sex(df):
    plt.figure()
    sns.scatterplot(
        data=df,
        x="features.flipper_length_mm",
        y="features.body_mass_g",
        hue="sex"
    )
    plt.title("Flipper length vs body mass by sex")
    plt.savefig("plots/flipper_scatter.png")
    plt.close()



def correlation_matrix(df):
    cols = {
        "features.bill_length_mm": "bill_length_mm",
        "features.bill_depth_mm": "bill_depth_mm",
        "features.flipper_length_mm": "flipper_length_mm",
        "features.body_mass_g": "body_mass_g"
    }

    display_df = df[list(cols.keys())].rename(columns=cols)

    # Force numeric (important avec Mongo)
    display_df = display_df.apply(pd.to_numeric, errors="coerce")

    # Drop NA seulement maintenant
    display_df = display_df.dropna()

    corr = display_df.corr()

    plt.figure()
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Correlation matrix")
    
    plt.subplots_adjust(left=0.25, bottom=0.32)
    plt.savefig("plots/correlation_matrix.png", dpi=300)

    plt.close()


def main():
    df = load_dataframe()
    plot_distributions(df)

    scatter_bill_by_species(df) 

    scatter_flipper_mass_by_sex(df)

    correlation_matrix(df)


if __name__ == "__main__":
    main()