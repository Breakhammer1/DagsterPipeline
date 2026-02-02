from dagster import asset, Definitions
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor


@asset
def raw_data():
    df = pd.read_csv("/content/powerconsumption.csv")
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    return df


@asset
def eda_analysis(raw_data):
    df = raw_data.copy()

    plt.figure(figsize=(10,8))
    sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("correlation_heatmap.png")
    plt.close()

    return "EDA Completed"



@asset
def processed_data(raw_data):
    df = raw_data.copy()

    df["hour"] = df["Datetime"].dt.hour
    df["day"] = df["Datetime"].dt.day
    df["month"] = df["Datetime"].dt.month

    df = df.dropna()

    y = df["PowerConsumption_Zone1"]

    X = df.drop(columns=[
        "Datetime",
        "PowerConsumption_Zone1",
        "PowerConsumption_Zone2",
        "PowerConsumption_Zone3"
    ])

    return train_test_split(X, y, test_size=0.2, random_state=42)


@asset
def trained_models(processed_data):
    X_train, X_test, y_train, y_test = processed_data

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=10),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(),
        "Extra Trees": ExtraTreesRegressor(n_estimators=200, random_state=42),
    }

    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model

    return trained_models, X_test, y_test


@asset
def model_comparison(trained_models):
    models, X_test, y_test = trained_models

    results = {}

    print("\nMODEL PERFORMANCE")
    print("-" * 30)

    for name, model in models.items():
        preds = model.predict(X_test)
        score = r2_score(y_test, preds)
        results[name] = score
        print(f"{name}: {score:.4f}")

    best_model = max(results, key=results.get)

    print("\nBest Model:", best_model)

    return results, best_model



defs = Definitions(
    assets=[
        raw_data,
        eda_analysis,
        processed_data,
        trained_models,
        model_comparison
    ]
)
