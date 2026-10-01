"""
Loan Approval Prediction - Training Pipeline
---------------------------------------------
Loads the data, cleans it, builds a preprocessing + model pipeline,
compares models, tunes the best one, evaluates it, and saves the
final trained pipeline to disk.

Run from the project root:
    python src/train.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

DATA_PATH = "data/loan_prediction.csv"
MODEL_PATH = "models/loan_approval_model.pkl"
OUTPUTS_DIR = "outputs"

NUMERIC_FEATURES = [
    "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
    "Loan_Amount_Term", "Credit_History",
]
CATEGORICAL_FEATURES = [
    "Gender", "Married", "Dependents", "Education",
    "Self_Employed", "Property_Area",
]


def load_and_clean_data(path):
    df = pd.read_csv(path)
    df = df.drop(columns=["Loan_ID"])
    df["Dependents"] = df["Dependents"].replace("3+", "3")
    df["Loan_Status"] = df["Loan_Status"].map({"N": 0, "Y": 1})
    return df


def plot_eda(df):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    sns.countplot(data=df, x="Loan_Status", ax=axes[0])
    axes[0].set_title("Loan Status Balance")

    sns.barplot(data=df, x="Credit_History", y=(df["Loan_Status"] == 1).astype(int), ax=axes[1])
    axes[1].set_title("Approval Rate by Credit History")
    axes[1].set_ylabel("Approval Rate")

    sns.boxplot(data=df, x="Loan_Status", y="ApplicantIncome", ax=axes[2])
    axes[2].set_title("Applicant Income by Status")
    axes[2].set_ylim(0, 20000)

    plt.tight_layout()
    plt.savefig(f"{OUTPUTS_DIR}/loan_eda.png")
    plt.close()


def build_preprocessor():
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(transformers=[
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ])


def compare_models(preprocessor, X_train, y_train):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=42),
    }

    print("\n--- Cross-validation (5-fold) ---")
    for name, model in models.items():
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])
        scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="accuracy")
        print(f"{name}: mean={scores.mean():.4f}, std={scores.std():.4f}")


def tune_model(preprocessor, X_train, y_train):
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42)),
    ])

    param_grid = {
        "classifier__n_estimators": [100, 200, 300],
        "classifier__max_depth": [4, 6, 8, None],
        "classifier__min_samples_split": [2, 5, 10],
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    grid_search.fit(X_train, y_train)

    print("\nBest params:", grid_search.best_params_)
    print("Best CV score:", grid_search.best_score_)

    return grid_search.best_estimator_


def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)

    print("\n--- Final Test Results ---")
    print(classification_report(y_test, preds))

    cm = confusion_matrix(y_test, preds)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Rejected", "Approved"]).plot()
    plt.title("Confusion Matrix - Best Model")
    plt.savefig(f"{OUTPUTS_DIR}/loan_confusion_matrix.png")
    plt.close()


def plot_feature_importance(model):
    classifier = model.named_steps["classifier"]
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()

    importances = pd.Series(classifier.feature_importances_, index=feature_names)
    importances.sort_values(ascending=False).plot(kind="bar", figsize=(10, 5))
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig(f"{OUTPUTS_DIR}/loan_feature_importance.png")
    plt.close()


def main():
    df = load_and_clean_data(DATA_PATH)
    plot_eda(df)

    X = df.drop(columns=["Loan_Status"])
    y = df["Loan_Status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor()
    compare_models(preprocessor, X_train, y_train)

    best_model = tune_model(preprocessor, X_train, y_train)
    evaluate_model(best_model, X_test, y_test)
    plot_feature_importance(best_model)

    joblib.dump(best_model, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
