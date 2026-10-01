"""
Loan Approval Prediction - Predict on a new applicant
--------------------------------------------------------
Loads the saved trained pipeline and predicts on a sample applicant.
Edit the `new_applicant` dictionary below to try your own values.

Run from the project root (after running train.py at least once):
    python src/predict.py
"""

import pandas as pd
import joblib

MODEL_PATH = "models/loan_approval_model.pkl"


def main():
    model = joblib.load(MODEL_PATH)

    new_applicant = pd.DataFrame([{
        "Gender": "Male",
        "Married": "Yes",
        "Dependents": "0",
        "Education": "Graduate",
        "Self_Employed": "No",
        "ApplicantIncome": 5000,
        "CoapplicantIncome": 1500,
        "LoanAmount": 120,
        "Loan_Amount_Term": 360,
        "Credit_History": 1,
        "Property_Area": "Urban",
    }])

    prediction = model.predict(new_applicant)[0]
    probability = model.predict_proba(new_applicant)[0]

    result = "Approved" if prediction == 1 else "Rejected"
    confidence = probability[prediction]

    print(f"Prediction: {result}")
    print(f"Confidence: {confidence:.2%}")


if __name__ == "__main__":
    main()
