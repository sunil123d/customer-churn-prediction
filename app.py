# app.py
from flask import Flask, request, jsonify, render_template
import joblib
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# ── Load model artifacts ──────────────────────────────────────
model           = pickle.load(open("model.pkl", "rb"))
scaler          = pickle.load(open("scaler.pkl", "rb"))
feature_columns = joblib.load("models/feature_columns.pkl")

print(f"Model loaded: {type(model).__name__}")
print(f"Features expected: {len(feature_columns)}")


def prepare_input(data: dict) -> pd.DataFrame:
    """
    Converts form input into the exact 36-feature format
    the Random Forest model was trained on.
    All missing columns default to 0.
    """
    row = {col: 0 for col in feature_columns}

    # ── Numeric fields ────────────────────────────────────────
    row["Count"]           = 1
    row["Zip Code"]        = int(data.get("zip_code", 90000))
    row["Latitude"]        = float(data.get("latitude", 34.0))
    row["Longitude"]       = float(data.get("longitude", -118.0))
    row["Tenure Months"]   = int(data.get("tenure", 12))
    row["Monthly Charges"] = float(data.get("monthly_charges", 65.0))
    row["Total Charges"]   = float(data.get("tenure", 12)) * float(data.get("monthly_charges", 65.0))
    row["Churn Score"]     = int(data.get("churn_score", 50))
    row["CLTV"]            = int(data.get("cltv", 3000))

    # ── Binary one-hot fields ─────────────────────────────────
    def yes(field):
        return 1 if data.get(field, "No") == "Yes" else 0

    row["Gender_Male"]           = 1 if data.get("gender", "Male") == "Male" else 0
    row["Senior Citizen_Yes"]    = yes("senior_citizen")
    row["Partner_Yes"]           = yes("partner")
    row["Dependents_Yes"]        = yes("dependents")
    row["Phone Service_Yes"]     = yes("phone_service")
    row["Paperless Billing_Yes"] = yes("paperless_billing")

    # ── Multiple Lines ────────────────────────────────────────
    ml = data.get("multiple_lines", "No")
    row["Multiple Lines_No phone service"] = 1 if ml == "No phone service" else 0
    row["Multiple Lines_Yes"]              = 1 if ml == "Yes" else 0

    # ── Internet Service ──────────────────────────────────────
    inet = data.get("internet_service", "DSL")
    row["Internet Service_Fiber optic"] = 1 if inet == "Fiber optic" else 0
    row["Internet Service_No"]          = 1 if inet == "No" else 0

    # ── Internet add-ons (No internet service = 1 if No internet) ─
    no_inet = 1 if inet == "No" else 0
    for field, key in [
        ("online_security",   "Online Security"),
        ("online_backup",     "Online Backup"),
        ("device_protection", "Device Protection"),
        ("tech_support",      "Tech Support"),
        ("streaming_tv",      "Streaming TV"),
        ("streaming_movies",  "Streaming Movies"),
    ]:
        val = data.get(field, "No")
        row[f"{key}_No internet service"] = no_inet
        row[f"{key}_Yes"] = 1 if val == "Yes" and inet != "No" else 0

    # ── Contract ──────────────────────────────────────────────
    contract = data.get("contract", "Month-to-month")
    row["Contract_One year"] = 1 if contract == "One year" else 0
    row["Contract_Two year"] = 1 if contract == "Two year" else 0

    # ── Payment Method ────────────────────────────────────────
    pay = data.get("payment_method", "Electronic check")
    row["Payment Method_Credit card (automatic)"] = 1 if pay == "Credit card (automatic)" else 0
    row["Payment Method_Electronic check"]        = 1 if pay == "Electronic check" else 0
    row["Payment Method_Mailed check"]            = 1 if pay == "Mailed check" else 0

    df = pd.DataFrame([row])[feature_columns]
    return df


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.form.to_dict()

        X        = prepare_input(data)
        X_scaled = scaler.transform(X)

        prediction  = model.predict(X_scaled)[0]
        probability = model.predict_proba(X_scaled)[0][1]

        return jsonify({
            "churn":       bool(prediction),
            "probability": round(float(probability) * 100, 1),
            "risk_level": (
                "High Risk"   if probability > 0.7 else
                "Medium Risk" if probability > 0.4 else
                "Low Risk"
            )
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": type(model).__name__})


if __name__ == "__main__":
    app.run(debug=True, port=5000)