#app.py
from flask import Flask, render_template, request
import pandas as pd
import joblib
from pathlib import Path

app = Flask(__name__)

MODEL_DIR = Path(__file__).resolve().parent

rf_path = MODEL_DIR / "random_forest_pipeline.pkl"
xgb_path = MODEL_DIR / "xgboost_pipeline.pkl"

#check files before loading
if not rf_path.exists():
    raise FileNotFoundError(f"Random Forest file not found: {rf_path}")

if not xgb_path.exists():
    raise FileNotFoundError(f"XGBoost file not found: {xgb_path}")

rf_model = joblib.load(rf_path)
xgb_model = joblib.load(xgb_path)


# Home page
@app.route("/")
def home():
    return render_template("about.html")

# About page
@app.route("/about")
def about():
    return render_template("about.html")

# About page
@app.route("/resume")
def resume():
    return render_template("resume.html")

# Prediction page
@app.route("/predict", methods=["GET", "POST"])
def predict():

    if request.method == "GET":
        return render_template("predict.html")

    try:

        # Collect user input
        patient_data = pd.DataFrame([{
            "gender": request.form["gender"],
            "age": float(request.form["age"]),
            "hypertension": int(request.form["hypertension"]),
            "heart_disease": int(request.form["heart_disease"]),
            "ever_married": request.form["ever_married"],
            "work_type": request.form["work_type"],
            "Residence_type": request.form["Residence_type"],
            "avg_glucose_level": float(
                request.form["avg_glucose_level"]
            ),
            "bmi": float(request.form["bmi"]),
            "smoking_status": request.form["smoking_status"]
        }])

        # Convert to DataFrame
        #input_df = pd.DataFrame([data])

        # Generate predictions
        rf_prediction = int(rf_model.predict(patient_data)[0])
        xgb_prediction = int(xgb_model.predict(patient_data)[0])

        # Calculate stroke probabilities as percentages
        rf_probs = rf_model.predict_proba(patient_data)[0]
        xgb_probs = xgb_model.predict_proba(patient_data)[0]

        rf_classes = list(rf_model.classes_)
        xgb_classes = list(xgb_model.classes_)

        rf_probability = rf_probs[rf_classes.index(1)] * 100
        xgb_probability = xgb_probs[xgb_classes.index(1)] * 100

         # Display results
        return render_template(
            "result.html",
            rf_prediction=rf_prediction,
            xgb_prediction=xgb_prediction,
            rf_probability=f"{rf_probability:.2f}",
            xgb_probability=f"{xgb_probability:.2f}",
            rf_no_stroke=f"{100 - rf_probability:.2f}",
            xgb_no_stroke=f"{100 - xgb_probability:.2f}"
        )

    except (ValueError, KeyError, TypeError) as error:
        return render_template(
            "predict.html",
            error=f"Please check your input values: {error}"
        ), 400


if __name__ == "__main__":
    app.run(debug=True)
