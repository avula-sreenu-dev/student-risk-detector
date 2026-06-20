from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("model.pkl")


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    confidence = None
    risk_percentage = None

    if request.method == "POST":

        attendance = float(request.form["attendance"])
        internal_marks = float(request.form["internal_marks"])
        assignment_score = float(request.form["assignment_score"])
        study_hours = float(request.form["study_hours"])
        backlogs = float(request.form["backlogs"])

        features = np.array([[
            attendance,
            internal_marks,
            assignment_score,
            study_hours,
            backlogs
        ]])

        # Predict class
        pred = model.predict(features)[0]

        # Get probabilities
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features)[0]

            # Probability of High Risk class
            risk_percentage = round(probabilities[1] * 100, 2)

            # Confidence of prediction
            confidence = round(max(probabilities) * 100, 2)
        else:
            risk_percentage = 0
            confidence = 0

        # Risk level based on percentage
        if risk_percentage >= 70:
            prediction = "High Risk"
        elif risk_percentage >= 40:
            prediction = "Moderate Risk"
        else:
            prediction = "Low Risk"

    return render_template(
        "index.html",
        prediction=prediction,
        risk_percentage=risk_percentage,
        confidence=confidence
    )


if __name__ == "__main__":
    app.run(debug=True)