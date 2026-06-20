from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

MODEL_PATH = "model.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "model.pkl not found. Run train.py first or place model.pkl in the project root."
    )

model = joblib.load(MODEL_PATH)


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))


def get_prediction_label(risk_percentage):
    if risk_percentage < 35:
        return "Low Risk"
    elif risk_percentage < 65:
        return "Moderate Risk"
    return "High Risk"


def get_status_text(prediction):
    if prediction == "Low Risk":
        return "The student currently shows a stable academic pattern with a lower probability of near-term performance decline."
    elif prediction == "Moderate Risk":
        return "The student shows mixed academic signals and may slip further without timely corrective action."
    return "The student shows strong warning indicators that suggest urgent academic intervention is needed."


def get_suggestion_text(prediction):
    if prediction == "Low Risk":
        return [
            "Maintain your current study rhythm and avoid sudden drops in attendance.",
            "Keep internal marks and assignment quality steady through weekly revision.",
            "Use this stable phase to strengthen weak topics before exams approach."
        ]
    elif prediction == "Moderate Risk":
        return [
            "Increase focused study time and improve attendance over the next 7 days.",
            "Complete pending assignments first and revise the lowest-scoring subject daily.",
            "A small but consistent routine can move this profile back toward low risk."
        ]
    return [
        "Start an immediate recovery plan for attendance, marks, and backlog reduction.",
        "Break subjects into small targets and meet a mentor or faculty guide this week.",
        "Strong short-term discipline can still reduce risk significantly before exams."
    ]


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    error = None
    form_data = {
        "attendance": "",
        "internal_marks": "",
        "assignment_score": "",
        "study_hours": "",
        "backlogs": ""
    }

    if request.method == "POST":
        form_data["attendance"] = request.form.get("attendance", "").strip()
        form_data["internal_marks"] = request.form.get("internal_marks", "").strip()
        form_data["assignment_score"] = request.form.get("assignment_score", "").strip()
        form_data["study_hours"] = request.form.get("study_hours", "").strip()
        form_data["backlogs"] = request.form.get("backlogs", "").strip()

        attendance = safe_float(form_data["attendance"])
        internal_marks = safe_float(form_data["internal_marks"])
        assignment_score = safe_float(form_data["assignment_score"])
        study_hours = safe_float(form_data["study_hours"])
        backlogs = safe_float(form_data["backlogs"])

        attendance = clamp(attendance, 0, 100)
        internal_marks = clamp(internal_marks, 0, 30)
        assignment_score = clamp(assignment_score, 0, 10)
        study_hours = clamp(study_hours, 0, 24)
        backlogs = clamp(backlogs, 0, 20)

        if (
            form_data["attendance"] == ""
            or form_data["internal_marks"] == ""
            or form_data["assignment_score"] == ""
            or form_data["study_hours"] == ""
            or form_data["backlogs"] == ""
        ):
            error = "Please fill in all fields before predicting."
        else:
            try:
                features = np.array([[attendance, internal_marks, assignment_score, study_hours, backlogs]])
                raw_pred = model.predict(features)[0]

                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(features)[0]
                    if len(probabilities) == 2:
                        risk_percentage = round(float(probabilities[1]) * 100, 2)
                    else:
                        risk_percentage = round(float(np.max(probabilities)) * 100, 2)
                else:
                    risk_percentage = 75.0 if int(raw_pred) == 1 else 25.0

                prediction = get_prediction_label(risk_percentage)
                confidence = round(max(risk_percentage, 100 - risk_percentage), 2)
                status_text = get_status_text(prediction)
                suggestions = get_suggestion_text(prediction)

                score_breakdown = {
                    "attendance_impact": "Strong" if attendance < 65 else "Stable",
                    "marks_impact": "Needs Attention" if internal_marks < 15 else "Healthy",
                    "assignment_impact": "Weak" if assignment_score < 5 else "Good",
                    "study_impact": "Low" if study_hours < 2 else "Adequate",
                    "backlog_impact": "Critical" if backlogs >= 3 else "Controlled"
                }

                result = {
                    "prediction": prediction,
                    "risk_percentage": risk_percentage,
                    "confidence": confidence,
                    "status_text": status_text,
                    "suggestions": suggestions,
                    "attendance": attendance,
                    "internal_marks": internal_marks,
                    "assignment_score": assignment_score,
                    "study_hours": study_hours,
                    "backlogs": backlogs,
                    "score_breakdown": score_breakdown
                }

            except Exception as e:
                error = f"Prediction failed: {str(e)}"

    return render_template("index.html", result=result, error=error, form_data=form_data)


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)