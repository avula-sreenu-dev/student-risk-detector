import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

df = pd.read_csv("D:\Sreenu\Projects\student-risk-detector\data\student_data.csv")

X = df[["attendance", "internal_marks", "assignment_score", "study_hours", "backlogs"]]
y = df["risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("Accuracy:", acc)
print(classification_report(y_test, y_pred))

joblib.dump(model, "model.pkl")
print("Model saved as model.pkl")