import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import os

os.makedirs("models", exist_ok=True)
results = {}

# DIABETES
print("Diabetes model ban raha hai...")
df = pd.read_csv("data/diabetes.csv")
X = df.drop("Outcome", axis=1)
y = df["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipe = Pipeline([("scaler", StandardScaler()), ("model", RandomForestClassifier(n_estimators=100, random_state=42))])
pipe.fit(X_train, y_train)
acc = accuracy_score(y_test, pipe.predict(X_test))
joblib.dump(pipe, "models/diabetes_model.pkl")
results["diabetes"] = {"accuracy": acc, "features": list(X.columns), "classes": ["No Diabetes", "Diabetes"]}
print(f"Diabetes Done! Accuracy: {acc*100:.2f}%")

# HEART
print("Heart model ban raha hai...")
df = pd.read_csv("data/heart.csv")
X = df.drop("target", axis=1)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipe = Pipeline([("scaler", StandardScaler()), ("model", RandomForestClassifier(n_estimators=100, random_state=42))])
pipe.fit(X_train, y_train)
acc = accuracy_score(y_test, pipe.predict(X_test))
joblib.dump(pipe, "models/heart_model.pkl")
results["heart"] = {"accuracy": acc, "features": list(X.columns), "classes": ["No Disease", "Heart Disease"]}
print(f"Heart Done! Accuracy: {acc*100:.2f}%")

# PARKINSONS
print("Parkinsons model ban raha hai...")
df = pd.read_csv("data/parkinsons.csv")
X = df.drop(["name", "status"], axis=1)
y = df["status"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipe = Pipeline([("scaler", StandardScaler()), ("model", RandomForestClassifier(n_estimators=100, random_state=42))])
pipe.fit(X_train, y_train)
acc = accuracy_score(y_test, pipe.predict(X_test))
joblib.dump(pipe, "models/parkinsons_model.pkl")
results["parkinsons"] = {"accuracy": acc, "features": list(X.columns), "classes": ["Healthy", "Parkinsons"]}
print(f"Parkinsons Done! Accuracy: {acc*100:.2f}%")

joblib.dump(results, "models/metadata.pkl")
print("\nSab models save ho gaye! Ab main.py chalao.")