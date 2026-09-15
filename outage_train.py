import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ---------------- LOAD DATA ----------------
df = pd.read_csv("tower_hourly_MT_321_out.csv")

# ---------------- FEATURE ENGINEERING ----------------
df["timestamp"] = pd.to_datetime(df["timestamp"])

df["hour"] = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.dayofweek

# ---------------- INPUT / OUTPUT ----------------
X = df[["hour", "day_of_week"]]
y = df["outage"]

# ---------------- TRAIN TEST SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- MODEL ----------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ---------------- EVALUATION ----------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Outage Model Accuracy:", accuracy)

# ---------------- SAVE MODEL ----------------
joblib.dump(model, "outage_model.pkl")

print("✅ Model saved as outage_model.pkl")