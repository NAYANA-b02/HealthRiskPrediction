import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# === CONFIG ===
DATA_PATH = "data/Diabetes_1000_based_on_previous_rows.csv"  
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# === LOAD DATA ===
df = pd.read_csv(DATA_PATH)

print("Columns in dataset:", df.columns.tolist())
print("Data shape:", df.shape)

# === HANDLE MISSING VALUES & CLEAN ===
df = df.dropna(subset=['Outcome'])  # Ensure target column is not missing

# === FEATURE/TARGET SPLIT ===
X = df.drop(columns=['Outcome'])
y = df['Outcome']

# === HANDLE CATEGORICAL VALUES (if any) ===
# Convert gender or yes/no columns to numeric if needed
X = X.replace({'Male': 0, 'Female': 1, 'Yes': 1, 'No': 0, True: 1, False: 0})

# === IMPUTATION ===
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# === SCALING ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# === TRAIN/TEST SPLIT ===
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# === MODEL TRAINING ===
model = RandomForestClassifier(
    n_estimators=150,
    max_depth=10,
    random_state=42
)
model.fit(X_train, y_train)

# === EVALUATION ===
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Diabetes Model Accuracy: {acc:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# === SAVE ARTIFACTS ===
joblib.dump(model, os.path.join(MODEL_DIR, "diabetes_model.joblib"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "diabetes_scaler.joblib"))
joblib.dump(imputer, os.path.join(MODEL_DIR, "diabetes_imputer.joblib"))

print("\n✅ Saved: diabetes_model.joblib, diabetes_scaler.joblib, diabetes_imputer.joblib")
