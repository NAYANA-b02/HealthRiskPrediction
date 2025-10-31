# train_hypertension.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
import joblib
import os

# === 1. Load Data ===
FILE_NAME = "data/Expanded_Hypertension_1000.csv"
df = pd.read_csv(FILE_NAME)

# === 2. Clean + Select Columns ===
# Remove unwanted columns
df = df.drop(columns=["Unnamed: 0", "Smoking", "Sporting"], errors='ignore')

# Features and target
X = df.drop(columns=["Hypertension_Tests"], errors='ignore')
y = df["Hypertension_Tests"]

# === 3. Handle Missing Values ===
imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(X)

# === 4. Scale Data ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# === 5. Split Data ===
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# === 6. Train Model ===
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# === 7. Evaluate ===
score = model.score(X_test, y_test)
print(f"✅ Hypertension Model trained successfully!")
print(f"Accuracy: {score:.2f}")

# === 8. Save Models ===
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/hypertension_model.joblib")
joblib.dump(scaler, "models/hypertension_scaler.joblib")
joblib.dump(imputer, "models/hypertension_imputer.joblib")

print("Models saved in /models folder ✅")
