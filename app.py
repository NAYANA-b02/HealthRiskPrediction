import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
from pathlib import Path

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Health Risk Predictor", layout="wide")
st.title("🩺 Health Risk Predictor — Diabetes & Hypertension")

ROOT = Path(".")
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"

# =========================
# SAFE LOADERS
# =========================
def safe_joblib_load(p: Path):
    try:
        return joblib.load(p)
    except Exception as e:
        st.error(f"❌ Failed to load {p.name}: {e}")
        st.stop()

@st.cache_resource
def load_assets():
    assets = {
        "diabetes_model": safe_joblib_load(MODELS_DIR / "diabetes_model_compressed.joblib"),
        "diabetes_scaler": safe_joblib_load(MODELS_DIR / "diabetes_scaler.joblib"),
        "diabetes_imputer": safe_joblib_load(MODELS_DIR / "diabetes_imputer.joblib"),
        "hypertension_model": safe_joblib_load(MODELS_DIR / "hypertension_model.joblib"),
        "hypertension_scaler": safe_joblib_load(MODELS_DIR / "hypertension_scaler.joblib"),
        "hypertension_imputer": safe_joblib_load(MODELS_DIR / "hypertension_imputer.joblib"),
        "reference_df": pd.read_csv(DATA_DIR / "expanded_reference_dataset.csv"),
        "suggestions_df": pd.read_csv(DATA_DIR / "age_specific_suggestions_dataset.csv")
    }
    return assets

assets = load_assets()

# =========================
# CONSTANTS
# =========================
DIABETES_FEATURES = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
HYPERTENSION_FEATURES = ['Age', 'Gender', 'Medical_History', 'BMI', 'Systolic_BP', 'Diastolic_BP', 'Heart_Rate']

SYMPTOM_LIST = [
    'Fatigue', 'FrequentThirst', 'FrequentUrination', 'Headache', 'Dizziness',
    'BlurredVision', 'Chest Pain', 'Nausea', 'Weight Loss',
    'Shortness of Breathing', 'Swelling in Feet', 'Tingling in Hands',
    'Sleep Disturbance', 'Rapid Heartbeat', 'None'
]

# =========================
# HELPERS
# =========================
def parse_range_string(s):
    if pd.isna(s): return None, None
    s = str(s).strip().replace("–", "-")
    if "-" in s:
        parts = re.findall(r"-?\d+\.?\d*", s)
        if len(parts) >= 2:
            return float(parts[0]), float(parts[1])
    if s.startswith("<"):
        try: return None, float(s[1:].strip())
        except: return None, None
    if s.startswith(">="):
        try: return float(s[2:].strip()), None
        except: return None, None
    return None, None

def get_reference_row(param, age_group):
    df = assets["reference_df"]
    mask = (df["Parameter"].astype(str).str.lower() == param.lower()) & (df["AgeGroup"] == age_group)
    return df[mask].iloc[0] if not df[mask].empty else None

def evaluate_metric(parameter, age_group, value):
    """Return risk level + reference range."""
    row = get_reference_row(parameter, age_group)
    if row is None:
        return "Unknown", "N/A"
    try:
        hypo_low, hypo_high = parse_range_string(row.get("HypoRange"))
        normal_low, normal_high = parse_range_string(row.get("NormalRange"))
        pre_low, pre_high = parse_range_string(row.get("PreRiskRange"))
        risk_low, _ = parse_range_string(row.get("RiskRange"))
        v = float(value)

        if hypo_high and v < hypo_high:
            return "Low", row.get("NormalRange", "")
        if normal_low and normal_high and normal_low <= v <= normal_high:
            return "Normal", row.get("NormalRange", "")
        if pre_low and pre_high and pre_low <= v <= pre_high:
            return "Moderate", row.get("PreRiskRange", "")
        if risk_low and v >= risk_low:
            return "High", row.get("RiskRange", "")
        return "Normal", row.get("NormalRange", "")
    except Exception:
        return "Unknown", "N/A"

def fetch_suggestions(condition, symptoms, age_group, risk_level):
    """Return category -> [suggestions] (deduped)."""
    df = assets["suggestions_df"].copy()
    df["Condition"] = df["Condition"].astype(str).str.lower()
    df["Symptom"] = df["Symptom"].astype(str).str.lower()
    df["AgeGroup"] = df["AgeGroup"].astype(str)
    df["RiskLevel"] = df["RiskLevel"].astype(str).str.lower()

    condition = condition.lower()
    risk = risk_level.lower()
    collected = []

    # Limit to top 2 symptoms to avoid repetition
    for s in symptoms[:2]:
        s = s.lower()
        matches = df[
            (df["Condition"] == condition) &
            (df["Symptom"] == s) &
            (df["AgeGroup"] == age_group) &
            (df["RiskLevel"] == risk)
        ]
        if matches.empty:
            matches = df[
                (df["Condition"] == condition) &
                (df["Symptom"] == s) &
                (df["AgeGroup"] == age_group)
            ]
        for _, r in matches.iterrows():
            cat = r.get("Category", "General")
            parts = [p.strip() for p in str(r["Suggestion"]).split('.') if len(p.strip()) > 2]
            for p in parts:
                collected.append((cat, p))

    # Remove duplicates across all categories
    grouped, seen = {}, set()
    for cat, txt in collected:
        if txt.lower() not in seen:
            seen.add(txt.lower())
            grouped.setdefault(cat, []).append(txt)
    return grouped

def age_group_of(a):
    if a < 18: return "0-17"
    if a < 40: return "18-39"
    if a < 60: return "40-59"
    if a < 75: return "60-74"
    return "75+"

# =========================
# SIDEBAR INPUTS
# =========================
with st.sidebar:
    st.header("Patient Details")
    age = st.slider("Age", 1, 100, 35)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    med_hist = st.selectbox("Known Medical History?", [0, 1], format_func=lambda x: "Yes" if x else "No")
    on_medication = st.checkbox("Currently on Medication (may affect results)", value=False)

st.subheader("Vitals")
c1, c2, c3 = st.columns(3)
with c1:
    bp_sys = st.number_input("Systolic BP (mmHg)", 50.0, 250.0, 120.0)
    bp_dia = st.number_input("Diastolic BP (mmHg)", 30.0, 150.0, 80.0)
with c2:
    sugar_f = st.number_input("Fasting Sugar (mg/dL)", 30.0, 500.0, 95.0)
    sugar_r = st.number_input("Random Sugar (mg/dL)", 30.0, 800.0, 140.0)
with c3:
    bmi = st.number_input("BMI (optional)", 10.0, 80.0, 25.0)
    hr = st.number_input("Heart Rate (optional)", 30, 200, 75)

st.subheader("Symptoms")
none = st.checkbox("None of the following symptoms", value=False)
if none:
    selected_symptoms = ["None"]
else:
    selected_symptoms = st.multiselect("Select symptoms", SYMPTOM_LIST)
if "None" in selected_symptoms and len(selected_symptoms) > 1:
    selected_symptoms = ["None"]

# =========================
# RUN PREDICTION
# =========================
if st.button("Run Prediction"):
    age_group = age_group_of(age)
    gender_num = 0 if gender == "Male" else 1
    avg_bp = (bp_sys + bp_dia) / 2

    diab_df = pd.DataFrame([[sugar_f, avg_bp, 0, 0, bmi, 0.47, age]], columns=DIABETES_FEATURES)
    hyper_df = pd.DataFrame([[age, gender_num, med_hist, bmi, bp_sys, bp_dia, hr]], columns=HYPERTENSION_FEATURES)

    diab_scaled = assets["diabetes_scaler"].transform(assets["diabetes_imputer"].transform(diab_df))
    hyper_scaled = assets["hypertension_scaler"].transform(assets["hypertension_imputer"].transform(hyper_df))

    diab_prob = assets["diabetes_model"].predict_proba(diab_scaled)[0][1]
    hyper_prob = assets["hypertension_model"].predict_proba(hyper_scaled)[0][1]

    def prob_to_risk(p):
        if p >= 0.7: return "High"
        if p >= 0.4: return "Moderate"
        return "Low"

    diab_model_risk = prob_to_risk(diab_prob)
    hyper_model_risk = prob_to_risk(hyper_prob)

    # Metric evaluation
    bp_sys_r, bp_sys_ref = evaluate_metric("BP_Systolic", age_group, bp_sys)
    bp_dia_r, bp_dia_ref = evaluate_metric("BP_Diastolic", age_group, bp_dia)
    sugar_f_r, sugar_f_ref = evaluate_metric("Sugar_Fasting", age_group, sugar_f)
    sugar_r_r, sugar_r_ref = evaluate_metric("Sugar_Random", age_group, sugar_r)

    # Hypo detection
    diab_cond = "Hypoglycemia" if sugar_f_r == "Low" or sugar_r_r == "Low" else "Diabetes"
    hyper_cond = "Hypotension" if bp_sys_r == "Low" or bp_dia_r == "Low" else "Hypertension"

    # Risk fusion
    diab_final = "High" if sugar_f_r == "High" or sugar_r_r == "High" else ("Low" if sugar_f_r == "Low" or sugar_r_r == "Low" else diab_model_risk)
    hyper_final = "High" if bp_sys_r == "High" or bp_dia_r == "High" else ("Low" if bp_sys_r == "Low" or bp_dia_r == "Low" else hyper_model_risk)

    # Overall condition
    if diab_final == "High" and hyper_final == "High":
        overall_cond, overall_risk = "Mixed", "High"
    elif diab_final == "High":
        overall_cond, overall_risk = diab_cond, "High"
    elif hyper_final == "High":
        overall_cond, overall_risk = hyper_cond, "High"
    elif diab_final == "Moderate" or hyper_final == "Moderate":
        overall_cond, overall_risk = "Combined", "Moderate"
    else:
        overall_cond, overall_risk = "Healthy", "Low"

    # =========================
    # DISPLAY RESULTS
    # =========================
    color_map = {"High": "🔴", "Moderate": "🟡", "Low": "🟢", "Unknown": "⚪"}
    st.markdown(f"## {color_map.get(overall_risk)} Overall Risk: **{overall_risk}** — Condition: **{overall_cond}**")

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Diabetes Probability", f"{diab_prob:.2f}", diab_final)
    with c2:
        st.metric("Hypertension Probability", f"{hyper_prob:.2f}", hyper_final)

    # Parameter table
    st.markdown("### Parameter Overview")
    df = pd.DataFrame([
        ("Systolic BP", bp_sys, bp_sys_ref, bp_sys_r),
        ("Diastolic BP", bp_dia, bp_dia_ref, bp_dia_r),
        ("Fasting Sugar", sugar_f, sugar_f_ref, sugar_f_r),
        ("Random Sugar", sugar_r, sugar_r_ref, sugar_r_r),
        ("BMI", bmi, "—", "—"),
        ("Heart Rate", hr, "—", "—")
    ], columns=["Parameter", "Value", "ReferenceRange", "Risk"])

    icon = {"Low": "🔵", "Normal": "🟢", "Moderate": "🟡", "High": "🔴", "Unknown": "⚪", "—": "⚪"}

    def icon_with_condition(param, risk):
        if risk == "Low":
            if "BP" in param:
                return "🔵 (Hypotension)"
            if "Sugar" in param:
                return "🔵 (Hypoglycemia)"
        return icon.get(risk, "⚪")

    df["Icon"] = df.apply(lambda x: icon_with_condition(x["Parameter"], x["Risk"]), axis=1)
    st.dataframe(df[["Parameter", "Value", "Icon", "Risk", "ReferenceRange"]], use_container_width=True)

    # =========================
    # SUGGESTIONS
    # =========================
    st.markdown("---")
    st.markdown("### 💡 Personalized Suggestions")

    conds = ["Diabetes", "Hypertension"] if overall_cond == "Mixed" else [overall_cond]
    aggregated, seen = {}, set()
    for c in conds:
        rlevel = diab_final if "diab" in c.lower() or "hypogly" in c.lower() else hyper_final
        group = fetch_suggestions(c, selected_symptoms or ["None"], age_group, rlevel)
        for cat, items in group.items():
            for it in items:
                if it.lower() not in seen:
                    aggregated.setdefault(cat, []).append(it)
                    seen.add(it.lower())

    if not aggregated:
        st.success("✅ Your profile indicates Low Risk. Maintain healthy habits!")
    else:
        order = ["Lifestyle Tip", "Self-Monitoring", "Medical Advice", "General"]
        for cat in order + [c for c in aggregated if c not in order]:
            if cat in aggregated:
                st.markdown(f"#### {cat}")
                for t in aggregated[cat]:
                    st.markdown(f"- {t}")

    # Print/Save
    st.download_button(
        "🖨️ Print or Save Report",
        df.to_csv(index=False).encode("utf-8"),
        file_name="health_risk_report.csv",
        mime="text/csv"
    )

    st.caption("⚠️ Educational use only. Consult a certified healthcare provider for medical advice.")
