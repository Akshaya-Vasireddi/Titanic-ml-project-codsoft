import streamlit as st
import numpy as np
import pickle

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered"
)

# ─────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────
@st.cache_resource
def load_model():
    with open('titanic_model.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────
st.title("🚢 Titanic Survival Prediction")
st.markdown("**By Akshaya Vasireddi** | Random Forest Model | Accuracy: 82.68%")
st.markdown("---")
st.write("Fill in the passenger details below to predict survival.")

# ─────────────────────────────────────────
# INPUT FORM
# ─────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("🎫 Ticket Class", [1, 2, 3],
                          format_func=lambda x: f"{x}{'st' if x==1 else 'nd' if x==2 else 'rd'} Class")
    sex = st.selectbox("👤 Gender", ["female", "male"])
    age = st.slider("🎂 Age", 1, 80, 25)
    sibsp = st.number_input("👫 Siblings / Spouses Aboard", 0, 8, 0)
    parch = st.number_input("👨‍👩‍👧 Parents / Children Aboard", 0, 6, 0)

with col2:
    fare = st.number_input("💷 Fare Paid (£)", 0.0, 520.0, 32.0)
    embarked = st.selectbox("⚓ Port of Embarkation",
                            ["S", "C", "Q"],
                            format_func=lambda x: {"S": "Southampton (S)",
                                                    "C": "Cherbourg (C)",
                                                    "Q": "Queenstown (Q)"}[x])
    title = st.selectbox("🎩 Title",
                         ["Mr", "Miss", "Mrs", "Master", "Officer", "Royalty"])
    has_cabin = st.selectbox("🛏️ Had a Cabin?", [0, 1],
                             format_func=lambda x: "Yes" if x == 1 else "No")

st.markdown("---")

# ─────────────────────────────────────────
# FEATURE ENGINEERING (matches training)
# ─────────────────────────────────────────
def engineer_features(pclass, sex, age, sibsp, parch,
                      fare, embarked, title, has_cabin):

    # Encodings
    sex_enc      = 1 if sex == 'male' else 0
    embarked_enc = {'S': 0, 'C': 1, 'Q': 2}.get(embarked, 0)
    title_enc    = {'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3,
                    'Officer': 4, 'Royalty': 5, 'Other': 6}.get(title, 0)

    # Engineered features
    family_size = sibsp + parch + 1
    is_alone    = 1 if family_size == 1 else 0

    # Age band
    if age <= 12:   age_band = 0
    elif age <= 18: age_band = 1
    elif age <= 35: age_band = 2
    elif age <= 60: age_band = 3
    else:           age_band = 4

    # Fare band
    if fare <= 7.9:    fare_band = 0
    elif fare <= 14.4: fare_band = 1
    elif fare <= 31.0: fare_band = 2
    else:              fare_band = 3

    return np.array([[pclass, sex_enc, age, sibsp, parch,
                      fare, embarked_enc, title_enc,
                      family_size, is_alone, has_cabin,
                      age_band, fare_band]])

# ─────────────────────────────────────────
# PREDICT BUTTON
# ─────────────────────────────────────────
if st.button("🔍 Predict Survival", use_container_width=True):

    features    = engineer_features(pclass, sex, age, sibsp, parch,
                                    fare, embarked, title, has_cabin)
    prediction  = model.predict(features)[0]
    probability = model.predict_proba(features)[0]

    st.markdown("---")
    st.subheader("📊 Prediction Result")

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Survival Probability",  f"{probability[1]*100:.1f}%")
    with col_b:
        st.metric("Non-Survival Probability", f"{probability[0]*100:.1f}%")

    if prediction == 1:
        st.success("✅ This passenger would have **SURVIVED**!")
        st.balloons()
    else:
        st.error("❌ This passenger would **NOT** have survived.")

    # Passenger summary
    st.markdown("**Passenger Summary:**")
    summary = {
        "Class": f"{pclass}{'st' if pclass==1 else 'nd' if pclass==2 else 'rd'}",
        "Gender": sex.capitalize(),
        "Age": age,
        "Title": title,
        "Fare": f"£{fare}",
        "Family Size": sibsp + parch + 1,
        "Travelling Alone": "Yes" if (sibsp + parch) == 0 else "No",
        "Has Cabin": "Yes" if has_cabin else "No"
    }
    st.table(summary)

# ─────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────
st.markdown("---")
st.caption("Built with ❤️ by Akshaya Vasireddi | Titanic ML Project | Streamlit + Scikit-learn")
