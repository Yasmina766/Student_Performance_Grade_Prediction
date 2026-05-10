import warnings
warnings.simplefilter('ignore')

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Grade Prediction",
    page_icon="🎓",
    layout="wide"
)
st.title("🎓 Student Performance Grade Prediction")
st.caption("Supervised Classification · Decision Tree · Random Forest · KNN")
st.divider()


# ── Data loading & model training ──────────────────────────────────────────────
@st.cache_data
def train_models():
    data = pd.read_csv('Student_performance_10k.csv')

    # Fix numeric columns
    for col in ['math_score', 'reading_score', 'writing_score', 'science_score', 'total_score']:
        data[col] = pd.to_numeric(data[col].astype(str).str.strip(), errors='coerce')

    # Clean gender — handle 'Boy', 'Girl', literal '\t' prefix
    data['gender'] = (
        data['gender']
        .astype(str).str.strip()
        .str.replace(r'\\t', '', regex=True)
        .str.replace(r'\\n', '', regex=True)
        .str.lower()
        .replace({'boy': 'male', 'girl': 'female', 'nan': None})
    )

    # Clean race_ethnicity — unify 'group A/B/C/D/E' and 'A/B/C/D/E' -> 'A'-'E'
    data['race_ethnicity'] = (
        data['race_ethnicity']
        .astype(str).str.strip()
        .str.replace(r'\\n', '', regex=True)
        .str.replace(r'\\t', '', regex=True)
        .str.lower()
        .str.replace(r'^group\s*', '', regex=True)
        .str.upper()
        .replace({'NAN': None})
    )

    data['parental_level_of_education'] = (
        data['parental_level_of_education'].astype(str).str.strip().str.lower()
    )

    # Drop rows with missing values, then drop unused column
    data.dropna(inplace=True)
    data.drop(columns=['roll_no'], inplace=True)

    # Outlier removal using IQR x3
    for col in ['math_score', 'reading_score', 'writing_score', 'science_score']:
        Q1, Q3 = data[col].quantile(0.25), data[col].quantile(0.75)
        IQR = Q3 - Q1
        data = data[(data[col] >= Q1 - 3*IQR) & (data[col] <= Q3 + 3*IQR)]

    # Encode categorical features
    encoders = {}
    for col in ['gender', 'race_ethnicity', 'parental_level_of_education']:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        encoders[col] = le

    # Encode target
    le_target = LabelEncoder()
    data['grade'] = le_target.fit_transform(data['grade'])

    X = data.drop(columns=['grade', 'total_score'])
    y = data['grade']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=23, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # Train models and store results
    models = {
        'Decision Tree': DecisionTreeClassifier(random_state=23),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=23),
        'KNN':           KNeighborsClassifier(n_neighbors=5),
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = {
            'model':     model,
            'accuracy':  accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall':    recall_score(y_test, y_pred, average='weighted'),
        }

    return results, le_target, encoders, scaler, X.columns.tolist()


# ── Train ───────────────────────────────────────────────────────────────────────
with st.spinner("Training models, please wait..."):
    results, le_target, encoders, scaler, feature_names = train_models()

best = max(results, key=lambda k: results[k]['accuracy'])


# ── Model performance ───────────────────────────────────────────────────────────
st.subheader("📊 Model Performance")
cols = st.columns(3)
for i, (name, res) in enumerate(results.items()):
    with cols[i]:
        label = f"{name} ⭐" if name == best else name
        st.metric(
            label,
            f"{res['accuracy']*100:.1f}%",
            f"Precision: {res['precision']*100:.1f}%  |  Recall: {res['recall']*100:.1f}%"
        )

st.divider()


# ── Prediction form ─────────────────────────────────────────────────────────────
st.subheader("🔍 Predict a Student's Grade")

with st.form("predict_form"):
    c1, c2, c3 = st.columns(3)

    with c1:
        gender   = st.selectbox("Gender",           ["male", "female"])
        race     = st.selectbox("Race / Ethnicity", ["A", "B", "C", "D", "E"])
        parental = st.selectbox("Parental Education", [
            "some high school", "high school", "some college",
            "associate's degree", "bachelor's degree", "master's degree"
        ])

    with c2:
        lunch     = st.selectbox("Lunch Type",              ["Standard", "Free / Reduced"])
        test_prep = st.selectbox("Test Preparation Course", ["Completed", "None"])
        math      = st.slider("Math Score",    0, 100, 70)

    with c3:
        reading = st.slider("Reading Score", 0, 100, 70)
        writing = st.slider("Writing Score", 0, 100, 70)
        science = st.slider("Science Score", 0, 100, 70)

    submitted = st.form_submit_button("🎯 Predict Grade", use_container_width=True)


# ── Prediction output ───────────────────────────────────────────────────────────
if submitted:
    lunch_val     = 1 if lunch     == "Standard"  else 0
    test_prep_val = 1 if test_prep == "Completed" else 0

    input_row = pd.DataFrame([[
        encoders['gender'].transform([gender])[0],
        encoders['race_ethnicity'].transform([race])[0],
        encoders['parental_level_of_education'].transform([parental])[0],
        lunch_val, test_prep_val, math, reading, writing, science
    ]], columns=feature_names)

    input_scaled = scaler.transform(input_row)

    st.divider()
    st.subheader("🏆 Prediction Results")

    icons = {'Decision Tree': '🟠', 'Random Forest': '🟣', 'KNN': '🔵'}
    pred_cols = st.columns(3)

    for i, (name, res) in enumerate(results.items()):
        grade = le_target.inverse_transform(res['model'].predict(input_scaled))[0]
        with pred_cols[i]:
            st.markdown(f"### {icons[name]} {name}")
            st.markdown(f"## **Grade: {grade}**")
            st.caption(f"Model Accuracy: {res['accuracy']*100:.1f}%")
            if name == best:
                st.success("⭐ Best Model")

    st.divider()
    best_grade = le_target.inverse_transform(results[best]['model'].predict(input_scaled))[0]
    st.info(
        f"**Best model ({best}) predicts:** Grade **{best_grade}**"
        f" — Accuracy: {results[best]['accuracy']*100:.1f}%"
    )
