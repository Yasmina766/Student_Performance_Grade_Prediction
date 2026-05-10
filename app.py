import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score


st.set_page_config(page_title="Student Performance Prediction", page_icon="🎓", layout="wide")
st.title("🎓 Student Performance Grade Prediction")
st.divider()
@st.cache_data
def load_and_train():
    data = pd.read_csv("Student_performance_10k.csv")
    data.drop(columns=["roll_no"], inplace=True)
    data.dropna(inplace=True)
    # cleaning
    data["gender"] = data["gender"].astype(str).str.lower().str.strip().replace({
        "boy": "male", "girl": "female"
    })
    data["race_ethnicity"] = data["race_ethnicity"].astype(str).str.lower().str.strip().replace({
        "a":"group a","b":"group b","c":"group c","d":"group d","e":"group e"
    })
    data["parental_level_of_education"] = data["parental_level_of_education"].astype(str).str.lower().str.strip()
    # numeric fix
    num_cols = ["math_score","reading_score","writing_score","science_score","total_score"]
    for col in num_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data.dropna(inplace=True)
    # outliers
    for col in ["math_score","reading_score","writing_score","science_score"]:
        q1, q3 = data[col].quantile(0.25), data[col].quantile(0.75)
        iqr = q3 - q1
        data = data[(data[col] >= q1 - 3*iqr) & (data[col] <= q3 + 3*iqr)]
    # encoding
    encoders = {}
    for col in ["gender","race_ethnicity","parental_level_of_education"]:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        encoders[col] = le

    target = LabelEncoder()
    data["grade"] = target.fit_transform(data["grade"])

    X = data.drop(["grade","total_score"], axis=1)
    y = data["grade"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=23, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=23),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=23),
        "KNN": KNeighborsClassifier(n_neighbors=5)
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        results[name] = {
            "model": model,
            "acc": accuracy_score(y_test, pred),
            "prec": precision_score(y_test, pred, average="weighted"),
            "rec": recall_score(y_test, pred, average="weighted"),
        }
    return results, target, encoders, scaler, X.columns
results, le_target, encoders, scaler, features = load_and_train()
best = max(results, key=lambda x: results[x]["acc"])
# ===== UI =====
st.subheader("📊 Model Performance")

cols = st.columns(3)
for i, (name, res) in enumerate(results.items()):
    with cols[i]:
        st.metric(
            name,
            f"{res['acc']*100:.1f}%",
            f"P:{res['prec']*100:.1f}% R:{res['rec']*100:.1f}%"
        )

st.divider()
st.subheader("🔍 Predict Grade")
with st.form("form"):
    gender = st.selectbox("Gender", ["male","female"])
    race = st.selectbox("Race", ["group a","group b","group c","group d","group e"])
    parental = st.selectbox("Parent Education", [
        "some high school","high school","some college",
        "associate's degree","bachelor's degree","master's degree"
    ])

    math = st.slider("Math",0,100,70)
    reading = st.slider("Reading",0,100,70)
    writing = st.slider("Writing",0,100,70)
    science = st.slider("Science",0,100,70)

    submit = st.form_submit_button("Predict 🎯")
if submit:
    x = pd.DataFrame([[
        encoders["gender"].transform([gender])[0],
        encoders["race_ethnicity"].transform([race])[0],
        encoders["parental_level_of_education"].transform([parental])[0],
        math, reading, writing, science
    ]], columns=features)
    x = scaler.transform(x)
    st.divider()
    st.subheader("🏆 Results")
    for name, res in results.items():
        pred = le_target.inverse_transform(res["model"].predict(x))[0]

        st.write(f"**{name}:** Grade → {pred}")

    best_pred = le_target.inverse_transform(results[best]["model"].predict(x))[0]

    st.success(f"⭐ Best Model ({best}): {best_pred}")