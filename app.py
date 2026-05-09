import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter('ignore')

st.set_page_config(page_title="Student Performance Prediction", page_icon="🎓", layout="wide")
st.title("🎓 Student Performance Grade Prediction")
st.caption("Supervised Classification · Decision Tree · Random Forest · KNN")
st.divider()


@st.cache_data
def train_models():
    data = pd.read_csv('Student_performance_10k.csv')

    # Clean & normalize messy categorical values
    data['gender'] = data['gender'].astype(str).str.strip().str.lower()
    data['gender'] = data['gender'].replace({'boy': 'male', 'girl': 'female'})

    data['race_ethnicity'] = data['race_ethnicity'].astype(str).str.strip().str.lower()
    data['race_ethnicity'] = data['race_ethnicity'].replace({
        'a': 'group a', 'b': 'group b', 'c': 'group c', 'd': 'group d', 'e': 'group e'
    })

    data['parental_level_of_education'] = data['parental_level_of_education'].astype(str).str.strip().str.lower()

    # Fix numeric columns
    for col in ['math_score', 'reading_score', 'writing_score', 'science_score', 'total_score']:
        data[col] = pd.to_numeric(data[col].astype(str).str.strip(), errors='coerce')

    data.drop(columns=['roll_no'], inplace=True)
    data.dropna(inplace=True)

    # Outlier removal
    for col in ['math_score', 'reading_score', 'writing_score', 'science_score']:
        Q1, Q3 = data[col].quantile(0.25), data[col].quantile(0.75)
        IQR = Q3 - Q1
        data = data[(data[col] >= Q1 - 3*IQR) & (data[col] <= Q3 + 3*IQR)]

    # Encode categoricals — save encoders for prediction
    encoders = {}
    for col in ['gender', 'race_ethnicity', 'parental_level_of_education']:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        encoders[col] = le

    le_target = LabelEncoder()
    data['grade'] = le_target.fit_transform(data['grade'])

    X = data.drop(columns=['grade', 'total_score'])
    y = data['grade']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23, stratify=y)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

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
            'cm':        confusion_matrix(y_test, y_pred),
        }

    return results, le_target, encoders, scaler, X.columns.tolist()


with st.spinner("Training models..."):
    results, le_target, encoders, scaler, feature_names = train_models()

best = max(results, key=lambda k: results[k]['accuracy'])

# ── Model Performance ──
st.subheader("📊 Model Performance")
cols = st.columns(3)
for i, (name, res) in enumerate(results.items()):
    with cols[i]:
        label = f"{name} ⭐" if name == best else name
        st.metric(label, f"{res['accuracy']*100:.1f}%",
                  f"P: {res['precision']*100:.1f}%  R: {res['recall']*100:.1f}%")

st.divider()

# ── Predict Section ──
st.subheader("🔍 Predict a Student's Grade")

with st.form("predict_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        gender   = st.selectbox("Gender",   ["male", "female"])
        race     = st.selectbox("Race / Ethnicity", ["group a", "group b", "group c", "group d", "group e"])
        parental = st.selectbox("Parental Education", [
            "some high school", "high school", "some college",
            "associate's degree", "bachelor's degree", "master's degree"
        ])
    with c2:
        lunch    = st.selectbox("Lunch Type", ["standard (1)", "free/reduced (0)"])
        test_prep = st.selectbox("Test Preparation Course", ["completed (1)", "none (0)"])
        math     = st.slider("Math Score",     0, 100, 70)
    with c3:
        reading  = st.slider("Reading Score",  0, 100, 70)
        writing  = st.slider("Writing Score",  0, 100, 70)
        science  = st.slider("Science Score",  0, 100, 70)

    submitted = st.form_submit_button("🎯 Predict Grade", use_container_width=True)

if submitted:
    lunch_val     = 1 if "1" in lunch     else 0
    test_prep_val = 1 if "1" in test_prep else 0

    gender_enc  = encoders['gender'].transform([gender])[0]
    race_enc    = encoders['race_ethnicity'].transform([race])[0]
    parental_enc = encoders['parental_level_of_education'].transform([parental])[0]

    input_row = pd.DataFrame([[
        gender_enc, race_enc, parental_enc,
        lunch_val, test_prep_val,
        math, reading, writing, science
    ]], columns=feature_names)

    input_scaled = scaler.transform(input_row)

    st.divider()
    st.subheader("🏆 Prediction Results")

    pred_cols = st.columns(3)
    colors_map = {'Decision Tree': '🟠', 'Random Forest': '🟣', 'KNN': '🔵'}

    predictions = {}
    for i, (name, res) in enumerate(results.items()):
        pred_label = le_target.inverse_transform(res['model'].predict(input_scaled))[0]
        predictions[name] = pred_label

        with pred_cols[i]:
            is_best = name == best
            st.markdown(f"### {colors_map[name]} {name}")
            st.markdown(f"## **Grade: {pred_label}**")
            st.caption(f"Model Accuracy: {res['accuracy']*100:.1f}%")
            if is_best:
                st.success("⭐ Most Accurate Model")

    # Best model recommendation
    st.divider()
    best_pred = predictions[best]
    st.info(f"**Recommended prediction (from {best}):** Grade **{best_pred}** — based on the highest accuracy model ({results[best]['accuracy']*100:.1f}%)")

st.divider()

# ── Charts ──
left, right = st.columns(2)

with left:
    st.subheader("Accuracy Comparison")
    fig, ax = plt.subplots(figsize=(6, 3.5))
    names = list(results.keys())
    accs  = [results[n]['accuracy'] for n in names]
    bars  = ax.bar(names, accs, color=['#f97316', '#7c6af7', '#22d3ee'], width=0.5)
    ax.bar_label(bars, fmt=lambda v: f'{v*100:.1f}%', padding=4, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.spines[:].set_visible(False)
    ax.yaxis.set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)

with right:
    st.subheader(f"Confusion Matrix - {best}")
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    sns.heatmap(results[best]['cm'], annot=True, fmt='d', cmap='Purples',
                xticklabels=le_target.classes_, yticklabels=le_target.classes_, ax=ax2)
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    plt.tight_layout()
    st.pyplot(fig2)

# ── Feature Importance ──
st.divider()
st.subheader("Feature Importances - Random Forest")
importances = pd.Series(results['Random Forest']['model'].feature_importances_, index=feature_names).sort_values()
fig3, ax3 = plt.subplots(figsize=(10, 3.5))
bars = ax3.barh(importances.index, importances.values, color='#7c6af7')
ax3.bar_label(bars, fmt='%.3f', padding=3, fontsize=8)
ax3.spines[:].set_visible(False)
ax3.xaxis.set_visible(False)
plt.tight_layout()
st.pyplot(fig3)

# ── Conclusion ──
st.divider()
st.subheader("Conclusion")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("**Decision Tree**")
    st.caption(f"Accuracy: {results['Decision Tree']['accuracy']*100:.1f}% — Fast and interpretable but prone to overfitting.")
with c2:
    st.markdown("**Random Forest**")
    st.caption(f"Accuracy: {results['Random Forest']['accuracy']*100:.1f}% — Best model. Combines 100 trees for robust predictions.")
with c3:
    st.markdown("**KNN**")
    st.caption(f"Accuracy: {results['KNN']['accuracy']*100:.1f}% — Simple and effective but sensitive to scaling.")
