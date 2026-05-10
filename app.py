import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Student Performance Prediction", layout="wide")
st.title("🎓 Student Performance Prediction")

@st.cache_data
def load_data():
    df = pd.read_csv("Student_performance_10k.csv")

    df.drop(columns=["roll_no"], inplace=True)
    df.dropna(inplace=True)

    for col in ["math_score","reading_score","writing_score","science_score"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(inplace=True)
    return df


@st.cache_resource
def train_models(df):

    X = df.drop("grade", axis=1)
    y = df["grade"]

    cat_cols = ["gender","race_ethnicity","parental_level_of_education","lunch","test_preparation_course"]
    num_cols = ["math_score","reading_score","writing_score","science_score"]

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", StandardScaler(), num_cols)
    ])

    models = {
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "KNN": KNeighborsClassifier()
    }

    results = {}

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    for name, model in models.items():

        clf = Pipeline([
            ("prep", preprocessor),
            ("model", model)
        ])

        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)

        results[name] = {
            "model": clf,
            "acc": accuracy_score(y_test, pred),
            "prec": precision_score(y_test, pred, average="weighted"),
            "rec": recall_score(y_test, pred, average="weighted")
        }

    return results


df = load_data()
results = train_models(df)

best = max(results, key=lambda x: results[x]["acc"])

st.subheader("📊 Model Results")

for name, res in results.items():
    st.write(f"""
    **{name}**
    - Accuracy: {res['acc']:.2f}
    - Precision: {res['prec']:.2f}
    - Recall: {res['rec']:.2f}
    """)

st.success(f"🏆 Best Model: {best}")