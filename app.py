import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import chi2
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter('ignore')

# ── Page Config ──
st.set_page_config(page_title="Student Dropout Prediction", page_icon="🎓", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Sora', sans-serif; }

.main { background: #0f1117; }

.metric-card {
    background: #1a1d2e;
    border: 1px solid #2a2d3e;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
}
.metric-value { font-size: 2rem; font-weight: 700; color: #7c6af7; }
.metric-label { font-size: 0.85rem; color: #888; margin-top: 4px; }

.best-badge {
    display: inline-block;
    background: linear-gradient(135deg, #7c6af7, #a78bfa);
    color: white;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-left: 8px;
}
.section-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #c9c9e3;
    margin-bottom: 12px;
    border-left: 3px solid #7c6af7;
    padding-left: 10px;
}
</style>
""", unsafe_allow_html=True)


# ── Train Models (cached) ──
@st.cache_data
def train_models(path='dataset.csv'):
    data = pd.read_csv(path)

    # Outlier removal
    for col in ['Curricular units 1st sem (grade)', 'Curricular units 2nd sem (grade)']:
        Q1, Q3 = data[col].quantile(0.25), data[col].quantile(0.75)
        IQR = Q3 - Q1
        data = data[(data[col] >= Q1 - 3*IQR) & (data[col] <= Q3 + 3*IQR)]

    # Encode target
    le = LabelEncoder()
    data['Target'] = le.fit_transform(data['Target'])

    # Feature selection
    cat_cols = [c for c in data.columns if data[c].dtype == 'int64' and c != 'Target']
    p_vals = pd.Series(chi2(data[cat_cols], data['Target'])[1], index=cat_cols)
    data.drop(columns=p_vals[p_vals >= 0.05].index.tolist(), inplace=True)

    X = data.drop('Target', axis=1)
    y = data['Target']

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
            'y_pred':    y_pred,
            'accuracy':  accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall':    recall_score(y_test, y_pred, average='weighted'),
            'cm':        confusion_matrix(y_test, y_pred),
        }

    return results, le.classes_, y_test, X.columns.tolist()


# ── Load ──
with st.spinner("Training models..."):
    results, classes, y_test, feature_names = train_models()

best_model = max(results, key=lambda k: results[k]['accuracy'])

# ── Header ──
st.markdown("## 🎓 Student Dropout Prediction")
st.markdown("**Supervised Classification** · Decision Tree · Random Forest · KNN")
st.divider()

# ── Metrics Row ──
cols = st.columns(3)
colors = {'Decision Tree': '#f97316', 'Random Forest': '#7c6af7', 'KNN': '#22d3ee'}

for i, (name, res) in enumerate(results.items()):
    with cols[i]:
        badge = '<span class="best-badge">⭐ Best</span>' if name == best_model else ''
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size:0.9rem; font-weight:600; color:#c9c9e3">{name} {badge}</div>
            <div class="metric-value" style="color:{colors[name]}">{res['accuracy']*100:.1f}%</div>
            <div class="metric-label">Accuracy</div>
            <div style="margin-top:12px; display:flex; justify-content:space-around;">
                <div><div style="font-size:1rem;font-weight:600;color:#aaa">{res['precision']*100:.1f}%</div><div class="metric-label">Precision</div></div>
                <div><div style="font-size:1rem;font-weight:600;color:#aaa">{res['recall']*100:.1f}%</div><div class="metric-label">Recall</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.divider()

# ── Charts Row ──
left, right = st.columns([1, 1])

with left:
    st.markdown('<div class="section-title">Accuracy Comparison</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    fig.patch.set_facecolor('#1a1d2e')
    ax.set_facecolor('#1a1d2e')

    names = list(results.keys())
    accs  = [results[n]['accuracy'] for n in names]
    bar_colors = [colors[n] for n in names]

    bars = ax.bar(names, accs, color=bar_colors, width=0.5, edgecolor='none')
    for bar, val in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val*100:.1f}%', ha='center', color='white', fontsize=10, fontweight='bold')

    ax.set_ylim(0, 1)
    ax.tick_params(colors='#888')
    ax.spines[:].set_visible(False)
    ax.yaxis.set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)

with right:
    st.markdown('<div class="section-title">Confusion Matrix – Best Model (Random Forest)</div>', unsafe_allow_html=True)
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    fig2.patch.set_facecolor('#1a1d2e')
    ax2.set_facecolor('#1a1d2e')

    cm = results[best_model]['cm']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
                xticklabels=classes, yticklabels=classes, ax=ax2,
                linewidths=0.5, linecolor='#0f1117')

    ax2.set_xlabel('Predicted', color='#888')
    ax2.set_ylabel('Actual', color='#888')
    ax2.tick_params(colors='#aaa')
    plt.tight_layout()
    st.pyplot(fig2)

# ── Feature Importance ──
st.divider()
st.markdown('<div class="section-title">Top 10 Feature Importances (Random Forest)</div>', unsafe_allow_html=True)

rf_model = results['Random Forest']['model']
importances = pd.Series(rf_model.feature_importances_, index=feature_names).nlargest(10).sort_values()

fig3, ax3 = plt.subplots(figsize=(10, 3.5))
fig3.patch.set_facecolor('#1a1d2e')
ax3.set_facecolor('#1a1d2e')

bars = ax3.barh(importances.index, importances.values, color='#7c6af7', edgecolor='none')
ax3.tick_params(colors='#aaa', labelsize=9)
ax3.spines[:].set_visible(False)
ax3.xaxis.set_visible(False)

for bar, val in zip(bars, importances.values):
    ax3.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
             f'{val:.3f}', va='center', color='#aaa', fontsize=8)

plt.tight_layout()
st.pyplot(fig3)

# ── Conclusion ──
st.divider()
st.markdown('<div class="section-title">Conclusion</div>', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("**🌳 Decision Tree**")
    st.caption(f"Accuracy: {results['Decision Tree']['accuracy']*100:.1f}% — Fast and interpretable but prone to overfitting.")
with c2:
    st.markdown("**🌲 Random Forest ⭐**")
    st.caption(f"Accuracy: {results['Random Forest']['accuracy']*100:.1f}% — Best model. Combines 100 trees for robust predictions.")
with c3:
    st.markdown("**📍 KNN**")
    st.caption(f"Accuracy: {results['KNN']['accuracy']*100:.1f}% — Simple and effective but sensitive to scaling.")
