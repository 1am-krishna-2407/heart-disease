import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix

# Load model and scaler
model = pickle.load(open("heart_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
feature_names = ["age", "sex", "cp", "trestbps", "chol", "fbs",
                 "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]

st.set_page_config(page_title="Heart Disease Risk Predictor", layout="centered")
st.title("❤️ Heart Disease Prediction Dashboard")
st.markdown("This app uses a trained ML model to predict heart disease risk based on patient data.")

# Sidebar Input
st.sidebar.header("Enter Patient Medical Data")
inputs = {}
for feature in feature_names:
    if feature in ["ca", "thal"]:
        val = st.sidebar.slider(f"{feature}", 0.0, 4.0, 1.0)
    elif feature == "oldpeak":
        val = st.sidebar.slider(f"{feature}", 0.0, 6.0, 1.0)
    elif feature in ["sex", "fbs", "exang", "restecg", "slope", "cp"]:
        val = st.sidebar.selectbox(f"{feature}", [0, 1, 2, 3])
    else:
        val = st.sidebar.slider(f"{feature}", float(0), float(200), float(100))
    inputs[feature] = val

input_df = pd.DataFrame([inputs])
input_scaled = scaler.transform(input_df)

# Prediction
if st.button("🔍 Predict"):
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.subheader("🔎 Prediction Result")
    if prediction == 1:
        st.error("🚨 High Risk: Likely Heart Disease")
    else:
        st.success("✅ Low Risk: Unlikely Heart Disease")

    # Input Summary
    st.subheader("📋 Patient Input Summary")
    st.dataframe(input_df.T, use_container_width=True)

    # Probability Chart
    st.subheader("📊 Prediction Probability")
    st.bar_chart(pd.DataFrame({'Heart Disease Risk': [probability], 'No Risk': [1 - probability]}).T)

    # Feature Importance Chart
    st.subheader("📌 Model Feature Importance")
    feat_importance = pd.read_csv("feature_importance.csv", index_col=0)
    st.bar_chart(feat_importance.sort_values(by=feat_importance.columns[0], ascending=True))


    # ROC & Confusion Matrix
    st.subheader("📈 Model Performance on Test Dataset")

    # Reload full dataset for demo evaluation
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    cols = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]
    df = pd.read_csv(url, names=cols, na_values='?')
    df.dropna(inplace=True)
    df = df.astype(float)
    df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)

    X = df.drop("target", axis=1)
    y = df["target"]
    X_scaled = scaler.transform(X)

    y_prob = model.predict_proba(X_scaled)[:, 1]
    y_pred = model.predict(X_scaled)

    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)
    fig_roc, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    st.pyplot(fig_roc)

    # Confusion Matrix
    cm = confusion_matrix(y, y_pred)
    fig_cm, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig_cm)
