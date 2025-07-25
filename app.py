
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.set_page_config(page_title="Heart Disease Prediction", layout="wide")
st.title("Heart Disease Prediction App")

# Load dataset
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('heart.csv')
        return df
    except FileNotFoundError:
        st.error("heart.csv not found. Please ensure the file is in the root directory.")
        return None

df = load_data()

if df is not None:
    st.subheader("Dataset Preview")
    st.write(df.head())

    # Split data and train model
    X = df.drop(columns='target')
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Sidebar: user input
    st.sidebar.header("Patient Input Features")
    def user_input():
        age = st.sidebar.slider("Age", 29, 77, 54)
        sex = st.sidebar.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
        cp = st.sidebar.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
        trestbps = st.sidebar.slider("Resting Blood Pressure (mm Hg)", 90, 200, 130)
        chol = st.sidebar.slider("Cholesterol (mg/dl)", 100, 600, 250)
        fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
        restecg = st.sidebar.selectbox("Resting ECG Results", [0, 1, 2])
        thalach = st.sidebar.slider("Max Heart Rate Achieved", 70, 210, 150)
        exang = st.sidebar.selectbox("Exercise Induced Angina", [0, 1])
        oldpeak = st.sidebar.slider("ST Depression", 0.0, 6.0, 1.0, step=0.1)
        slope = st.sidebar.selectbox("Slope of Peak Exercise ST", [0, 1, 2])
        ca = st.sidebar.selectbox("Number of Major Vessels (0-4)", [0, 1, 2, 3, 4])
        thal = st.sidebar.selectbox("Thalassemia", [0, 1, 2, 3])

        data = {
            'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps,
            'chol': chol, 'fbs': fbs, 'restecg': restecg, 'thalach': thalach,
            'exang': exang, 'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
        }
        return pd.DataFrame(data, index=[0])

    input_df = user_input()

    st.subheader("User Input")
    st.write(input_df)

    # Prediction
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    st.subheader("Prediction Result")
    st.write("Prediction:", "Heart Disease" if prediction[0] == 1 else "No Heart Disease")
    st.write("Prediction Probability:", prediction_proba[0])

    # Evaluation
    st.subheader("Model Evaluation on Test Set")
    y_pred = model.predict(X_test)
    st.write("Accuracy Score:", accuracy_score(y_test, y_pred))
    st.write("Confusion Matrix:")
    st.write(confusion_matrix(y_test, y_pred))
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

