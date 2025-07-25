
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Streamlit app title
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

    # EDA
    st.subheader("Basic Info")
    st.write("Shape of dataset:", df.shape)
    st.write("Columns:", df.columns.tolist())
    st.write("Missing values per column:")
    st.write(df.isnull().sum())

    st.subheader("Descriptive Statistics")
    st.write(df.describe())

    st.subheader("Target Distribution")
    fig1, ax1 = plt.subplots()
    df['target'].value_counts().plot(kind='bar', ax=ax1)
    ax1.set_title('Distribution of Target Variable')
    ax1.set_xlabel('Heart Disease (1=Yes, 0=No)')
    ax1.set_ylabel('Count')
    st.pyplot(fig1)

    st.subheader("Age Distribution")
    fig2, ax2 = plt.subplots()
    sns.histplot(df['age'], bins=20, kde=True, ax=ax2)
    ax2.set_title('Age Distribution')
    st.pyplot(fig2)

    st.subheader("Chest Pain Type Distribution")
    fig3, ax3 = plt.subplots()
    sns.countplot(x='cp', data=df, ax=ax3)
    ax3.set_title('Chest Pain Type Distribution')
    ax3.set_xlabel('Chest Pain Type (0-3)')
    st.pyplot(fig3)

    # Optional: Pairplot (may be slow)
    if st.checkbox("Show pairplot (may be slow)", False):
        st.subheader("Pairplot of Selected Features")
        fig4 = sns.pairplot(df, vars=['age', 'trestbps', 'chol', 'thalach', 'oldpeak'], hue='target')
        st.pyplot(fig4)

    st.subheader("Boxplots for Outlier Detection")
    numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    fig5, axs = plt.subplots(2, 3, figsize=(15, 8))
    axs = axs.flatten()
    for i, col in enumerate(numeric_features):
        sns.boxplot(y=df[col], ax=axs[i])
        axs[i].set_title(f'{col}')
    plt.tight_layout()
    st.pyplot(fig5)

    # Split data and train model
    st.subheader("Model Training & Prediction")
    X = df.drop(columns='target')
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    st.write("Model Accuracy:", accuracy_score(y_test, y_pred))
    st.write("Confusion Matrix:")
    st.write(confusion_matrix(y_test, y_pred))
    st.write("Classification Report:")
    st.text(classification_report(y_test, y_pred))
