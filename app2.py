import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.tree import DecisionTreeClassifier

# Load and train the model
@st.cache_data
def train_model():
    df = pd.read_csv(r"C:\Environment\AI PROJECT\creditcard.csv")
    X = df.drop('Class', axis=1)
    y = df['Class']
    model = DecisionTreeClassifier(criterion='entropy', random_state=0)
    model.fit(X, y)
    return model, X.columns

# UI
st.title("Credit Card Fraud Detection")
st.write("Enter the transaction details to check if it's fraudulent.")

model, feature_names = train_model()

# Create input fields for all features
user_input = []
for feature in feature_names:
    val = st.number_input(f"{feature}", value=0.0, step=0.1)
    user_input.append(val)

if st.button("Predict"):
    input_df = pd.DataFrame([user_input], columns=feature_names)
    prediction = model.predict(input_df)
    result = "Fraud" if prediction[0] == 1 else "Not Fraud"
    st.success(f"Prediction: {result}")
