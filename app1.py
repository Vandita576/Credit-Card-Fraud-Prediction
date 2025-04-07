import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Load data and train model once
@st.cache_data
def load_and_train():
    df = pd.read_csv(r"C:\Environment\AI PROJECT\creditcard.csv")
    X = df.drop('Class', axis=1)
    y = df['Class']
    model = DecisionTreeClassifier(criterion='entropy', random_state=0)
    model.fit(X, y)
    return df, model

df, model = load_and_train()

st.title("💳 Credit Card Fraud Detection")

# Predict Random Fraud
if st.button("Predict Random Fraud Transaction"):
    fraud_sample = df[df['Class'] == 1].sample(1)
    input_data = fraud_sample.drop('Class', axis=1)
    prediction = model.predict(input_data)[0]
    st.subheader("🔍 Prediction: Fraud" if prediction == 1 else "🔍 Prediction: Not Fraud")
    st.write("📄 Transaction data:")
    st.dataframe(input_data)

# Predict Random Non-Fraud
if st.button("Predict Random Non-Fraud Transaction"):
    nonfraud_sample = df[df['Class'] == 0].sample(1)
    input_data = nonfraud_sample.drop('Class', axis=1)
    prediction = model.predict(input_data)[0]
    st.subheader("🔍 Prediction: Fraud" if prediction == 1 else "🔍 Prediction: Not Fraud")
    st.write("📄 Transaction data:")
    st.dataframe(input_data)




