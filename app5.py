import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")

# Cache dataset loading
@st.cache_data
def load_data():
    return pd.read_csv(r"C:\Environment\AI PROJECT\creditcard.csv")

# Cache model training
@st.cache_resource
def train_model(data):
    X = data.drop('Class', axis=1)
    y = data['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = DecisionTreeClassifier(criterion='entropy', random_state=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return model, X, y, acc

# Load and train
df = load_data()
model, X, y, acc = train_model(df)

# --- UI ---
st.title("💳 Credit Card Fraud Detection Dashboard")

# KPI Metrics
col1, col2, col3 = st.columns(3)
total_txns = len(df)
fraud_count = df[df['Class'] == 1].shape[0]
fraud_percent = (fraud_count / total_txns) * 100
col1.metric("Total Transactions", f"{total_txns:,}")
col2.metric("Fraud Cases", f"{fraud_count:,}", f"{fraud_percent:.2f}%")
col3.metric("Model Accuracy", f"{acc*100:.2f}%")

# --- Prediction Buttons ---
fraud_sample = df[df['Class'] == 1].sample(1, random_state=np.random.randint(1000))
nonfraud_sample = df[df['Class'] == 0].sample(1, random_state=np.random.randint(1000))

col4, col5 = st.columns(2)

with col4:
    if st.button("🔍 Predict Random Fraud Transaction"):
        input_data = fraud_sample.drop(['Class'], axis=1)
        prediction = model.predict(input_data)[0]
        st.write("Prediction:", "🛑 Fraud" if prediction == 1 else "✅ Not Fraud")
        st.dataframe(input_data)

with col5:
    if st.button("🔍 Predict Random Non-Fraud Transaction"):
        input_data = nonfraud_sample.drop(['Class'], axis=1)
        prediction = model.predict(input_data)[0]
        st.write("Prediction:", "🛑 Fraud" if prediction == 1 else "✅ Not Fraud")
        st.dataframe(input_data)

# --- Dataset Viewer ---
st.markdown("### 📊 Dataset Viewer")
view_option = st.radio("Select data to view:", ["Full Dataset", "Only Fraud", "Only Non-Fraud"])

if view_option == "Full Dataset":
    st.dataframe(df)
elif view_option == "Only Fraud":
    st.dataframe(df[df['Class'] == 1])
else:
    st.dataframe(df[df['Class'] == 0])

