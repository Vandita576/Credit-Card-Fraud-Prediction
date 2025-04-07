import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")

@st.cache_resource
def load_and_train():
    df = pd.read_csv(r"C:\Environment\AI PROJECT\creditcard.csv")
    X = df.drop('Class', axis=1)
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = DecisionTreeClassifier(criterion='entropy', random_state=0)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, df, acc

model, df, acc = load_and_train()

st.title("💳 Credit Card Fraud Detection")
st.markdown(f"### ✅ Model Accuracy: `{acc*100:.2f}%`")

st.markdown("---")
st.markdown("### 🎲 Random Transaction Predictions")

col1, col2 = st.columns(2)

with col1:
    if st.button("🔍 Predict Random Fraud Transaction"):
        fraud_sample = df[df['Class'] == 1].sample(1, random_state=np.random.randint(1000))
        input_data = fraud_sample.drop(['Class'], axis=1)
        prediction = model.predict(input_data)[0]
        st.success("Prediction: 🛑 Fraud" if prediction == 1 else "✅ Not Fraud")
        st.dataframe(input_data)

with col2:
    if st.button("🔍 Predict Random Non-Fraud Transaction"):
        nonfraud_sample = df[df['Class'] == 0].sample(1, random_state=np.random.randint(1000))
        input_data = nonfraud_sample.drop(['Class'], axis=1)
        prediction = model.predict(input_data)[0]
        st.success("Prediction: 🛑 Fraud" if prediction == 1 else "✅ Not Fraud")
        st.dataframe(input_data)

st.markdown("---")
st.markdown("### 📊 Dataset Viewer")
view_option = st.radio("Select data to view:", ["Full Dataset", "Only Fraud", "Only Non-Fraud"])

if view_option == "Full Dataset":
    st.dataframe(df.head(1000))  # Show first 1000 rows to improve speed
elif view_option == "Only Fraud":
    st.dataframe(df[df['Class'] == 1].head(1000))
else:
    st.dataframe(df[df['Class'] == 0].head(1000))

# Optional Visualization: Pie chart of class distribution
st.markdown("---")
st.markdown("### 📈 Fraud vs Non-Fraud Distribution")
fig, ax = plt.subplots()
labels = ['Non-Fraud', 'Fraud']
sizes = df['Class'].value_counts().sort_index()
ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=['#00cc99', '#ff6666'], startangle=140)
ax.axis('equal')
st.pyplot(fig)
