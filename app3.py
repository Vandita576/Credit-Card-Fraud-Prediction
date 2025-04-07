import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load and train the model once
@st.cache_data
def load_and_train():
    df = pd.read_csv(r"C:\Environment\AI PROJECT\creditcard.csv")
    X = df.drop('Class', axis=1)
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = DecisionTreeClassifier(criterion='entropy', random_state=0)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return df, model, acc

df, model, accuracy = load_and_train()

# App UI
st.title("💳 Credit Card Fraud Detection")
st.markdown(f"**🔎 Model Accuracy:** `{accuracy * 100:.2f}%`")

# Prediction buttons
col1, col2 = st.columns(2)

with col1:
    if st.button("🎲 Predict Random Fraud Transaction"):
        fraud_sample = df[df['Class'] == 1].sample(1)
        input_data = fraud_sample.drop('Class', axis=1)
        prediction = model.predict(input_data)[0]
        st.subheader("🔍 Prediction: Fraud" if prediction == 1 else "🔍 Prediction: Not Fraud")
        st.write("📄 Transaction data:")
        st.dataframe(input_data)

with col2:
    if st.button("🎲 Predict Random Non-Fraud Transaction"):
        nonfraud_sample = df[df['Class'] == 0].sample(1)
        input_data = nonfraud_sample.drop('Class', axis=1)
        prediction = model.predict(input_data)[0]
        st.subheader("🔍 Prediction: Fraud" if prediction == 1 else "🔍 Prediction: Not Fraud")
        st.write("📄 Transaction data:")
        st.dataframe(input_data)

# Dataset Viewer
st.markdown("---")
view_option = st.radio("📊 View Dataset:", ["Full Dataset", "Only Fraud Transactions"])

if view_option == "Full Dataset":
    st.dataframe(df)
else:
    st.dataframe(df[df['Class'] == 1])



import matplotlib.pyplot as plt

# Pie Chart
st.markdown("### 📈 Class Distribution")
class_counts = df['Class'].value_counts()
labels = ['Not Fraud', 'Fraud']
sizes = [class_counts[0], class_counts[1]]
colors = ['#66b3ff', '#ff6666']

fig, ax = plt.subplots()
ax.pie(sizes, labels=labels, autopct='%1.2f%%', startangle=90, colors=colors, explode=(0, 0.1))
ax.axis('equal')  # Equal aspect ratio ensures pie is drawn as a circle
st.pyplot(fig)

