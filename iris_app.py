import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("iris_model.pkl")

# Iris class names
class_names = ['setosa', 'versicolor', 'virginica']

st.title("🌸 Iris Flower Classifier")
st.write("Enter flower measurements to predict the species.")

# Input fields for measurements
sepal_length = st.slider("Sepal length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal width (cm)", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal length (cm)", 1.0, 7.0, 1.4)
petal_width = st.slider("Petal width (cm)", 0.1, 2.5, 0.2)

# Predict when button is clicked
if st.button("Predict"):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    pred = model.predict(features)[0]
    st.success(f"🌟 Predicted class: **{class_names[pred]}**")
