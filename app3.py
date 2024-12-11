import streamlit as st
import pickle
import numpy as np

# Load the pre-trained model
with open('face_mask_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Set up the Streamlit app
st.title("Face Mask Detection App")
st.write("This app predicts whether a person is wearing a face mask or not.")

# Input fields for user data
st.sidebar.header("Input Features")
def user_input_features():
    feature1 = st.sidebar.slider("Feature 1", 0.0, 1.0, 0.5)
    feature2 = st.sidebar.slider("Feature 2", 0.0, 1.0, 0.5)
    feature3 = st.sidebar.slider("Feature 3", 0.0, 1.0, 0.5)
    feature4 = st.sidebar.slider("Feature 4", 0.0, 1.0, 0.5)
    return np.array([[feature1, feature2, feature3, feature4]])

# Get user input
input_data = user_input_features()

# Display input data
st.write("### Input Features:")
st.write(input_data)

# Make predictions
if st.button("Predict"):
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    st.write("### Prediction:")
    st.write("Mask Detected" if prediction[0] == 1 else "No Mask Detected")

    st.write("### Prediction Probabilities:")
    st.write(f"Mask: {prediction_proba[0][1]:.2f}, No Mask: {prediction_proba[0][0]:.2f}")

