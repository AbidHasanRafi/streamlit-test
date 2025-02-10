import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("swell_model_lstm.h5")

# Define the feature names (same as in your training data)
features = [
    'MEAN_RR', 'MEDIAN_RR', 'SDRR', 'RMSSD', 'SDSD', 'SDRR_RMSSD', 'HR',
    'pNN25', 'pNN50', 'SD1', 'SD2', 'KURT', 'SKEW', 'MEAN_REL_RR', 'MEDIAN_REL_RR',
    'SDRR_REL_RR', 'RMSSD_REL_RR', 'SDSD_REL_RR', 'SDRR_RMSSD_REL_RR', 'KURT_REL_RR',
    'SKEW_REL_RR', 'VLF', 'VLF_PCT', 'LF', 'LF_PCT', 'LF_NU', 'HF', 'HF_PCT', 'HF_NU',
    'TP', 'LF_HF', 'HF_LF', 'sampen', 'higuci'
]

# Load LabelEncoder for encoding/decoding conditions
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
label_encoder.fit(["No Stress", "Time Pressure", "Interruption"])

# Streamlit UI
st.title("SWELL Dataset Condition Prediction")

st.write("Enter the values for the features to predict the condition:")

# Create input fields for each feature
inputs = {}
for feature in features:
    inputs[feature] = st.number_input(f"Enter {feature}", min_value=-100.0, max_value=100.0, value=0.0, step=0.01)

# When the user clicks the "Predict" button
if st.button("Predict"):
    # Prepare the input data (turn it into a 2D array)
    input_data = np.array(list(inputs.values())).reshape(1, 1, -1)

    # Standardize the input data
    scaler = StandardScaler()
    input_data = scaler.fit_transform(input_data[0].reshape(-1, 1)).reshape(1, 1, -1)  # Reshaped for LSTM input

    # Make the prediction
    prediction = model.predict(input_data)

    # Decode the prediction
    predicted_condition = label_encoder.inverse_transform([np.argmax(prediction)])

    # Display the result
    st.write(f"Predicted Condition: {predicted_condition[0]}")

