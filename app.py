# app.py
import pandas as pd
import streamlit as st
import joblib

# Load the models and label encoders
mlp_pipeline_medicine = joblib.load('medicine_predictor.pkl')
mlp_pipeline_dosage = joblib.load('dosage_predictor.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# Streamlit app
st.title('Medicine Predictor')

# User inputs
age = st.slider('Age', min_value=0, max_value=100, value=25)
sex = st.radio('Sex', options=label_encoders['Sex'].classes_)
symptom1 = st.selectbox('Symptom 1', options=[
    'Asthma', 'Diabetes', 'High Blood Pressure', 'Heart Disease', 'Allergy',
    'Runny Nose', 'Sneezing', 'Sore Throat', 'Cough', 'Fever', 'Headache',
    'Congestion', 'Fatigue', 'Body Aches'
])
symptom2 = st.selectbox('Symptom 2', options=[
    'Asthma', 'Diabetes', 'High Blood Pressure', 'Heart Disease', 'Allergy',
    'Runny Nose', 'Sneezing', 'Sore Throat', 'Cough', 'Fever', 'Headache',
    'Congestion', 'Fatigue', 'Body Aches'
])
symptom3 = st.selectbox('Symptom 3', options=[
    'Asthma', 'Diabetes', 'High Blood Pressure', 'Heart Disease', 'Allergy',
    'Runny Nose', 'Sneezing', 'Sore Throat', 'Cough', 'Fever', 'Headache',
    'Congestion', 'Fatigue', 'Body Aches'
])
symptom4 = st.selectbox('Symptom 4', options=[
    'Asthma', 'Diabetes', 'High Blood Pressure', 'Heart Disease', 'Allergy',
    'Runny Nose', 'Sneezing', 'Sore Throat', 'Cough', 'Fever', 'Headache',
    'Congestion', 'Fatigue', 'Body Aches'
])

# Create a dictionary to hold the symptom values
symptom_values = {
    'Asthma': 0,
    'Diabetes': 0,
    'High Blood Pressure': 0,
    'Heart Disease': 0,
    'Allergy': 0,
    'Runny Nose': 0,
    'Sneezing': 0,
    'Sore Throat': 0,
    'Cough': 0,
    'Fever': 0,
    'Headache': 0,
    'Congestion': 0,
    'Fatigue': 0,
    'Body Aches': 0,
    'No. of Days': 0  # Default value for No. of Days
}

# Update the symptom values based on user input
symptom_values[symptom1] = 1
symptom_values[symptom2] = 1
symptom_values[symptom3] = 1
symptom_values[symptom4] = 1

# Generate predictions when the button is pressed
if st.button('Generate'):
    # Create the input data for prediction
    input_data = pd.DataFrame({
        'Age': [age],
        'Sex': [label_encoders['Sex'].transform([sex])[0]],
        **{key: [value] for key, value in symptom_values.items()}
    })

    # Predict medicine
    predicted_medicine = mlp_pipeline_medicine.predict(input_data)
    predicted_medicine = label_encoders['Medicine'].inverse_transform(predicted_medicine)

    st.write(f'Predicted Medicine: {predicted_medicine[0]}')

    # Predict dosage
    predicted_dosage = mlp_pipeline_dosage.predict(input_data)
    predicted_dosage = label_encoders['Dosage'].inverse_transform(predicted_dosage)

    st.write(f'Recommended Dosage: {predicted_dosage[0]}')

# To run the Streamlit app, use the following command in your terminal:
# streamlit run app.py
