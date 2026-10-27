import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load('SVC_Heart_model.pkl')
scaler = joblib.load('scaler_Heart_model.pkl')
expecated_columns = joblib.load('Column_Heart_model.pkl')

st.title('Heart Disease Prediction')
st.write('This app predicts heart disease using machine learning')
st.sidebar.header('Enter your details')

age = st.sidebar.number_input('Age', 18,100,40)
sex = st.sidebar.radio('Sex', ["M", "F"])
pain = st.sidebar.selectbox('Chest Pain Type', ["Typical Angina", "ATA", "NAP", "TA","ASY"])
rbp = st.sidebar.number_input('Resting Blood Pressure(mm,Hg)', 80,200,120)
cholesterol = st.sidebar.number_input('Cholesterol (mg/dl)', 10,564,200)
fbs = st.sidebar.selectbox('Fasting Blood Sugar > 120 (mg/dl)', [0,1])
restecg = st.sidebar.selectbox('Resting Electrocardiographic Results', ["Normal", "ST", "LVH"])
mhr = st.sidebar.number_input('Maximum Heart Rate (bpm)', 71,202,150)
exang = st.sidebar.radio('Exercise Induced Angina', ["Y", "N"])
oldpeak = st.sidebar.number_input('ST Depression Induced by Exercise', 0.0,6.2,2.0)
slope = st.sidebar.selectbox('Slope of the Peak Exercise ST Segment', ["Upsloping", "Flat", "Downsloping"])

if st.button('Predict'):
    # Age	Sex	ChestPainType	RestingBP	Cholesterol	FastingBS	RestingECG	MaxHR	ExerciseAngina	Oldpeak	ST_Slope	
    data = {'Age':age,'Sex':sex,'ChestPainType':pain,'RestingBP':rbp,'Cholesterol':cholesterol,'FastingBS':fbs,'RestingECG':restecg,'MaxHR':mhr,'ExerciseAngina':exang,'Oldpeak':oldpeak,'ST_Slope':slope}

    input_df=pd.DataFrame([data])

    # Scale numerical features first
    num_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
    input_df[num_cols] = scaler.transform(input_df[num_cols])

    # Add categorical dummy features expected by the model
    input_df['is_Male'] = 1 if sex == 'M' else 0
    input_df['ChestPainType_ATA'] = 1 if pain == 'ATA' else 0
    input_df['ChestPainType_NAP'] = 1 if pain == 'NAP' else 0
    input_df['RestingECG_ST'] = 1 if restecg == 'ST' else 0
    input_df['ExerciseAngina_Y'] = 1 if exang == 'Y' else 0
    input_df['ST_Slope_Flat'] = 1 if slope == 'Flat' else 0
    input_df['ST_Slope_Up'] = 1 if slope == 'Upsloping' else 0

    for col in expecated_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expecated_columns]
    prediction = model.predict(input_df)
    if prediction[0] == 1:
        st.write('Heart Disease')
    else:
        st.write('No Heart Disease')