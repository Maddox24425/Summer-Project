import streamlit as st
import pandas as pd
import numpy as np
import pickle
import random

st.header('Diabetes Prediction Using Machine Learning')

data = '''The Diabetes prediction dataset is a collection of medical and demographic data from patients, along with their diabetes status (positive or negative). The data includes features such as age, gender, body mass index (BMI), hypertension, heart disease, smoking history, HbA1c level, and blood glucose level. This dataset can be used to build machine learning models to predict diabetes in patients based on their medical history and demographic information. This can be useful for healthcare professionals in identifying patients who may be at risk of developing diabetes and in developing personalized treatment plans. Additionally, the dataset can be used by researchers to explore the relationships between various medical and demographic factors and the likelihood of developing diabetes.'''

st.markdown(data)


st.image('https://user-images.githubusercontent.com/103222259/255432423-8157a6d0-8d82-43b4-894f-53cd2125c891.png')

with open(r'D:\Data Science\Summer-Project\diabetes_model_XGB.pkl','rb') as f:
    chatgpt = pickle.load(f)

# Load data
url = '''https://github.com/Maddox24425/Summer-Project/blob/main/diabetes_prediction_dataset.csv?raw=true'''
df = pd.read_csv(url)


st.sidebar.header('Select Features to Diabetes')
st.sidebar.image('https://www.eresvihda.es/wp-content/uploads/2023/10/Diabetes.gif')
all_values = []


num_col=['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level',
       'blood_glucose_level']
for i in num_col:
    min_value, max_value = df[i].agg(['min','max'])

    var =st.sidebar.slider(f'Select {i} value', int(min_value), int(max_value), 
                      random.randint(int(min_value),int(max_value)))

    all_values.append(var)

#Gender dropdown here
gender_option = st.sidebar.selectbox("Select Gender", ["Male", "Female", "Other"])
if gender_option == "Male":
    gender_encoded = 1
elif gender_option == "Female":
    gender_encoded = 0


all_values.append(gender_encoded)


#Smoking History dropdown here
smoking_option = st.sidebar.selectbox("Select Smoking History",
                                      ["No Info", "current", "former", "never"])

smoking_No_Info = 1 if smoking_option == "No Info" else 0
smoking_current = 1 if smoking_option == "current" else 0
smoking_former = 1 if smoking_option == "former" else 0
smoking_never = 1 if smoking_option == "never" else 0

all_values.extend([smoking_No_Info, smoking_current, smoking_former, smoking_never])

final_value = [all_values]

ans = chatgpt.predict(final_value)[0]

import time
random.seed(132)
progress_bar = st.progress(0)
placeholder = st.empty()
placeholder.subheader('Predicting Diabetes') 

place = st.empty()
place.image('https://i.makeagif.com/media/1-17-2024/dw-jXM.gif',width = 200)

for i in range(100):
    time.sleep(0.05)
    progress_bar.progress(i + 1)

if ans == 0:
    body = f'yes'
    placeholder.empty()
    place.empty()
    st.success(body)
    progress_bar = st.progress(0)
else:
    body = 'no'
    placeholder.empty()
    place.empty()
    st.warning(body)
    progress_bar = st.progress(0)


st.markdown('Designed by: **Divyanshu Raj & Farhan Khan**')
