import streamlit as st
import pandas as pd
import numpy as np
import pickle
import random

st.header('Diabetes Prediction Using Machine Learning')

data = '''Trying'''

st.markdown(data)


st.image('https://www.shutterstock.com/image-vector/abstract-blue-human-heart-red-600nw-2526983201.jpg')

with open(r'D:\Data Science\Summer-Project\diabetes_model_XGB.pkl','rb') as f:
    chatgpt = pickle.load(f)

# Load data
url = '''https://github.com/Maddox24425/Summer-Project/blob/main/diabetes_prediction_dataset.csv?raw=true'''
df = pd.read_csv(url)


st.sidebar.header('Select Features to Diabetes')
st.sidebar.image('https://humanbiomedia.org/animations/circulatory-system/cardiac-cycle/heart-beating.gif')

all_values = []


col=['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level',
       'blood_glucose_level', 'gender_encoded', 'smoking_No_Info',
       'smoking_current', 'smoking_former', 'smoking_never']
for i in col:
    min_value, max_value = df[i].agg(['min','max'])

    var =st.sidebar.slider(f'Select {i} value', int(min_value), int(max_value), 
                      random.randint(int(min_value),int(max_value)))

    all_values.append(var)

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


st.markdown('Designed by: **Farhan Khan**')
