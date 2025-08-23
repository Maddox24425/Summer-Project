import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import random
import time
import kagglehub
import warnings
warnings.filterwarnings("ignore")

st.header("Diabetes Prediction Using Machine Learning")
data='''The Diabetes prediction dataset is a collection of medical and demographic data from patients, along with their diabetes status (positive or negative). The data includes features such as age, gender, body mass index (BMI), hypertension, heart disease, smoking history, HbA1c level, and blood glucose level. This dataset can be used to build machine learning models to predict diabetes in patients based on their medical history and demographic information. This can be useful for healthcare professionals in identifying patients who may be at risk of developing diabetes and in developing personalized treatment plans. Additionally, the dataset can be used by researchers to explore the relationships between various medical and demographic factors and the likelihood of developing diabetes.'''

#open our pkl file

with open (r"D:\Data Science\Summer-Project\diabetes_model_XGB.pkl","rb") as f:
    model=pickle.load(f)

st.subheader(data)

# adding data 
path = kagglehub.dataset_download("iammustafatz/diabetes-prediction-dataset")
print("Path to dataset files:", path)

all_files=os.listdir(path)
path=path+'/'+all_files[0]

data=pd.read_csv(path)

st.sidebar.header("Select feature to predict Diabetes")
st.sidebar.image("https://www.eresvihda.es/wp-content/uploads/2023/10/Diabetes.gif")


# gender encoding
data['gender_encoded'] = (data['gender'] == 'Male').astype(int)

#droping the original 'gender' column as it's no longer needed
data = data.drop('gender', axis=1)
# Male: 1
# Female: 0

smoking_replace_dict={'not current':'former',
                      'ever':'former'
                       ,'No Info': 'No_Info'
                      }
data.replace(smoking_replace_dict, inplace=True)

# Create the one-hot encoded columns
data=pd.get_dummies(data, columns=['smoking_history'], prefix='smoking', dtype=int)

col=['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level',
       'blood_glucose_level', 'gender_encoded', 'smoking_No_Info',
       'smoking_current', 'smoking_former', 'smoking_never']

all_values = []
random.seed(57)
for i in col:
    min_value, max_value = data[i].agg(['min', 'max'])
    var = st.sidebar.slider(f'Select {i} value', int(min_value), int(max_value),
                            random.randint(int(min_value), int(max_value)))
    all_values.append(var)

# Convert to numpy array
final_value = np.array(all_values).reshape(1, -1)  
# Or as DataFrame (better if you trained with DataFrame and column names)
final_value = pd.DataFrame([all_values], columns=col)
ans = model.predict(final_value)[0]




ans = model.predict(final_value)[0]


progress_bar = st.progress(0)
placeholder = st.empty()
placeholder.subheader('Predicting Diabetes')

place = st.empty()
place.image('https://media0.giphy.com/media/aPBXEeY01Dfp9tjyqi/giphy.gif', width=200)

for i in range(100):
    time.sleep(0.05)
    progress_bar.progress(i + 1)

if ans == 0:
    body = f'No Diabetes Detected'
    placeholder.empty()
    place.empty()
    st.success(body)
else:
    body = 'Diabetes Found'
    placeholder.empty()
    place.empty()
    st.warning(body)
