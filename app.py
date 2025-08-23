import os
import joblib   # ✅ use joblib instead of pickle
import numpy as np
import pandas as pd
import streamlit as st
import random
import time
import kagglehub
import warnings
warnings.filterwarnings("ignore")

st.image("https://user-images.githubusercontent.com/103222259/255432423-8157a6d0-8d82-43b4-894f-53cd2125c891.png")

st.header("Diabetes Prediction Using Machine Learning")

data_info = '''The Diabetes prediction dataset is a collection of medical and demographic data from patients, along with their diabetes status (positive or negative). The data includes features such as age, gender, body mass index (BMI), hypertension, heart disease, smoking history, HbA1c level, and blood glucose level. This dataset can be used to build machine learning models to predict diabetes in patients based on their medical history and demographic information. This can be useful for healthcare professionals in identifying patients who may be at risk of developing diabetes and in developing personalized treatment plans. Additionally, the dataset can be used by researchers to explore the relationships between various medical and demographic factors and the likelihood of developing diabetes.'''
st.subheader(data_info)

# ✅ Load model safely with joblib
model = joblib.load("diabetes_model_XGB.pkl")
joblib.dump(model, "diabetes_model_XGB.pkl")

# ✅ Load dataset
path = kagglehub.dataset_download("iammustafatz/diabetes-prediction-dataset")
all_files = os.listdir(path)
path = os.path.join(path, all_files[0])
data = pd.read_csv(path)

st.sidebar.header("Select feature to predict Diabetes")
st.sidebar.image("https://www.eresvihda.es/wp-content/uploads/2023/10/Diabetes.gif")

# ✅ Gender encoding
data['gender_encoded'] = (data['gender'] == 'Male').astype(int)
data = data.drop('gender', axis=1)

# ✅ Clean smoking column
smoking_replace_dict = {
    'not current': 'former',
    'ever': 'former',
    'No Info': 'No_Info'
}
data.replace(smoking_replace_dict, inplace=True)

# ✅ One-hot encoding with all categories enforced
data['smoking_history'] = pd.Categorical(
    data['smoking_history'], 
    categories=['No_Info', 'current', 'former', 'never']
)
data = pd.get_dummies(data, columns=['smoking_history'], prefix='smoking', dtype=int)

# ✅ Ensure all smoking columns exist
for col_name in ['smoking_No_Info', 'smoking_current', 'smoking_former', 'smoking_never']:
    if col_name not in data.columns:
        data[col_name] = 0

col = ['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level',
       'blood_glucose_level', 'gender_encoded',
       'smoking_No_Info', 'smoking_current', 'smoking_former', 'smoking_never']

# ✅ Sidebar input sliders
all_values = []
random.seed(57)
for i in col:
    min_value, max_value = data[i].agg(['min', 'max'])
    var = st.sidebar.slider(f'Select {i} value',
                            int(min_value), int(max_value),
                            random.randint(int(min_value), int(max_value)))
    all_values.append(var)

# ✅ Convert to DataFrame for prediction
final_value = pd.DataFrame([all_values], columns=col)

# ✅ Predict
ans = model.predict(final_value)[0]

# ✅ Progress bar animation
progress_bar = st.progress(0)
placeholder = st.empty()
placeholder.subheader('Predicting Diabetes')

place = st.empty()
place.image('https://media0.giphy.com/media/aPBXEeY01Dfp9tjyqi/giphy.gif', width=200)

for i in range(100):
    time.sleep(0.05)
    progress_bar.progress(i + 1)

# ✅ Show result
if ans == 0:
    placeholder.empty()
    place.empty()
    st.success('No Diabetes Detected')
else:
    placeholder.empty()
    place.empty()
    st.warning('Diabetes Found')

st.markdown("Design By : Divyanshu Raj")