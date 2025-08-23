import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import warnings

# --- Page Configuration ---
st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="wide"
)

# Suppress warnings for a cleaner interface
warnings.filterwarnings("ignore")

# --- Model and Data Loading ---
# This function caches the model to avoid reloading it on every interaction.
@st.cache_resource
def load_xgboost_model():
    """Load the pre-trained XGBoost model."""
    try:
        # IMPORTANT: You must upload your actual XGBoost model file
        # and ensure it's named 'diabetes_model_XGB.pkl'.
        model = joblib.load("diabetes_model_XGB.pkl")
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        st.info("Please ensure a valid XGBoost model file named 'diabetes_model_XGB.pkl' is in the repository.")
        return None

@st.cache_data
def load_dataset():
    """Load the dataset from a local CSV file."""
    try:
        # IMPORTANT: You must add the 'diabetes_prediction_dataset.csv' file to your GitHub repository.
        data = pd.read_csv("diabetes_prediction_dataset.csv")
        return data
    except FileNotFoundError:
        st.error("Dataset file not found.")
        st.info("Please upload 'diabetes_prediction_dataset.csv' to your GitHub repository.")
        return None

model = load_xgboost_model()
data = load_dataset()

# --- Main Application ---
if model is not None and data is not None:
    # --- UI Setup ---
    st.image("https://user-images.githubusercontent.com/103222259/255432423-8157a6d0-8d82-43b4-894f-53cd2125c891.png")
    st.title("🚀 Diabetes Prediction Using XGBoost")

    st.markdown("""
    This application predicts the likelihood of diabetes using a powerful **XGBoost** model.
    Adjust the sliders in the sidebar to match the patient's data and click the predict button.
    """)
    st.markdown("---")

    # --- Sidebar for User Input ---
    st.sidebar.header("Select Patient Features")
    st.sidebar.image("https://www.eresvihda.es/wp-content/uploads/2023/10/Diabetes.gif")

    # --- User Input Collection ---
    input_values = {}
    
    input_values['age'] = st.sidebar.slider('Age', int(data['age'].min()), int(data['age'].max()), 25)
    input_values['bmi'] = st.sidebar.slider('Body Mass Index (BMI)', float(data['bmi'].min()), float(data['bmi'].max()), 22.0)
    input_values['HbA1c_level'] = st.sidebar.slider('HbA1c Level', float(data['HbA1c_level'].min()), float(data['HbA1c_level'].max()), 5.7)
    input_values['blood_glucose_level'] = st.sidebar.slider('Blood Glucose Level', int(data['blood_glucose_level'].min()), int(data['blood_glucose_level'].max()), 100)

    gender_map = {'Female': 0, 'Male': 1}
    gender = st.sidebar.selectbox('Gender', list(gender_map.keys()))
    input_values['gender'] = gender_map[gender]

    input_values['hypertension'] = st.sidebar.selectbox('Hypertension', [0, 1], help="0 = No, 1 = Yes")
    input_values['heart_disease'] = st.sidebar.selectbox('Heart Disease', [0, 1], help="0 = No, 1 = Yes")

    smoking_history_map = {'never': 4, 'No Info': 0, 'current': 1, 'former': 2, 'ever': 3, 'not current': 5}
    smoking_status = st.sidebar.selectbox('Smoking History', list(smoking_history_map.keys()))
    input_values['smoking_history'] = smoking_history_map[smoking_status]


    # --- Prediction Logic ---
    if st.sidebar.button("Predict Diabetes Status", type="primary"):
        # Create a DataFrame from user inputs.
        # The column order must exactly match the order used for training the XGBoost model.
        # We will create a dataframe with all possible columns and then select the ones we need
        # in the correct order.
        
        # This is a simplified feature engineering step. Ensure it matches your training script.
        input_df = pd.DataFrame([input_values])

        # Define the feature order the model expects.
        # This is the most common source of errors. Double-check your training script!
        feature_order = [
            'gender', 'age', 'hypertension', 'heart_disease', 'smoking_history',
            'bmi', 'HbA1c_level', 'blood_glucose_level'
        ]
        
        # Reorder the dataframe columns
        input_df = input_df[feature_order]

        # Get the prediction (XGBoost typically outputs 0 or 1 directly)
        prediction = model.predict(input_df)[0]
        
        # --- Display Results ---
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(100):
            time.sleep(0.02)
            progress_bar.progress(i + 1)
        
        status_text.subheader("Prediction Complete!")
        
        if prediction == 0:
            st.success('**Result:** No Diabetes Detected')
            st.image('https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExb252M2VpZWZ2aGZmMmZ5a2V0Z3V5a3J0eGU5b3Nuc3RzZzY3a3Y4eSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/JpG2A9P3dPH2w/giphy.gif', width=300)
        else:
            st.warning('**Result:** Diabetes Found')
            st.image('https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExM2J2a3JtY255enV2N2U0b21iY210a3d0b2N3eWJ2aHk0d2R0c2g5MyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/3o7WTqo27pLRYxRtg4/giphy.gif', width=300)

    st.markdown("---")
    st.markdown("Design By : Divyanshu Raj")

else:
    st.warning("Application could not start. Please check the file requirements in the information boxes above.")

