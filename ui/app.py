import streamlit as st
import pandas as pd
import numpy as np
import joblib
st.markdown(
    """
    <style>
    .stApp {
        background-color: #87BFE5;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.set_page_config(page_title="Heart Disease Predictor", layout="wide")

def load_and_predict(input_df):
    try:
        scaler = joblib.load('C:/MENNA_DATA/Heart_Disease_Project/models/scaler.pkl')
        pca = joblib.load('C:/MENNA_DATA/Heart_Disease_Project/models/pca.pkl')
        model = joblib.load('C:/MENNA_DATA/Heart_Disease_Project/models/final_model.pkl')

        try:
            cleaned_df = pd.read_csv('C:/MENNA_DATA/Heart_Disease_Project/dataset attributes/cleaned_dataset.csv')
            numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
            categorical_cols = [col for col in cleaned_df.drop('target', axis=1).columns if col not in numerical_cols]
            selected_features_df = pd.read_csv('C:/MENNA_DATA/Heart_Disease_Project/dataset attributes/feature_selected_dataset.csv')
            final_features = selected_features_df.drop('target', axis=1).columns.tolist()

        except FileNotFoundError:
            st.error("Error: One or more required CSV files ('cleaned_heart_disease.csv' or 'feature_selected_dataset.csv') not found. Please ensure they exist.")
            return None
        
        input_df_numerical = input_df[numerical_cols]
        input_df_categorical = input_df[categorical_cols]
        input_scaled_numerical = scaler.transform(input_df_numerical)
        input_scaled_numerical_df = pd.DataFrame(input_scaled_numerical, columns=numerical_cols)
        input_combined_df = pd.concat([input_scaled_numerical_df, input_df_categorical], axis=1)
        pca_transformed = pca.transform(input_combined_df)
        pca_cols = [f'PC_{i+1}' for i in range(pca_transformed.shape[1])]
        pca_transformed_df = pd.DataFrame(pca_transformed, columns=pca_cols)
        final_input_df = pca_transformed_df[final_features]
        prediction = model.predict(final_input_df)
        
        return prediction[0]
        
    except FileNotFoundError as e:
        st.error(f"Error: One or more required model files not found. Please ensure the following files exist in the app's directory: 'scaler.pkl', 'pca.pkl', 'final_model.pkl'.")
        return None
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        return None
from PIL import Image


img = Image.open("C:/MENNA_DATA/Heart_Disease_Project/background.jpg/heart_disease.jpg")


img = img.resize((800, 200))  

st.image(img, use_container_width=True)

st.markdown(
    """
    <div style="text-align: center; margin-top: -100px;">
        <h1 style="color: #ffffff; font-size: 40px; margin: 0;">Heart Checkup App</h1>
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown(
    '<p style="text-align: center; color: #212121;">Provide the required information below to assess your heart health risk 🩺</p>',
    unsafe_allow_html=True
)
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=50  )
    sex = st.selectbox("Sex", options=[("Male", 1.0), ("Female", 0.0)], format_func=lambda x: x[0])
    cp = st.selectbox("Chest Pain Type", options=[("Typical Angina", 1.0), ("Atypical Angina", 2.0), ("Non-Anginal Pain", 3.0), ("Asymptomatic", 4.0)], format_func=lambda x: x[0])
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=200, value=120)

with col2:
    chol = st.number_input("Cholesterol (mg/dl)", min_value=50, max_value=600, value=250)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[("No", 0.0), ("Yes", 1.0)], format_func=lambda x: x[0])
    restecg = st.selectbox("Resting ECG Results", options=[("Normal", 0.0), ("ST-T Wave Abnormality", 1.0), ("Left Ventricular Hypertrophy", 2.0)], format_func=lambda x: x[0])
    thalach = st.number_input("Max Heart Rate Achieved", min_value=50, max_value=250, value=150)
    thal = st.selectbox("Thalassemia", options=[("Normal", 3.0), ("Fixed Defect", 6.0), ("Reversible Defect", 7.0)], format_func=lambda x: x[0])

with col3:
    exang = st.selectbox("Exercise Induced Angina", options=[("No", 0.0), ("Yes", 1.0)], format_func=lambda x: x[0])
    oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    slope = st.selectbox("Slope of ST Segment", options=[("Upsloping", 1.0), ("Flat", 2.0), ("Downsloping", 3.0)], format_func=lambda x: x[0])
    ca = st.number_input("Number of Major Vessels (0-3)", min_value=0, max_value=3, value=0)

if st.button("Check Risk"):

    try:
        cleaned_df = pd.read_csv('C:/MENNA_DATA/Heart_Disease_Project/dataset attributes/cleaned_dataset.csv')
        original_cols = cleaned_df.drop('target', axis=1).columns.tolist()
    except FileNotFoundError:
        st.error("Error: The 'cleaned_dataset.csv' file could not be found. This is needed to get the correct column order for preprocessing.")
        st.stop()
    
    input_data_dict = {col: 0.0 for col in original_cols}
    input_data_dict['age'] = age
    input_data_dict['trestbps'] = trestbps
    input_data_dict['chol'] = chol
    input_data_dict['thalach'] = thalach
    input_data_dict['oldpeak'] = oldpeak
    input_data_dict[f'sex_{int(sex[1])}.0'] = 1.0
    input_data_dict[f'cp_{int(cp[1])}.0'] = 1.0
    input_data_dict[f'fbs_{int(fbs[1])}.0'] = 1.0
    input_data_dict[f'restecg_{int(restecg[1])}.0'] = 1.0
    input_data_dict[f'exang_{int(exang[1])}.0'] = 1.0
    input_data_dict[f'slope_{int(slope[1])}.0'] = 1.0
    input_data_dict[f'ca_{int(ca)}.0'] = 1.0
    input_data_dict[f'thal_{int(thal[1])}.0'] = 1.0

    input_df = pd.DataFrame([input_data_dict], columns=original_cols)
    prediction = load_and_predict(input_df)
    
    if prediction is not None:
        if prediction == 1:
            st.error("##### ❤️‍🩹 **Elevated Risk of Heart Disease – Please consult a healthcare professional.**")
        else:
            st.success("##### ❤️ **No Significant Risk of Heart Disease Detected – No immediate concerns identified.**")
        st.warning(
    "⚠️ **Important Notice:** The results provided by this model are for educational and informational purposes only. "
    "They should not be considered as medical advice, diagnosis, or treatment. "
    "Always consult a qualified healthcare professional regarding any concerns or questions about your health."
)
        st.markdown(
    "<p style='text-align: center; color: grey;'>Developed by <b>Menna Kamel</b></p>", 
    unsafe_allow_html=True
)