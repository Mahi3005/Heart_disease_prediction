import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import streamlit as st


# Load the data
@st.cache_data
def load_data():
    return pd.read_csv('datasets_4123_6408_framingham.csv')


df = load_data()

# Data preprocessing
df = df.drop(columns=['education'])
df_clean = df.copy()

bin_cols = ["male", "currentSmoker", "prevalentStroke", "prevalentHyp", "diabetes"]
numeric_cols = ["cigsPerDay", "BPMeds", "totChol", "BMI", "heartRate", "glucose"]

for col in bin_cols:
    if col in df_clean.columns:
        mode_val = df_clean[col].mode().iloc[0]
        df_clean[col] = df_clean[col].fillna(mode_val)

for col in numeric_cols:
    if col in df_clean.columns:
        median_val = df_clean[col].median()
        df_clean[col] = df_clean[col].fillna(median_val)

# Balance Dataset
df_majority = df_clean[df_clean['TenYearCHD'] == 0]
df_minority = df_clean[df_clean['TenYearCHD'] == 1]
df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=42)
df_balanced = pd.concat([df_majority, df_minority_upsampled])

# Train Test Split and Feature Scaling
X = df_balanced.drop(columns=['TenYearCHD'])
y = df_balanced['TenYearCHD']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Train the model
@st.cache_resource
def train_model():
    rf_classifier = RandomForestClassifier(random_state=42)
    rf_classifier.fit(X_train_scaled, y_train)
    return rf_classifier


rf_classifier = train_model()


# Updated prediction function
def predict(model, scaler, features_df):
    scaled_features = scaler.transform(features_df)
    result = model.predict(scaled_features)
    probability = model.predict_proba(scaled_features)[0][1]  # Probability of class 1
    return result[0], probability


# Streamlit App
st.title('Heart Disease Prediction App')

st.write("""
This app predicts the likelihood of heart disease based on various health factors.
Please fill in the following information:
""")

# Input fields
col1, col2 = st.columns(2)

with col1:
    male = st.selectbox('Gender', ['Male', 'Female'])
    age = st.number_input('Age', min_value=0, max_value=120, value=50)
    currentSmoker = st.selectbox('Current Smoker', ['Yes', 'No'])
    cigsPerDay = st.number_input('Cigarettes per Day', min_value=0, max_value=100, value=0)
    BPMeds = st.selectbox('BP Medication', ['Yes', 'No'])
    prevalentStroke = st.selectbox('Prevalent Stroke', ['Yes', 'No'])
    prevalentHyp = st.selectbox('Prevalent Hypertension', ['Yes', 'No'])

with col2:
    diabetes = st.selectbox('Diabetes', ['Yes', 'No'])
    totChol = st.number_input('Total Cholesterol', min_value=0, value=200)
    sysBP = st.number_input('Systolic BP', min_value=0, value=120)
    diaBP = st.number_input('Diastolic BP', min_value=0, value=80)
    BMI = st.number_input('BMI', min_value=0.0, max_value=60.0, value=25.0)
    heartRate = st.number_input('Heart Rate', min_value=0, max_value=200, value=70)
    glucose = st.number_input('Glucose Level', min_value=0, value=80)

# Prediction button
if st.button('Predict Heart Disease Risk'):
    # Prepare input data
    input_data = {
        'male': 1 if male == 'Male' else 0,
        'age': age,
        'currentSmoker': 1 if currentSmoker == 'Yes' else 0,
        'cigsPerDay': cigsPerDay,
        'BPMeds': 1 if BPMeds == 'Yes' else 0,
        'prevalentStroke': 1 if prevalentStroke == 'Yes' else 0,
        'prevalentHyp': 1 if prevalentHyp == 'Yes' else 0,
        'diabetes': 1 if diabetes == 'Yes' else 0,
        'totChol': totChol,
        'sysBP': sysBP,
        'diaBP': diaBP,
        'BMI': BMI,
        'heartRate': heartRate,
        'glucose': glucose
    }

    features_df = pd.DataFrame([input_data])

    # Make prediction
    result, probability = predict(rf_classifier, scaler, features_df)

    # Display result
    st.write(f"Prediction: {'High' if result == 1 else 'Low'} risk of heart disease")
    st.write(f"Probability of heart disease: {probability:.2f}")

    if result == 1:
        st.error('High risk of heart disease. Please consult with a healthcare professional.')
    else:
        st.success('Low risk of heart disease. Keep maintaining a healthy lifestyle!')

    # Display input data for verification
    st.write("Input data:")
    st.write(input_data)

# Model performance metrics
st.sidebar.header('Model Performance')
y_pred = rf_classifier.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
st.sidebar.write(f'Model Accuracy: {accuracy:.2f}')

st.sidebar.write('Classification Report:')
st.sidebar.code(classification_report(y_test, y_pred))

st.sidebar.write('Confusion Matrix:')
st.sidebar.write(confusion_matrix(y_test, y_pred))

# Disclaimer
st.sidebar.header('Disclaimer')
st.sidebar.write("""
This app is for educational purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.
""")