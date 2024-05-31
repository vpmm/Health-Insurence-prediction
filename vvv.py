import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import matplotlib.pyplot as plt

# Streamlit application
st.title("Health Insurance Cost Prediction")

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv('/home/vs/Downloads/insurance (1).csv')

medical_dataset = load_data()

# Clone the dataset to prevent mutation warnings
medical_dataset = medical_dataset.copy()

# User inputs
st.sidebar.header("User Input Parameters")
age = st.sidebar.slider("Age", 18, 100, 30)
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
bmi = st.sidebar.slider("BMI", 10.0, 50.0, 25.0)
children = st.sidebar.slider("Children", 0, 10, 0)
smoker = st.sidebar.selectbox("Smoker", ["Yes", "No"])
region = st.sidebar.selectbox("Region", ["Southeast", "Southwest", "Northeast", "Northwest"])

# Convert categorical variables to numerical
sex = 1 if sex == 'Female' else 0
smoker = 0 if smoker == 'Yes' else 1
region = ['Southeast', 'Southwest', 'Northeast', 'Northwest'].index(region)

# Data Preprocessing
medical_dataset.replace({'sex': {'male': 0, 'female': 1}}, inplace=True)
medical_dataset.replace({'smoker': {'yes': 0, 'no': 1}}, inplace=True)
medical_dataset.replace({'region': {'southeast': 0, 'southwest': 1, 'northeast': 2, 'northwest': 3}}, inplace=True)

# Splitting the feature and target
X = medical_dataset.drop(columns='charges', axis=1)
Y = medical_dataset['charges']

# Model Preparation
gbm_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit both models on the data
gbm_model.fit(X, Y)
rf_model.fit(X, Y)

# Combine predictions from both models
def ensemble_predict(X):
    gbm_pred = gbm_model.predict(X)
    rf_pred = rf_model.predict(X)
    ensemble_pred = (gbm_pred + rf_pred) / 2
    return ensemble_pred

# Predict insurance charge
user_input_data = np.array([age, sex, bmi, children, smoker, region]).reshape(1, -1)
predicted_charge = ensemble_predict(user_input_data)[0]

# Determine obesity status based on BMI
obesity_status = "Obese" if bmi >= 30 else "Not Obese"

# Display predicted charge with bold and highlighted label
st.markdown(f'<p style="font-size:24px; font-weight:bold;">Predicted Insurance Charge:</p><p style="font-size:20px; color:blue;">${predicted_charge:.2f}</p>', unsafe_allow_html=True)

# Display obesity status
st.write(f"Obesity Status: {obesity_status}")

# Scatter plot section
fig, ax = plt.subplots(len(X.columns), 1, figsize=(10, 20))
colors = np.where(medical_dataset['smoker'] == 0, 'red', 'blue')
sex_colors = np.where(medical_dataset['sex'] == 1, 'green', 'orange')

for ind, col in enumerate(X.columns):
    ax[ind].scatter(X[col], Y, c=colors if col == 'smoker' else sex_colors, s=5)
    ax[ind].set_xlabel(col)
    ax[ind].set_ylabel("Charges")
    if col == 'smoker':
        ax[ind].text(0.95, 0.95, 'Red: Smokers\nBlue: Non-smokers', ha='right', va='top', transform=ax[ind].transAxes)
    elif col == 'sex':
        ax[ind].text(0.95, 0.95, 'Green: Female\nOrange: Male', ha='right', va='top', transform=ax[ind].transAxes)

for i, col in enumerate(X.columns):
    if col == 'smoker':
        ax[i].scatter(user_input_data[0][i], predicted_charge, color='red', marker='o')
    else:
        ax[i].scatter(user_input_data[0][i], predicted_charge, color='red', marker='o', label='User Input')
        ax[i].legend()

plt.tight_layout()
st.pyplot(fig)
