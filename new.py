import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the saved model
with open('BestFitmodel.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Define the function for prediction
def predict_churn(input_data):
    input_df = pd.DataFrame([input_data], columns=[
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure_group', 'PhoneService',
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
        'PaymentMethod', 'MonthlyCharges', 'TotalCharges'
    ])
    prediction = loaded_model.predict(input_df)
    return 'Churn' if prediction[0] == 1 else 'Not Churn'

# Streamlit app layout
st.title("Customer Churn Prediction App")

# Collecting user input
gender = st.selectbox("Gender", ['Male', 'Female'])
senior_citizen = st.selectbox("Senior Citizen", ['Yes', 'No'])
partner = st.selectbox("Partner", ['Yes', 'No'])
dependents = st.selectbox("Dependents", ['Yes', 'No'])
phone_service = st.selectbox("Phone Service", ['Yes', 'No'])
multiple_lines = st.selectbox("Multiple Lines", ['Yes', 'No', 'No phone service'])
internet_service = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
online_security = st.selectbox("Online Security", ['Yes', 'No', 'No internet service'])
online_backup = st.selectbox("Online Backup", ['Yes', 'No', 'No internet service'])
device_protection = st.selectbox("Device Protection", ['Yes', 'No', 'No internet service'])
tech_support = st.selectbox("Tech Support", ['Yes', 'No', 'No internet service'])
streaming_tv = st.selectbox("Streaming TV", ['Yes', 'No', 'No internet service'])
streaming_movies = st.selectbox("Streaming Movies", ['Yes', 'No', 'No internet service'])
contract = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
paperless_billing = st.selectbox("Paperless Billing", ['Yes', 'No'])
payment_method = st.selectbox("Payment Method", ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
monthly_charges = st.number_input("Monthly Charges", min_value=0.0)
total_charges = st.number_input("Total Charges", min_value=0.0)
tenure_group = st.selectbox("Tenure Group", ['1 - 12', '13 - 24', '25 - 36', '37 - 48', '49 - 60', '61 - 72'])

# Encoding user inputs
def encode_input(gender, senior_citizen, partner, dependents, tenure_group, phone_service, multiple_lines, internet_service,
                 online_security, online_backup, device_protection, tech_support, streaming_tv, streaming_movies,
                 contract, paperless_billing, payment_method, monthly_charges, total_charges):

    gender = 0 if gender == 'Female' else 1
    senior_citizen = 0 if senior_citizen == 'No' else 1
    partner = 0 if partner == 'No' else 1
    dependents = 0 if dependents == 'No' else 1
    phone_service = 0 if phone_service == 'No' else 1
    multiple_lines = ['No', 'No phone service', 'Yes'].index(multiple_lines)
    internet_service = ['No', 'DSL', 'Fiber optic'].index(internet_service)
    online_security = ['No', 'No internet service', 'Yes'].index(online_security)
    online_backup = ['No', 'No internet service', 'Yes'].index(online_backup)
    device_protection = ['No', 'No internet service', 'Yes'].index(device_protection)
    tech_support = ['No', 'No internet service', 'Yes'].index(tech_support)
    streaming_tv = ['No', 'No internet service', 'Yes'].index(streaming_tv)
    streaming_movies = ['No', 'No internet service', 'Yes'].index(streaming_movies)
    contract = ['Month-to-month', 'One year', 'Two year'].index(contract)
    paperless_billing = 0 if paperless_billing == 'No' else 1
    payment_method = ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'].index(payment_method)
    tenure_group = ['1 - 12', '13 - 24', '25 - 36', '37 - 48', '49 - 60', '61 - 72'].index(tenure_group)

    return {
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'tenure_group': tenure_group,
    }

# Button for prediction
if st.button("Predict"):
    input_data = encode_input(gender, senior_citizen, partner, dependents, tenure_group, phone_service, 
                              multiple_lines, internet_service, online_security, online_backup, device_protection,
                              tech_support, streaming_tv, streaming_movies, contract, paperless_billing, 
                              payment_method, monthly_charges, total_charges)
    result = predict_churn(input_data)
    st.success(f"The customer is predicted to: {result}")
