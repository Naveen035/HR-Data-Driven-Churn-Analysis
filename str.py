import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Load the model
with open('BestFitmodel.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Streamlit app
st.title("Hr Data Driven Churn Analysis")

st.sidebar.header("Input Customer Details")

def user_input_features():
    gender = st.sidebar.selectbox("Gender", ("Male", "Female"))
    senior_citizen = st.sidebar.selectbox("Senior Citizen", (0, 1))
    partner = st.sidebar.selectbox("Partner", ("Yes", "No"))
    dependents = st.sidebar.selectbox("Dependents", ("Yes", "No"))
    phone_service = st.sidebar.selectbox("Phone Service", ("Yes", "No"))
    multiple_lines = st.sidebar.selectbox("Multiple Lines", ("Yes", "No", "No phone service"))
    internet_service = st.sidebar.selectbox("Internet Service", ("DSL", "Fiber optic", "No"))
    online_security = st.sidebar.selectbox("Online Security", ("Yes", "No", "No internet service"))
    online_backup = st.sidebar.selectbox("Online Backup", ("Yes", "No", "No internet service"))
    device_protection = st.sidebar.selectbox("Device Protection", ("Yes", "No", "No internet service"))
    tech_support = st.sidebar.selectbox("Tech Support", ("Yes", "No", "No internet service"))
    streaming_tv = st.sidebar.selectbox("Streaming TV", ("Yes", "No", "No internet service"))
    streaming_movies = st.sidebar.selectbox("Streaming Movies", ("Yes", "No", "No internet service"))
    contract = st.sidebar.selectbox("Contract", ("Month-to-month", "One year", "Two year"))
    paperless_billing = st.sidebar.selectbox("Paperless Billing", ("Yes", "No"))
    payment_method = st.sidebar.selectbox("Payment Method", ("Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"))
    monthly_charges = st.sidebar.number_input("Monthly Charges", min_value=0.0, step=0.1)
    total_charges = st.sidebar.number_input("Total Charges", min_value=0.0, step=0.1)
    tenure_group = st.sidebar.selectbox("Tenure Group", ['1 - 12', '13 - 24', '25 - 36', '37 - 48', '49 - 60', '61 - 72'])

    data = {
        'gender': 0 if gender == "Female" else 1,
        'SeniorCitizen': senior_citizen,
        'Partner': 0 if partner == "No" else 1,
        'Dependents': 0 if dependents == "No" else 1,
        'PhoneService': 0 if phone_service == "No" else 1,
        'MultipleLines': ['No', 'Yes', 'No phone service'].index(multiple_lines),
        'InternetService': ['DSL', 'Fiber optic', 'No'].index(internet_service),
        'OnlineSecurity': ['No', 'Yes', 'No internet service'].index(online_security),
        'OnlineBackup': ['No', 'Yes', 'No internet service'].index(online_backup),
        'DeviceProtection': ['No', 'Yes', 'No internet service'].index(device_protection),
        'TechSupport': ['No', 'Yes', 'No internet service'].index(tech_support),
        'StreamingTV': ['No', 'Yes', 'No internet service'].index(streaming_tv),
        'StreamingMovies': ['No', 'Yes', 'No internet service'].index(streaming_movies),
        'Contract': ['Month-to-month', 'One year', 'Two year'].index(contract),
        'PaperlessBilling': 0 if paperless_billing == "No" else 1,
        'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'].index(payment_method),
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'tenure_group': ['1 - 12', '13 - 24', '25 - 36', '37 - 48', '49 - 60', '61 - 72'].index(tenure_group),
    }

    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Display user input
st.subheader("Customer Details")
st.write(input_df)

# Make prediction
st.write('Predict Wheather they will stay or leave')
if st.button("Predict"):
    prediction = loaded_model.predict(input_df)
    st.write("Leave" if prediction[0] == 1 else "Stay")

# Display ROC and accuracies
st.subheader("Model Evaluation Metrics")

# Model accuracies
accuracies = {
    "DecisionTree": 0.78,
    "LogisticRegression": 0.80,
    "RandomForest": 0.82,
    "KNN": 0.77,
    "AdaBoost": 0.84,
    "XGBoost": 0.85,
    "GradientBoosting": 0.87
}

# Display accuracies
for model, acc in accuracies.items():
    st.write(f"{model}: {acc * 100:.2f}%")

# Display ROC Curve
def plot_roc_curve(fpr, tpr, model_name):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'{model_name} ROC Curve')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} ROC Curve')
    plt.legend()
    st.pyplot(plt)

# Example ROC curves for display
# You can modify these with the actual fpr, tpr values from your training phase
fpr_example = [0.0, 0.1, 0.4, 0.9, 1.0]
tpr_example = [0.0, 0.5, 0.7, 0.9, 1.0]
plot_roc_curve(fpr_example, tpr_example, "GradientBoosting")
