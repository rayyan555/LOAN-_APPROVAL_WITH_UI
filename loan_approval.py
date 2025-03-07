import streamlit as st
import pickle
import numpy as np

# Load the saved Naïve Bayes model
with open("naive_bayes_model.pkl", "rb") as file:
    nb_clf = pickle.load(file)

# Function to make predictions with conditions
def predict_loan_approval(features):
    credit_score, income, loan_amount, debt_ratio, loan_term, employment_length, home_ownership = features

    # Custom rejection conditions
    if credit_score < 600:
        return "Rejected (Low Credit Score)"
    if debt_ratio > 50:
        return "Rejected (High Debt-to-Income Ratio)"
    if loan_amount > income * 5:
        return "Rejected (Loan Amount Too High)"
    if employment_length < 1:
        return "Rejected (Insufficient Work Experience)"
    
    # Model-based prediction
    prediction = nb_clf.predict([features])
    return "Approved" if prediction[0] == 1 else "Rejected"

# Streamlit UI
st.title("Loan Approval Prediction")
st.write("Enter the loan details to check approval status.")

# User input fields
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=600)
income = st.number_input("Applicant Income (in $)", min_value=1000, value=5000)
loan_amount = st.number_input("Loan Amount (in $)", min_value=1000, value=20000)
debt_ratio = st.slider("Debt-to-Income Ratio (%)", min_value=0, max_value=100, value=30)
loan_term = st.selectbox("Loan Term (in months)", [12, 24, 36, 48, 60], index=2)
employment_length = st.number_input("Employment Length (years)", min_value=0, max_value=40, value=5)
home_ownership = st.selectbox("Home Ownership", ["Rent", "Own", "Mortgage"], index=0)

# Convert categorical features to numerical (if required)
home_ownership_mapping = {"Rent": 0, "Own": 1, "Mortgage": 2}
home_ownership = home_ownership_mapping[home_ownership]

# Convert inputs to model format
features = np.array([credit_score, income, loan_amount, debt_ratio, loan_term, employment_length, home_ownership])

# Predict button
if st.button("Predict Loan Approval"):
    result = predict_loan_approval(features)
    st.success(f"Loan Application Status: {result}")
