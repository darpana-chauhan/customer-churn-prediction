import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# =====================================
# Page Configuration
# =====================================
st.set_page_config(
    page_title="Customer Churn Prediction App",
    page_icon="📊",
    layout="centered"
)

# =====================================
# Custom CSS (Pointer cursor on dropdowns & buttons)
# =====================================
st.markdown("""
<style>
div[data-baseweb="select"] > div {
    cursor: pointer;
}
button {
    cursor: pointer !important;
}
</style>
""", unsafe_allow_html=True)

# =====================================
# Load Model
# =====================================
model_path = "churn_model.pkl"
with open(model_path, "rb") as file:
    model = pickle.load(file)

# =====================================
# Load Cleaned Dataset (for column reference)
# =====================================
data_path = "cleaned_telco_churn.csv"
df = pd.read_csv(data_path)

# =====================================
# App Title
# =====================================
st.title("📉 Customer Churn Prediction")
st.markdown(
    """
    This application predicts whether a telecom customer is **likely to churn**
    based on their service usage and account details.
    """
)

st.divider()

# =====================================
# Sidebar Inputs
# =====================================
st.sidebar.header("🔧 Customer Details")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.sidebar.selectbox("Has Partner?", ["Yes", "No"])
dependents = st.sidebar.selectbox("Has Dependents?", ["Yes", "No"])

tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
monthly_charges = st.sidebar.slider("Monthly Charges", 18.0, 120.0, 70.0)
total_charges = st.sidebar.slider("Total Charges", 0.0, 9000.0, 1500.0)

contract = st.sidebar.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)

internet_service = st.sidebar.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

payment_method = st.sidebar.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

paperless = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])

# =====================================
# Encode Inputs (same as training)
# =====================================
input_data = {
    "tenure": tenure,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
    "SeniorCitizen": 1 if senior == "Yes" else 0,

    "gender_Male": 1 if gender == "Male" else 0,
    "Partner_Yes": 1 if partner == "Yes" else 0,
    "Dependents_Yes": 1 if dependents == "Yes" else 0,
    "PaperlessBilling_Yes": 1 if paperless == "Yes" else 0,

    "Contract_One year": 1 if contract == "One year" else 0,
    "Contract_Two year": 1 if contract == "Two year" else 0,

    "InternetService_Fiber optic": 1 if internet_service == "Fiber optic" else 0,
    "InternetService_No": 1 if internet_service == "No" else 0,

    "PaymentMethod_Electronic check": 1 if payment_method == "Electronic check" else 0,
    "PaymentMethod_Mailed check": 1 if payment_method == "Mailed check" else 0,
    "PaymentMethod_Bank transfer (automatic)": 1 if payment_method == "Bank transfer (automatic)" else 0,
    "PaymentMethod_Credit card (automatic)": 1 if payment_method == "Credit card (automatic)" else 0,
}

input_df = pd.DataFrame([input_data])

# Ensure column order matches training
for col in model.feature_names_in_:
    if col not in input_df.columns:
        input_df[col] = 0

input_df = input_df[model.feature_names_in_]

# =====================================
# Prediction Section
# =====================================
st.divider()

if st.button("🔍 Predict Churn"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]

    churn_prob = probability[1]
    stay_prob = probability[0]

    # Result Message
    if prediction == 1:
        st.error(f"⚠️ Customer is likely to **CHURN**\n\nProbability: {churn_prob:.2%}")
    else:
        st.success(f"✅ Customer is likely to **STAY**\n\nProbability: {stay_prob:.2%}")

    # =====================================
    # Probability Visualization
    # =====================================
    st.subheader("📊 Prediction Probability")

    fig, ax = plt.subplots()
    ax.bar(
        ["Stay", "Churn"],
        [stay_prob, churn_prob]
    )

    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)
    ax.set_title("Customer Churn Prediction Probability")

    st.pyplot(fig)

# =====================================
# Footer
# =====================================
st.divider()
st.caption("📘 Built using Machine Learning & Streamlit | Customer Churn Prediction Project")

