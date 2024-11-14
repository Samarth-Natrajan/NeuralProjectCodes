import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("churn_model.keras")

# Scaling parameters (replace these with the actual min and max values from your dataset)
tenure_min, tenure_max = 1, 72
monthly_charges_min, monthly_charges_max = 18.25, 118.75
total_charges_min, total_charges_max = 18.8, 8684.8

# Function to normalize based on min and max values
def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val)

# Define a function to make predictions
def predict_churn(input_data):
    input_data = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_data)
    if prediction[0][0] >= 0.5:
        return {"message":"🚨 Warning: Looks like they might be leaving. Let’s give them a reason to stay!","result":prediction[0][0]}
    else:
        return {"message":"😊 Great news! This customer is likely to stick around!","result":prediction[0][0]}


# Streamlit app layout
st.title("Customer Churn Prediction App")
st.write("Enter customer details to predict churn:")

# Mapping human-readable selections to model-compatible values
gender_map = {"Male": 1, "Female": 0}
senior_citizen_map = {"Yes": 1, "No": 0}
partner_map = {"Yes": 1, "No": 0}
dependents_map = {"Yes": 1, "No": 0}
phone_service_map = {"Yes": 1, "No": 0}
multi_lines_map = {"Yes": 1, "No": 0}
online_security_map = {"Yes": 1, "No": 0}
online_backup_map = {"Yes": 1, "No": 0}
device_protection_map = {"Yes": 1, "No": 0}
tech_support_map = {"Yes": 1, "No": 0}
streaming_tv_map = {"Yes": 1, "No": 0}
streaming_movies_map = {"Yes": 1, "No": 0}
paperless_billing_map = {"Yes": 1, "No": 0}

internet_service_map = {"DSL": [1, 0, 0], "Fiber optic": [0, 1, 0], "No Internet": [0, 0, 1]}
contract_map = {"Month-to-month": [1, 0, 0], "One year": [0, 1, 0], "Two year": [0, 0, 1]}
payment_method_map = {
    "Bank transfer (automatic)": [1, 0, 0, 0],
    "Credit card (automatic)": [0, 1, 0, 0],
    "Electronic check": [0, 0, 1, 0],
    "Mailed check": [0, 0, 0, 1]
}
col1,col2,col3,col4,col5 = st.columns(5)
# Gather inputs
with col1:
    gender = st.selectbox("Gender:", list(gender_map.keys()))
    senior_citizen = st.selectbox("Senior Citizen:", list(senior_citizen_map.keys()))
    partner = st.selectbox("Partner:", list(partner_map.keys()))
    dependents = st.selectbox("Dependents:", list(dependents_map.keys()))

with col2:
    tenure = st.slider("Tenure (in months):", min_value=0, max_value=72, value=36)
    monthly_charges = st.slider("Monthly Charges (in dollars):", min_value=0.0, max_value=120.0, value=50.0)
    total_charges = st.slider("Total Charges (in dollars):", min_value=0.0, max_value=9000.0, value=500.0)
with col3:
    phone_service = st.selectbox("Phone Service:", list(phone_service_map.keys()))
    multiple_lines = st.selectbox("Multiple Lines:", list(multi_lines_map.keys()))
    online_security = st.selectbox("Online Security:", list(online_security_map.keys()))
    online_backup = st.selectbox("Online Backup:", list(online_backup_map.keys()))
with col4:
    device_protection = st.selectbox("Device Protection:", list(device_protection_map.keys()))
    tech_support = st.selectbox("Tech Support:", list(tech_support_map.keys()))
    streaming_tv = st.selectbox("Streaming TV:", list(streaming_tv_map.keys()))
    streaming_movies = st.selectbox("Streaming Movies:", list(streaming_movies_map.keys()))
with col5:
    paperless_billing = st.selectbox("Paperless Billing:", list(paperless_billing_map.keys()))
    internet_service = st.selectbox("Internet Service:", list(internet_service_map.keys()))
    contract = st.selectbox("Contract:", list(contract_map.keys()))
    payment_method = st.selectbox("Payment Method:", list(payment_method_map.keys()))

# Convert inputs to model-compatible values
normalized_tenure = normalize(tenure, tenure_min, tenure_max)
normalized_monthly_charges = normalize(monthly_charges, monthly_charges_min, monthly_charges_max)
normalized_total_charges = normalize(total_charges, total_charges_min, total_charges_max)
input_data = [
    gender_map[gender],
    senior_citizen_map[senior_citizen],
    partner_map[partner],
    dependents_map[dependents],
    normalized_tenure,
    phone_service_map[phone_service],
    multi_lines_map[multiple_lines],
    online_security_map[online_security],
    online_backup_map[online_backup],
    device_protection_map[device_protection],
    tech_support_map[tech_support],
    streaming_tv_map[streaming_tv],
    streaming_movies_map[streaming_movies],
    paperless_billing_map[paperless_billing],
    normalized_monthly_charges,
    normalized_total_charges
] + internet_service_map[internet_service] + contract_map[contract] + payment_method_map[payment_method]

# Predict and display result
if st.button("Predict Churn"):
    print(normalized_monthly_charges,normalized_tenure,normalized_total_charges)
    output = predict_churn(input_data)
    st.write(f"The model predicts: {output["message"]}")
    st.write(f"The likelyhood of the customer staying: {((1-output["result"])*10000)//100} %")
