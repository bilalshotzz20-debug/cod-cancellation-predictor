import streamlit as st
import pandas as pd
import joblib

# 1. Load the saved model and columns
model = joblib.load('cancellation_model.pkl')
model_columns = joblib.load('model_columns.pkl')

# 2. Website Title
st.set_page_config(page_title="COD Cancellation Predictor", page_icon="📦")
st.title("📦 COD Cancellation Predictor")
st.write("Enter order details below to predict if the order will be cancelled.")

# 3. User Inputs
city = st.selectbox("City", ["Karachi", "Lahore", "Islamabad", "Rawalpindi", "Kahuta", "Faisalabad", "Multan", "Peshawar"])
order_value = st.number_input("Order Value (PKR)", min_value=500, max_value=15000, value=2500)
customer_age_days = st.number_input("Customer Age (Days)", min_value=1, max_value=365, value=30)
num_items = st.number_input("Number of Items", min_value=1, max_value=5, value=1)
delivery_days = st.number_input("Expected Delivery Days", min_value=1, max_value=10, value=3)

# 4. Predict Button
if st.button("Predict Cancellation Risk"):
    # Create a dictionary with the inputs
    input_data = {
        "order_value": [order_value],
        "customer_age_days": [customer_age_days],
        "num_items": [num_items],
        "delivery_days": [delivery_days]
    }
    
    # Convert to DataFrame
    input_df = pd.DataFrame(input_data)
    
    # Add the city dummy variables (set all to 0 first)
    for col in model_columns:
        if col.startswith("city_"):
            input_df[col] = 0
            
    # Set the selected city to 1
    selected_city_col = f"city_{city}"
    if selected_city_col in input_df.columns:
        input_df[selected_city_col] = 1
        
    # Ensure columns are in the exact order the model expects
    input_df = input_df[model_columns]
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1] * 100
    
    # Show Result
    st.divider()
    if prediction == 1:
        st.error(f"⚠️ HIGH RISK: This order is likely to be CANCELLED. (Risk: {probability:.1f}%)")
    else:
        st.success(f"✅ LOW RISK: This order is likely to be DELIVERED. (Risk: {probability:.1f}%)")