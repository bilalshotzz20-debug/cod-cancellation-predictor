import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 1. Yeh function website khulte hi model ko train karega
@st.cache_resource
def train_model():
    np.random.seed(42)
    n = 1000
    cities = ["Karachi", "Lahore", "Islamabad", "Rawalpindi", "Kahuta", "Faisalabad", "Multan", "Peshawar"]
    
    data_big = {
        "order_id": [f"ORD{i:04d}" for i in range(1, n + 1)],
        "city": np.random.choice(cities, n),
        "order_value": np.random.randint(500, 15000, n),
        "customer_age_days": np.random.randint(1, 365, n),
        "num_items": np.random.randint(1, 6, n),
        "delivery_days": np.random.randint(1, 11, n)
    }
    
    df_big = pd.DataFrame(data_big)
    df_big["is_cancelled"] = 0
    
    for i in range(n):
        if df_big.loc[i, "delivery_days"] > 6:
            df_big.loc[i, "is_cancelled"] = np.random.choice([0, 1], p=[0.2, 0.8])
        elif df_big.loc[i, "delivery_days"] > 3:
            df_big.loc[i, "is_cancelled"] = np.random.choice([0, 1], p=[0.6, 0.4])
        else:
            df_big.loc[i, "is_cancelled"] = np.random.choice([0, 1], p=[0.9, 0.1])

    df_encoded = pd.get_dummies(df_big, columns=["city"], dtype=int).drop(columns=["order_id"])
    X = df_encoded.drop(columns=["is_cancelled"])
    y = df_encoded["is_cancelled"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    return model, list(X.columns)

# Model aur columns load karo
model, model_columns = train_model()

# 2. Website ka UI
st.set_page_config(page_title="COD Cancellation Predictor", page_icon="📦")
st.title("📦 COD Cancellation Predictor")
st.write("Enter order details below to predict if the order will be cancelled.")

city = st.selectbox("City", ["Karachi", "Lahore", "Islamabad", "Rawalpindi", "Kahuta", "Faisalabad", "Multan", "Peshawar"])
order_value = st.number_input("Order Value (PKR)", min_value=500, max_value=15000, value=2500)
customer_age_days = st.number_input("Customer Age (Days)", min_value=1, max_value=365, value=30)
num_items = st.number_input("Number of Items", min_value=1, max_value=5, value=1)
delivery_days = st.number_input("Expected Delivery Days", min_value=1, max_value=10, value=3)

if st.button("Predict Cancellation Risk"):
    input_data = {
        "order_value": [order_value],
        "customer_age_days": [customer_age_days],
        "num_items": [num_items],
        "delivery_days": [delivery_days]
    }
    
    input_df = pd.DataFrame(input_data)
    
    for col in model_columns:
        if col.startswith("city_"):
            input_df[col] = 0
            
    selected_city_col = f"city_{city}"
    if selected_city_col in input_df.columns:
        input_df[selected_city_col] = 1
        
    input_df = input_df[model_columns]
    
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1] * 100
    
    st.divider()
    if prediction == 1:
        st.error(f"⚠️ HIGH RISK: This order is likely to be CANCELLED. (Risk: {probability:.1f}%)")
    else:
        st.success(f"✅ LOW RISK: This order is likely to be DELIVERED. (Risk: {probability:.1f}%)")
