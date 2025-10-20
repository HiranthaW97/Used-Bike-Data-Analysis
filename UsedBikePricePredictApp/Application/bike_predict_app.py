import streamlit as st
import requests
from config import BASE_URL

# Page Config
st.set_page_config(page_title="Bike Price Prediction", layout="wide")

# App Title
st.title("Market Value Predictor for Used Bikes")

# Tabs
home_tab, predict_tab = st.tabs(["Home", "Predict Price"])

# --- Home Tab ---
with home_tab:
    st.write("""
        This application allows you to estimate the current market price of a used motorbike in Sri Lanka based on real-world and synthetic data from 2022 to 2025.
    """)
    
    st.image("Images/bikes.jpg", use_container_width=True)
    
# --- Predict Tab ---
with predict_tab:
    st.header("Predict Bike Price")

    with st.form("predict_form"):
        col1, col2 = st.columns(2)

        with col1:
            bike_type = st.selectbox("Bike Type", ["Motorbikes", "Scooters", "E-bikes"])
            brand = st.text_input("Brand", value="KTM")
            edition = st.text_input("Edition", value="Standard")
            model = st.text_input("Model", value="Duke 790")

        with col2:
            year = st.number_input("Registered Year", min_value=1990, max_value=2025, value=2023)
            mileage = st.number_input("Mileage (km)", min_value=0, value=26955)
            capacity = st.number_input("Engine Capacity (cc)", min_value=45, value=799)

        submitted = st.form_submit_button("Predict Price")

    if submitted:
        with st.spinner("Predicting price..."):
            input_data = {
                "Bike Type": bike_type,
                "Brand": brand,
                "Edition": edition,
                "Model": model,
                "Year": int(year),
                "Mileage": int(mileage),
                "Capacity": int(capacity)
            }

            try:
                response = requests.post(f"{BASE_URL}/predict", json=input_data)
                if response.status_code == 200:
                    result = response.json()
                    predicted_price = result.get("predicted_price")

                    if predicted_price is not None:
                        st.success(f"**Predicted Price:** Rs {predicted_price:,.2f}")
                    else:
                        st.warning("Prediction response did not contain a valid price.")
                else:
                    st.error(f"API error: {response.status_code} - {response.text}")

            except Exception as e:
                st.error(f"Unexpected error occurred: {str(e)}")
