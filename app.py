import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model and scaler
with open('housing_price_model.pkl','rb') as f:         
    model = pickle.load(f)
# Streamlit app
st.title("House Price Prediction")
st.write("Enter house details below:")
# Inputs
area = st.number_input("Area (sq ft)", min_value=100.0, step=10.0)
bedrooms = st.number_input("Number of Bedrooms", min_value=1, step=1)
bathrooms = st.number_input("Number of Bathrooms", min_value=1, step=1)
stories = st.number_input("Number of Stories", min_value=1, step=1)
mainroad = st.selectbox("Main Road", ['yes', 'no'])
guestroom = st.selectbox("Guest Room", ['yes', 'no'])
basement = st.selectbox("Basement", ['yes', 'no'])
hotwaterheating = st.selectbox("Hot Water Heating", ['yes', 'no'])
airconditioning = st.selectbox("Air Conditioning", ['yes', 'no'])
parking = st.number_input("Parking Spots", min_value=0, step=1)
prefarea = st.selectbox("Preferred Area", ['yes', 'no'])
furnishingstatus = st.selectbox("Furnishing Status", ['furnished', 'semi-furnished', 'unfurnished'])

# Encode binary variables
bin_map = {'yes': 1, 'no': 0}
mainroad = bin_map[mainroad]
guestroom = bin_map[guestroom]
basement = bin_map[basement]
hotwaterheating = bin_map[hotwaterheating]
airconditioning = bin_map[airconditioning]
prefarea = bin_map[prefarea]
# One-hot encode furnishing status (drop first)
furnishingstatus_semi = 1 if furnishingstatus == 'semi-furnished' else 0
furnishingstatus_unf = 1 if furnishingstatus == 'unfurnished' else 0
#log
area_transformed = np.log1p(area)
# Scale numeric features
input_df = pd.DataFrame({
    'area': [area_transformed],
    'bedrooms': [bedrooms],
    'bathrooms': [bathrooms],
    'stories': [stories],
    'mainroad': [mainroad],
    'guestroom': [guestroom],
    'basement': [basement],
    'hotwaterheating': [hotwaterheating],
    'airconditioning': [airconditioning],
    'parking': [parking],
    'prefarea': [prefarea],
    'furnishingstatus_semi-furnished': [furnishingstatus_semi],
    'furnishingstatus_unfurnished': [furnishingstatus_unf]
})

# Predict
prediction_log = model.predict(input_df)
prediction = np.expm1(prediction_log)  # reverse log1p

st.subheader("Predicted House Price:")
st.write(f"₹ {prediction[0]:,.2f}") 