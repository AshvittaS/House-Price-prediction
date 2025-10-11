import streamlit as st
import pandas as pd
import numpy as np
import pickle
# Load model
with open('housing_price_model.pkl', 'rb') as f:
    model = pickle.load(f)
st.set_page_config(page_title="House Price Predictor", layout="wide")
# Helper function to create colored sections
def colored_section(title, color, content_func):
    st.markdown(f"""
    <div style="background-color:{color}; padding:25px; border-radius:10px; margin-bottom:20px">
    <h2 style="color:white;">{title}</h2>
    </div>
    """, unsafe_allow_html=True)
    
    with st.container():
        content_func()
# HEADER SECTION
def header_content():
    st.write("Predict your house price by entering all the features below.")

colored_section("🏠 House Price Prediction", "#f06292", header_content)  # Pink background
# INPUT SECTION
def input_content():
    st.header("📝 Enter House Details")

    col1, col2, col3 = st.columns(3)
    with col1:
        area = st.number_input("Area (in sqft)", min_value=100, max_value=10000, value=500)
        bedrooms = st.slider("Bedrooms", 1, 10, 1)
        bathrooms = st.slider("Bathrooms", 1, 10, 1)

    with col2:
        stories = st.slider("Stories", 1, 5, 1)
        parking = st.slider("Parking Spots", 0, 5, 1)
        mainroad = st.selectbox("Main Road", ['yes', 'no'])

    with col3:
        guestroom = st.selectbox("Guest Room", ['yes', 'no'])
        basement = st.selectbox("Basement", ['yes', 'no'])
        airconditioning = st.selectbox("Air Conditioning", ['yes', 'no'])

    with st.expander("Additional Features"):
        hotwaterheating = st.selectbox("Hot Water Heating", ['yes', 'no'])
        prefarea = st.selectbox("Preferred Area", ['yes', 'no'])
        furnishingstatus = st.selectbox("Furnishing Status", ['furnished','semi-furnished','unfurnished'])
    
    # Save inputs to session_state for later
    st.session_state.inputs = {
        "area": area, "bedrooms": bedrooms, "bathrooms": bathrooms, "stories": stories,
        "parking": parking, "mainroad": mainroad, "guestroom": guestroom,
        "basement": basement, "airconditioning": airconditioning,
        "hotwaterheating": hotwaterheating, "prefarea": prefarea,
        "furnishingstatus": furnishingstatus
    }

colored_section("📝 House Details", "#64b5f6", input_content)  # Blue background

# -----------------------------
# PREDICTION SECTION
# -----------------------------
def prediction_content():
    if st.button("Predict Price"):
        data = st.session_state.inputs
        bin_map = {'yes':1, 'no':0}
        mainroad = bin_map[data["mainroad"]]
        guestroom = bin_map[data["guestroom"]]
        basement = bin_map[data["basement"]]
        hotwaterheating = bin_map[data["hotwaterheating"]]
        airconditioning = bin_map[data["airconditioning"]]
        prefarea = bin_map[data["prefarea"]]

        furnishingstatus_semi = 1 if data["furnishingstatus"]=='semi-furnished' else 0
        furnishingstatus_unf = 1 if data["furnishingstatus"]=='unfurnished' else 0

        area_transformed = np.log1p(data["area"])

        input_df = pd.DataFrame({
            'area':[area_transformed],
            'bedrooms':[data["bedrooms"]],
            'bathrooms':[data["bathrooms"]],
            'stories':[data["stories"]],
            'mainroad':[mainroad],
            'guestroom':[guestroom],
            'basement':[basement],
            'hotwaterheating':[hotwaterheating],
            'airconditioning':[airconditioning],
            'parking':[data["parking"]],
            'prefarea':[prefarea],
            'furnishingstatus_semi-furnished':[furnishingstatus_semi],
            'furnishingstatus_unfurnished':[furnishingstatus_unf]
        })

        prediction_log = model.predict(input_df)
        prediction = np.expm1(prediction_log)

        st.markdown(f"""
        <div style='background-color:#81c784; padding:20px; border-radius:10px'>
        <h2 style='color:white;'>💰 Predicted House Price: ₹ {prediction[0]:,.2f}</h2>
        </div>
        """, unsafe_allow_html=True)

colored_section("💰 Prediction", "#388e3c", prediction_content)  # Dark Green background
