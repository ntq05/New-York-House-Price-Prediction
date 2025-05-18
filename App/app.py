import streamlit as st
import pandas as pd
from Model_Pipeline import predict_price
import re

st.title("NY House Price Prediction")

df = pd.read_csv("..\\Datasets\\Transformed_Training_set.csv")

beds = st.number_input("Beds", min_value=0, step=1)
bath = st.number_input("Bath", min_value=0, step=1)
propertysqft = st.number_input("Property Sqft", min_value=0)
latitude = st.number_input("Latitude", format="%.6f")
longitude = st.number_input("Longitude", format="%.6f")

if st.button("Predict"):

    input_dict = {
        'beds': [beds],
        'bath': [bath],
        'propertysqft': [propertysqft],
        'latitude': [latitude],
        'longitude': [longitude]
    }
    
    input_df = pd.DataFrame(input_dict)

    price = predict_price(input_df)
    st.success(f"Predicted Price: ${price[0]:,.2f}")