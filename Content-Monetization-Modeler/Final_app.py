import streamlit as st
import joblib
import pandas as pd

# Load Model and Encoders
model = joblib.load('youtube_revenue_model.pkl')
le_category = joblib.load('category_encoder.pkl')
le_device = joblib.load('device_encoder.pkl')
le_country = joblib.load('country_encoder.pkl')


st.sidebar.header("📌 Project Information")
st.sidebar.info("""
Model Used: Random Forest Regressor

R² Score: 0.9485

MAE: 3.67

RMSE: 14.05
""")

# Title
st.title("YouTube Ad Revenue Predictor")

st.write("Predict YouTube Ad Revenue using Machine Learning")

# User Inputs
views = st.number_input("Views", min_value=0)

likes = st.number_input("Likes", min_value=0)

comments = st.number_input("Comments", min_value=0)

watch_time_minutes = st.number_input(
    "Watch Time Minutes",
    min_value=0.0
)

video_length_minutes = st.number_input(
    "Video Length Minutes",
    min_value=0.0
)

subscribers = st.number_input(
    "Subscribers",
    min_value=0
)

# Dropdown Inputs
category = st.selectbox(
    "Category",
    list(le_category.classes_)
)

device = st.selectbox(
    "Device",
    list(le_device.classes_)
)

country = st.selectbox(
    "Country",
    list(le_country.classes_)
)

# Predict Button
if st.button("Predict Revenue"):

    # Encode Inputs
    category_encoded = le_category.transform([category])[0]
    device_encoded = le_device.transform([device])[0]
    country_encoded = le_country.transform([country])[0]

    # Feature Engineering
    if views == 0:
        engagement_rate = 0
    else:
        engagement_rate = (likes + comments) / views

    # Create DataFrame
    input_data = pd.DataFrame([[
        views,
        likes,
        comments,
        watch_time_minutes,
        video_length_minutes,
        subscribers,
        category_encoded,
        device_encoded,
        country_encoded,
        engagement_rate
    ]],
    columns=[
        'views',
        'likes',
        'comments',
        'watch_time_minutes',
        'video_length_minutes',
        'subscribers',
        'category',
        'device',
        'country',
        'engagement_rate'
    ])

    # Prediction
    if views <= 0 or likes <= 0  or subscribers <= 0:
        st.error("Please enter values greater than 0 for all fields.")
    else:
        prediction = model.predict(input_data)
        st.success(f"Predicted Revenue: ${prediction[0]:.2f}")

    
