import streamlit as st
import pickle
import numpy as np

# Load the model and dataset
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

st.title("Laptop Price Predictor")

# Brand
company = st.selectbox('Brand', df['Company'].unique())

# Type of laptop
laptop_type = st.selectbox('Type', df['TypeName'].unique())

# RAM
ram = st.selectbox('RAM (in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])

# Weight
weight = st.number_input('Weight of the laptop (in kg)', min_value=1.2)

# Touchscreen
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])

# IPS
ips = st.selectbox('IPS', ['No', 'Yes'])

# Screen size
screen_size = st.number_input('Screen Size (in inches)', min_value=13.0)

# Screen resolution
resolution = st.selectbox('Screen Resolution', [
    '1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800',
    '2880x1800', '2560x1600', '2560x1440', '2304x1440'
])

# CPU
cpu = st.selectbox('CPU', df['Cpu brand'].unique())

# Hard drive
hdd = st.selectbox('HDD (in GB)', [0, 128, 256, 512, 1024, 2048])

# SSD
ssd = st.selectbox('SSD (in GB)', [0, 128, 256, 512, 1024])

# GPU
gpu = st.selectbox('GPU', df['Gpu brand'].unique())

# Operating System
os = st.selectbox('OS', df['os'].unique())

# Prediction logic
if st.button('Predict Price'):

    # Touchscreen and IPS encoding
    touchscreen = 1 if touchscreen == 'Yes' else 0
    ips = 1 if ips == 'Yes' else 0

    # Calculate PPI
    try:
        X_res, Y_res = map(int, resolution.split('x'))
        ppi = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / screen_size
    except ZeroDivisionError:
        st.error("Screen size must be greater than 0.")
        ppi = 0

    # Query for prediction
    query = np.array([company, laptop_type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os])

    # Reshape the query to match the model input
    query = query.reshape(1, 12)

    # Predict the price
    try:
        prediction = np.exp(pipe.predict(query).item())
        st.title(f"Predicted Price: {prediction:.2f}")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
