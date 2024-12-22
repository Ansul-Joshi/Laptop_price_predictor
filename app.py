import pandas as pd
import numpy as np
import streamlit as st
import pickle

# Load the model and dataframe
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

st.title("Laptop Predictor")

# brand
company = st.selectbox('Brand', df['Company'].unique())

# type of laptop
type = st.selectbox('Type', df['TypeName'].unique())

# RAM
ram = st.selectbox('RAM(in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])

# weight
weight = st.number_input('Weight of the Laptop')

# Touchscreen
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])

# IPS
ips = st.selectbox('IPS', ['No', 'Yes'])

# screen size
screen_size = st.slider('Screen size in inches', 10.0, 18.0, 13.0)

# resolution
resolution = st.selectbox('Screen Resolution', ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'])

# cpu
cpu = st.selectbox('CPU', df['CpuBrand'].unique())

hdd = st.selectbox('HDD(in GB)', [0, 128, 256, 512, 1024, 2048])

ssd = st.selectbox('SSD(in GB)', [0, 8, 128, 256, 512, 1024])

gpu = st.selectbox('GPU', df['GpuBrand'].unique())

os = st.selectbox('OS', df['os'].unique())

if st.button('Predict Price'):
    # Preprocess the inputs
    touchscreen = 1 if touchscreen == 'Yes' else 0
    Ips = 1 if ips == 'Yes' else 0
    X_res, Y_res = map(int, resolution.split('x'))
    ppi = np.sqrt(X_res**2 + Y_res**2) / screen_size

    # Create query with correct column names, including missing ones
    query = pd.DataFrame([{
        'Company': company,
        'TypeName': type,
        'CpuBrand': cpu,
        'GpuBrand': gpu,
        'os': os,
        'Ram': ram,  # Fixed column name
        'Weight': weight,  # Fixed column name
        'Touchscreen': touchscreen,  # Fixed column name
        'Ips': Ips,  # Fixed column name
        'ScreenSize': screen_size,  # Fixed column name
        'Resolution': resolution,  # Fixed column name
        'HDD': hdd,  # Fixed column name
        'SSD': ssd,  # Fixed column name
        'ppi': ppi  # Fixed column name
    }])

    # Predict using the pipeline
    st.title("The predicted price of this configuration is " + str(int(np.exp(pipe.predict(query)[0]))))
