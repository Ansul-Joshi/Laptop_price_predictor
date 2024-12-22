import pandas as pd
import numpy as np
import streamlit as st
import pickle

# Load the model and dataframe
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

# App title with color
st.title("Laptop Price Predictor")
st.markdown("""
Welcome to the **Laptop Price Prediction** tool! 
Fill in the details of your laptop configuration, and we will predict its price based on the given parameters. 
Get an estimate of your laptop's worth based on the specifications you choose.
""", unsafe_allow_html=True)

# Use columns to organize inputs better
col1, col2 = st.columns([3, 2])

with col1:
    company = st.selectbox('Brand', df['Company'].unique())
    type = st.selectbox('Type', df['TypeName'].unique())
    ram = st.selectbox('RAM (in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])
    weight = st.number_input('Weight of the Laptop (kg)', min_value=0.1, value=1.5, step=0.1)
    touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])
    ips = st.selectbox('IPS', ['No', 'Yes'])

with col2:
    screen_size = st.slider('Screen size (in inches)', 10.0, 18.0, 13.0)
    resolution = st.selectbox('Screen Resolution',
                              ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800', '2560x1600',
                               '2560x1440', '2304x1440'])
    cpu = st.selectbox('CPU', df['CpuBrand'].unique())
    hdd = st.selectbox('HDD (in GB)', [0, 128, 256, 512, 1024, 2048])
    ssd = st.selectbox('SSD (in GB)', [0, 8, 128, 256, 512, 1024])
    gpu = st.selectbox('GPU', df['GpuBrand'].unique())
    os = st.selectbox('OS', df['os'].unique())

# Add descriptions to some fields with color
st.markdown("""
#### **Quick Descriptions:**
- **RAM (Random Access Memory):** Determines how fast the system can handle multiple tasks.
- **HDD (Hard Disk Drive):** Traditional storage option for laptops.
- **SSD (Solid-State Drive):** Faster storage option than HDD.
- **Touchscreen & IPS:** Provide a better display and touch experience.
""", unsafe_allow_html=True)

# Stylish prediction button with interactivity
st.markdown("<hr>", unsafe_allow_html=True)  # Add a horizontal line to separate sections

if st.button('Predict Laptop Price'):
    # Preprocess the inputs
    touchscreen = 1 if touchscreen == 'Yes' else 0
    Ips = 1 if ips == 'Yes' else 0
    X_res, Y_res = map(int, resolution.split('x'))
    ppi = np.sqrt(X_res ** 2 + Y_res ** 2) / screen_size

    # Create query with correct column names
    query = pd.DataFrame([{
        'Company': company,
        'TypeName': type,
        'CpuBrand': cpu,
        'GpuBrand': gpu,
        'os': os,
        'Ram': ram,
        'Weight': weight,
        'Touchscreen': touchscreen,
        'Ips': Ips,
        'ScreenSize': screen_size,
        'Resolution': resolution,
        'HDD': hdd,
        'SSD': ssd,
        'ppi': ppi
    }])

    # Predict using the pipeline
    predicted_price = np.exp(pipe.predict(query)[0])

    # Show the prediction with rupees symbol and color
    st.markdown(f"""
    <h2 style='text-align: center; color: #FF6347;'>💻 **Predicted Price: ₹{int(predicted_price):,}** 💰</h2>
    <p style='text-align: center; font-size: 18px; color: #696969;'>This is an estimate based on historical data of similar laptops. Actual prices may vary based on market conditions.</p>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)  # Add another horizontal line for aesthetics
    st.markdown("""
    **Note:** The prediction is based on your selected laptop configuration. You can adjust the inputs to see how different components affect the price.
    """, unsafe_allow_html=True)

# Add additional spacing between sections for a cleaner look
st.markdown("<br>", unsafe_allow_html=True)
