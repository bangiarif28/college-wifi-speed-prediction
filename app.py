import streamlit as st
import pandas as pd
import joblib

model = joblib.load("wifi_speed_model.pkl")

st.set_page_config(
    page_title="College Wi-Fi Speed Predictor",
    page_icon="📶",
    layout="centered"
)

st.title("📶 College Wi-Fi Speed Predictor")
st.write("Predict Wi-Fi download speed using Machine Learning.")

st.subheader("Enter Network Details")

users = st.number_input(
    "Connected Users", min_value=1, max_value=1000, value=25
)

devices = st.number_input(
    "Active Devices", min_value=1, max_value=2000, value=35
)

hour = st.slider("Time of Day", 0, 23, 12)

day_names = [
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday"
]
day = st.selectbox("Day of Week", day_names)

signal = st.slider("Wi-Fi Signal Strength (dBm)", -90, -30, -55)

bandwidth = st.selectbox(
    "Available Bandwidth (Mbps)",
    [100, 200, 300, 500],
    index=1
)

if st.button("Predict Wi-Fi Speed"):
    input_data = pd.DataFrame([{
        "Connected_Users": users,
        "Active_Devices": devices,
        "Time_of_Day_Hour": hour,
        "Day_of_Week": day_names.index(day),
        "Signal_Strength_dBm": signal,
        "Available_Bandwidth_Mbps": bandwidth
    }])

    prediction = float(model.predict(input_data)[0])

    if prediction < 0:
        st.warning("The model returned an invalid negative speed.")
    else:
        st.metric("Predicted Download Speed", f"{prediction:.2f} Mbps")
        st.caption(
            "This is an ML estimate, not a live Wi-Fi speed test."
        )
