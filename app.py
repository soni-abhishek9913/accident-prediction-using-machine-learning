import streamlit as st
import joblib
import numpy as np

st.title("🚦 Accident Severity Prediction")


model = joblib.load('accident_model.pkl')


temp = st.slider("Temperature (°F)", 0, 120, 60)
humidity = st.slider("Humidity (%)", 0, 100, 50)
pressure = st.slider("Pressure (in)", 25, 35, 30)
visibility = st.slider("Visibility (mi)", 0, 20, 10)
wind = st.slider("Wind Speed (mph)", 0, 50, 10)
weather = st.number_input("Weather Condition Code", 0, 100, 5)
sunrise_sunset = st.radio("Day or Night", ["Day", "Night"])
hour = st.slider("Hour of Day", 0, 23, 12)


input_data = np.array([[temp, humidity, pressure, visibility, wind,
                        weather, 1 if sunrise_sunset == 'Day' else 0, hour]])


prediction = model.predict(input_data)
st.success(f"Predicted Severity Level: {prediction[0]}")