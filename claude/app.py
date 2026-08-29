import pickle
import datetime

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Seoul Bike Demand Predictor",
    page_icon="🚲",
    layout="centered",
)


@st.cache_resource
def load_artifact():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)


artifact = load_artifact()
model = artifact["model"]
categorical_options = artifact["categorical_options"]

st.title("🚲 Seoul Bike Demand Predictor")
st.write(
    "Predict the number of bikes that will be rented in a given hour, "
    "based on weather and calendar conditions. Model: Gradient Boosting Regressor."
)

st.header("Input conditions")

col1, col2 = st.columns(2)

with col1:
    date = st.date_input("Date", value=datetime.date.today())
    hour = st.slider("Hour of day", 0, 23, 12)
    temperature = st.slider("Temperature (°C)", -20.0, 40.0, 15.0)
    humidity = st.slider("Humidity (%)", 0, 100, 50)
    wind_speed = st.slider("Wind speed (m/s)", 0.0, 10.0, 2.0)
    visibility = st.slider("Visibility (10m)", 0, 2000, 1500)

with col2:
    dew_point = st.slider("Dew point temperature (°C)", -25.0, 30.0, 5.0)
    solar_radiation = st.slider("Solar Radiation (MJ/m2)", 0.0, 4.0, 0.5)
    rainfall = st.slider("Rainfall (mm)", 0.0, 35.0, 0.0)
    snowfall = st.slider("Snowfall (cm)", 0.0, 10.0, 0.0)
    seasons = st.selectbox("Season", categorical_options["Seasons"])
    holiday = st.selectbox("Holiday", categorical_options["Holiday"])
    functioning_day = st.selectbox(
        "Functioning Day", categorical_options["Functioning Day"]
    )

# Derived calendar features (must match training feature engineering)
day_of_week = date.weekday()
month = date.month
is_weekend = int(day_of_week >= 5)

input_df = pd.DataFrame(
    [
        {
            "Hour": hour,
            "Temperature(°C)": temperature,
            "Humidity(%)": humidity,
            "Wind speed (m/s)": wind_speed,
            "Visibility (10m)": visibility,
            "Dew point temperature(°C)": dew_point,
            "Solar Radiation (MJ/m2)": solar_radiation,
            "Rainfall(mm)": rainfall,
            "Snowfall (cm)": snowfall,
            "Seasons": seasons,
            "Holiday": holiday,
            "Functioning Day": functioning_day,
            "DayOfWeek": day_of_week,
            "Month": month,
            "IsWeekend": is_weekend,
        }
    ]
)

st.divider()

if st.button("Predict rented bike count", type="primary"):
    prediction = model.predict(input_df)[0]
    prediction = max(0, round(prediction))
    st.metric("Predicted Rented Bike Count", f"{prediction:,}")

with st.expander("See input data sent to the model"):
    st.dataframe(input_df)
