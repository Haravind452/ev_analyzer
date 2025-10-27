import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv("electric_vehicles_spec_2025.csv")

# Train model
X = data[['battery_capacity_kWh']]
y = data['range_km']
model = LinearRegression()
model.fit(X, y)

# Streamlit app
st.title("🚗 EV Range Predictor App")
st.write("Predict the driving range of an EV based on its battery capacity.")

# User input
battery_capacity = st.number_input("Enter Battery Capacity (kWh):", min_value=1.0, max_value=200.0, value=50.0)
if st.button("Predict Range"):
    prediction = model.predict([[battery_capacity]])[0]
    st.success(f"Estimated Range: {prediction:.2f} km")

# Plot data and regression line
fig, ax = plt.subplots()
ax.scatter(data['battery_capacity_kWh'], data['range_km'], label='Data Points')
ax.plot(data['battery_capacity_kWh'], model.predict(X), color='red', label='Regression Line')
ax.set_xlabel("Battery Capacity (kWh)")
ax.set_ylabel("Range (km)")
ax.legend()
st.pyplot(fig)
