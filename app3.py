import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# -------------------------------
# 🚗 Load Dataset Automatically
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\EV_Range_App\electric_vehicles_spec_2025.csv.csv")
    return df

df = load_data()

# -------------------------------
# 🧭 App Title
# -------------------------------
st.title("🔋 EV Range Prediction using Machine Learning")
st.write("Predict electric vehicle range based on battery capacity using a trained Linear Regression model.")

# -------------------------------
# 📘 Dataset Preview
# -------------------------------
st.subheader("📊 Dataset Preview")
#st.dataframe(df.head())

# -------------------------------
# 🧩 Data Preparation
# -------------------------------
X = df[['battery_capacity_kWh']]
y = df['range_km']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions for evaluation
y_pred = model.predict(X_test)

# -------------------------------
# 📈 Model Evaluation
# -------------------------------
#st.subheader("📉 Model Performance")
st.write(f"**R² Score:** {r2_score(y_test, y_pred):.3f}")
st.write(f"**Mean Absolute Error:** {mean_absolute_error(y_test, y_pred):.2f} km")

# -------------------------------
# ⚙️ User Input Section
# -------------------------------
st.subheader("⚙️ Predict EV Range")
battery_capacity = st.number_input("Enter Battery Capacity (kWh)", min_value=10.0, max_value=200.0, step=1.0)
required_range = st.number_input("Enter Desired Range (km)", min_value=50.0, max_value=1000.0, step=10.0)

# -------------------------------
# 🤖 Prediction Logic
# -------------------------------
if st.button("Predict Range"):
    predicted_range = model.predict([[battery_capacity]])[0]
    st.success(f"🔋 Estimated Range: **{predicted_range:.2f} km**")

    if predicted_range >= required_range:
        st.info("✅ This configuration can cover your required range!")
    else:
        st.warning("⚠️ Battery capacity may not be sufficient for your desired range.")

# -------------------------------
# 📊 Visualization
# -------------------------------
#st.subheader("📊 Battery Capacity vs Range (with Regression Line)")
#fig, ax = plt.subplots()
#sns.scatterplot(data=df, x="battery_capacity_kWh", y="range_km", ax=ax, label="Actual Data")
#sns.lineplot(x=X_train['battery_capacity_kWh'], y=model.predict(X_train), color="red", label="Regression Line", ax=ax)
plt.xlabel("Battery Capacity (kWh)")
plt.ylabel("Range (km)")
plt.legend()
#st.pyplot(fig)

# -------------------------------
# 🧠 Insights
# -------------------------------
st.subheader("🧠 Model Insights")
coef = model.coef_[0]
intercept = model.intercept_
st.write(f"Each additional **1 kWh** adds approximately **{coef:.2f} km** to the range.")
st.write(f"Base range (intercept): **{intercept:.2f} km** when battery capacity is 0.")
