import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import joblib

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_components():
    lstm_model = load_model("lstm_model_new.h5")
    scaler = joblib.load("scaler.save")
    outage_model = joblib.load("outage_model.pkl")
    return lstm_model, scaler, outage_model

model, scaler, outage_model = load_components()

# ---------------- PARAMETERS ----------------
battery_capacity = 1000
min_soc = 0.2
charge_efficiency = 0.9
grid_cost = 6
diesel_cost = 18

# ---------------- FORECAST ----------------
def generate_forecast(data, window_size=24, steps=8):
    scaled_data = scaler.transform(data.reshape(-1,1))
    x_input = scaled_data[-window_size:]
    x_input = x_input.reshape(1, window_size, 1)

    predictions = []
    for _ in range(steps):
        pred = model.predict(x_input, verbose=0)[0][0]
        predictions.append(pred)
        x_input = np.append(x_input[:,1:,:], [[[pred]]], axis=1)

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1,1))
    return predictions.flatten()

# ---------------- OUTAGE PREDICTION ----------------
def predict_outage_sequence(start_hour, steps=8):

    probs = []

    for i in range(steps):
        hour = (start_hour + i) % 24
        day = 0  # simplified

        prob = outage_model.predict_proba([[hour, day]])[0][1]
        probs.append(prob)

    return probs

# ---------------- REACTIVE ----------------
def reactive_schedule(demand, soc, grid_pattern):

    battery_energy = soc * battery_capacity
    total_cost = 0
    diesel_used = 0

    for i in range(len(demand)):

        if grid_pattern[i]:
            total_cost += demand[i] * grid_cost

            if battery_energy < battery_capacity:
                battery_energy += 50 * charge_efficiency

        else:
            usable = battery_energy - battery_capacity * min_soc

            if usable > 0:
                used = min(demand[i], usable)
                battery_energy -= used
                demand[i] -= used

            if demand[i] > 0:
                diesel_used += demand[i]
                total_cost += demand[i] * diesel_cost

    return total_cost, diesel_used

# ---------------- PREDICTIVE ----------------
def predictive_schedule(demand, forecast, soc, grid_pattern, outage_prob):

    battery_energy = soc * battery_capacity
    total_cost = 0
    diesel_used = 0
    soc_history = []
    smart_used = False

    for i in range(len(demand)):

        if grid_pattern[i]:

            total_cost += demand[i] * grid_cost

            # 🔮 Look ahead safely
            future_slice_demand = forecast[i+1:min(i+3, len(forecast))]
            future_demand = sum(future_slice_demand) if len(future_slice_demand) > 0 else 0

            future_slice_outage = outage_prob[i+1:min(i+3, len(outage_prob))]
            future_outage_prob = max(future_slice_outage) if len(future_slice_outage) > 0 else 0

            usable_battery = battery_energy - battery_capacity * min_soc
            soc_level = battery_energy / battery_capacity

            future_energy_needed = 0.6 * future_demand

            # 🚫 DO NOTHING CONDITION
            if battery_energy >= future_energy_needed:
                charge = 0

            # 🎯 smarter condition
            if (
                future_outage_prob > 0.75 and
                soc_level < 0.85
            ):
                smart_used = True

                charge = min(
                50 + 50 * future_outage_prob,
                battery_capacity - battery_energy
                )
            else:
                charge = min(30, battery_capacity - battery_energy)

            battery_energy += charge * charge_efficiency

        else:
            usable = battery_energy - battery_capacity * min_soc

            if usable > 0:
                used = min(demand[i], usable)
                battery_energy -= used
                demand[i] -= used

            if demand[i] > 0:
                diesel_used += demand[i]
                total_cost += demand[i] * diesel_cost

        soc_history.append(battery_energy / battery_capacity)

    return total_cost, diesel_used, soc_history, smart_used


# ---------------- UI ----------------
st.title("⚡ AI Energy Scheduling Simulator")

data = pd.read_csv("tower_hourly_MT_321.csv")["energy_demand_kWh"].values

soc = st.slider("Initial Battery SOC", 0.2, 1.0, 0.6)
steps = 8

# 🔥 MODE SWITCH
mode = st.selectbox("Grid Mode", ["Manual", "AI Outage Prediction"])

# ---------------- GRID PATTERN ----------------
if mode == "Manual":
    st.subheader("Grid Control")
    grid_pattern = []
    cols = st.columns(steps)

    for i in range(steps):
        with cols[i]:
            state = st.toggle(f"T{i}", value=True)
            grid_pattern.append(state)

    outage_prob = [0 if g else 1 for g in grid_pattern]

else:
    st.subheader("AI Predicted Outages")

    start_hour = st.slider("Start Hour", 0, 23, 10)

    outage_prob = predict_outage_sequence(start_hour, steps)
    grid_pattern = [False if p > 0.6 else True for p in outage_prob]

    st.line_chart(outage_prob)

# ---------------- FORECAST ----------------
forecast = generate_forecast(data, steps=steps)
demand = forecast.copy()

# ---------------- RUN ----------------
reactive_cost, reactive_diesel = reactive_schedule(
    demand.copy(), soc, grid_pattern
)

cost, diesel, soc_hist, smart_used = predictive_schedule(
    demand.copy(), forecast, soc, grid_pattern, outage_prob
)

# ---------------- RESULTS ----------------
st.subheader("Results")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### ⚙️ Reactive System")
    st.write(f"💰 Cost: {reactive_cost:.2f}")
    st.write(f"⛽ Diesel: {reactive_diesel:.2f}")

with col2:
    st.markdown("### 🤖 Predictive (AI)")
    st.write(f"💰 Cost: {cost:.2f}")
    st.write(f"⛽ Diesel: {diesel:.2f}")

# 🔥 IMPROVEMENT
cost_saved = reactive_cost - cost
diesel_saved = reactive_diesel - diesel

st.markdown("### 🚀 Improvement")

if cost_saved >= 0:
    st.success(f"💰 Cost Saved: {cost_saved:.2f}")
else:
    st.error(f"💸 Extra Cost: {-cost_saved:.2f}")

if diesel_saved >= 0:
    st.success(f"⛽ Diesel Saved: {diesel_saved:.2f}")
else:
    st.error(f"⛽ Extra Diesel Used: {-diesel_saved:.2f}")

# 🔥 SMART STATUS
if smart_used:
    st.success("⚡ Smart Charging Activated (Forecast + Outage Aware)")
else:
    st.info("Normal Charging Mode")

# ---------------- PLOTS ----------------
st.markdown("### 📈 Battery SOC Analysis")

fig, ax = plt.subplots()
ax.plot(soc_hist, marker='o')
ax.set_title("Battery State of Charge")
ax.set_xlabel("Time Step")
ax.set_ylabel("SOC (0-1)")
st.pyplot(fig, use_container_width=True)

st.markdown("### ⚡ Demand Forecast")

fig2, ax2 = plt.subplots()
ax2.plot(forecast, marker='o')
ax2.set_title("Forecasted Demand")
ax2.set_xlabel("Time Step")
ax2.set_ylabel("Energy Demand (kWh)")
st.pyplot(fig2, use_container_width=True)