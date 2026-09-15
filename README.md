# ⚡ AI-Based Energy Scheduling for Telecom Microgrids

An AI-based energy management system for telecom microgrids that combines **energy demand forecasting, grid outage prediction, and predictive battery scheduling** to improve energy utilization and reduce dependency on diesel generators.

## 📌 Overview

Telecom towers require continuous power to maintain reliable network connectivity. Grid interruptions can force towers to depend on battery backup and diesel generators, increasing operational costs and fuel consumption.

This project develops a simulation-based energy scheduling system that uses machine learning to anticipate future energy demand and possible grid outages. The predictions are then used to make battery charging and discharging decisions.

The system also compares the proposed **AI-based predictive scheduling** with a conventional **reactive scheduling** approach.

## 🚀 Key Features

* ⚡ **Energy Demand Forecasting** using LSTM
* 🔌 **Grid Outage Prediction** using Random Forest
* 🔋 **Predictive Battery Scheduling**
* ⛽ Diesel generator usage estimation
* 💰 Energy cost comparison
* 📊 Battery State of Charge (SOC) visualization
* 🤖 Manual and AI-based grid simulation modes
* 🌐 Interactive Streamlit dashboard

## 🧠 System Architecture

```text
Historical Energy Data
        │
        ▼
   Data Processing
        │
        ├──────────────────┐
        ▼                  ▼
 LSTM Forecasting    Random Forest
        │             Outage Prediction
        │                  │
        └────────┬─────────┘
                 ▼
       Predictive Scheduler
                 │
        ┌────────┴────────┐
        ▼                 ▼
   Battery Usage     Diesel Usage
        │                 │
        └────────┬────────┘
                 ▼
          Cost Analysis
                 │
                 ▼
        Streamlit Dashboard
```

## 📊 Dataset & Data Preparation

The project uses the **LD2011_2014 electricity consumption dataset**, which contains power consumption data from multiple clients over a four-year period.

Since the original dataset contains electricity usage patterns from different types of consumers, the client profiles were analyzed to identify a consumption pattern that closely resembles the typical power requirements of a **telecom tower**. The selected client's power consumption data was then extracted and processed to create the dataset used for the telecom microgrid simulation.

The selected power consumption profile was further processed into an hourly time-series dataset and used as the basis for:

* Energy demand forecasting using LSTM
* Grid outage simulation and prediction
* Battery scheduling
* Diesel generator usage analysis
* Energy cost comparison

This approach allows the project to experiment with telecom microgrid energy management using a real-world electricity consumption dataset while adapting the consumption profile to represent a telecom tower environment.


## 🤖 Machine Learning Models

### 1. LSTM Energy Demand Forecasting

An LSTM (Long Short-Term Memory) neural network is used to forecast upcoming energy demand.

* Input window: **24 hours**
* Forecast horizon: **8 hours**
* LSTM units: **50**
* Optimizer: **Adam**
* Loss function: **Mean Squared Error**
* Training epochs: **20**

The predicted values are converted back to the original energy scale using a saved `MinMaxScaler`.

### 2. Random Forest Outage Prediction

A Random Forest classifier is used to estimate the probability of grid outages.

The model uses:

* Hour of the day
* Day of the week

The outage dataset is generated from the available tower data using an evening outage pattern with additional random outages for simulation purposes.

The trained model is saved as:

```text
outage_model.pkl
```

## 🔋 Energy Scheduling

The project implements two scheduling strategies.

### Reactive Scheduling

The conventional approach reacts to the current grid condition:

* Uses grid power when available
* Charges the battery during grid availability
* Uses battery power during outages
* Uses diesel generation when battery power is insufficient

### Predictive Scheduling

The AI-based approach considers:

* Forecasted future energy demand
* Predicted outage probability
* Current battery SOC
* Minimum battery SOC threshold

This allows the scheduler to adjust battery charging based on expected future conditions rather than reacting only after an outage occurs.

## 📊 Simulation

The Streamlit application provides two grid simulation modes.

### Manual Mode

Users can manually specify whether the grid is available for each of the next 8 time steps.

### AI Outage Prediction Mode

The Random Forest model predicts outage probabilities for the next 8 time steps based on the selected starting hour.

The application then uses these predictions along with the LSTM demand forecast to perform predictive scheduling.

## 📈 Results

The dashboard compares the two scheduling strategies based on:

* Total energy cost
* Diesel energy used
* Battery State of Charge

It also reports the difference between the reactive and predictive approaches.

> **Note:** This project is a simulation/prototype. The outage data used for training is synthetically generated from predefined temporal patterns and randomness, so the results should not be interpreted as validated real-world telecom network performance.

## 🛠️ Technologies Used

* Python
* TensorFlow / Keras
* Scikit-learn
* Pandas
* NumPy
* Joblib
* Matplotlib
* Streamlit

## 📂 Project Structure

```text
├── appo.py
├── lstm_train.py
├── outage_data.py
├── outage_train.py
├── tower_hourly_MT_321.csv
├── tower_hourly_MT_321_out.csv
├── lstm_model_new.h5
├── outage_model.pkl
└── scaler.save
```

### File Description

| File                          | Description                                            |
| ----------------------------- | ------------------------------------------------------ |
| `appo.py`                     | Streamlit application and energy scheduling simulator  |
| `lstm_train.py`               | Trains and evaluates the LSTM demand forecasting model |
| `outage_data.py`              | Generates the simulated outage dataset                 |
| `outage_train.py`             | Trains the Random Forest outage prediction model       |
| `tower_hourly_MT_321.csv`     | Original hourly telecom tower energy data              |
| `tower_hourly_MT_321_out.csv` | Dataset containing simulated outage labels             |
| `lstm_model_new.h5`           | Trained LSTM model                                     |
| `outage_model.pkl`            | Trained Random Forest outage model                     |
| `scaler.save`                 | Saved MinMaxScaler used for LSTM preprocessing         |

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd <repository-folder>
```

### 2. Install dependencies

```bash
pip install streamlit tensorflow scikit-learn pandas numpy matplotlib joblib
```

### 3. Run the application

```bash
streamlit run appo.py
```

The Streamlit dashboard will open in your browser.

## 🔮 Future Improvements

* Use real telecom grid outage data instead of simulated outage labels
* Include additional weather and environmental features
* Incorporate solar generation and battery degradation models
* Optimize scheduling using reinforcement learning
* Add real-time IoT-based telecom tower monitoring
* Deploy the system for multiple telecom sites
* Improve outage prediction using additional historical grid features

## 👨‍💻 Project

Developed as an AI and Data Science project exploring the application of machine learning for **energy management and optimization in telecom microgrids**.
