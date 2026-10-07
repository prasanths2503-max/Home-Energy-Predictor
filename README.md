# 🏠 Home Energy Consumption Predictor

A machine learning project that predicts the **next hour's average household electricity consumption** using historical power usage and time-based patterns.

The project uses the **Individual Household Electric Power Consumption** dataset and a **Random Forest Regressor** to forecast future household power demand.

---

## 📌 Project Overview

Electricity consumption changes throughout the day depending on factors such as:

- Time of day
- Day of the week
- Previous electricity usage
- Daily usage patterns
- Weekly usage patterns

This project uses historical household electricity data to predict the **next hour's average Global Active Power**.

> **Target:** Next-hour average active power (kW)

---

## 🎯 Objective

The main objective is to build a machine learning model that can:

1. Process raw household electricity data.
2. Clean missing values.
3. Convert minute-level data into hourly data.
4. Create time-based features.
5. Create historical lag features.
6. Create rolling average features.
7. Train a Random Forest regression model.
8. Evaluate the model using regression metrics.
9. Save the trained model.
10. Use the model in a Streamlit application.

---

## 📊 Dataset

The project uses the **Individual Household Electric Power Consumption** dataset.

The original dataset contains measurements collected from a household over time.

Important columns include:

| Column | Description |
|---|---|
| `Date` | Date of measurement |
| `Time` | Time of measurement |
| `Global_active_power` | Global household active power (kW) |
| `Global_reactive_power` | Global reactive power (kW) |
| `Voltage` | Voltage (V) |
| `Global_intensity` | Global intensity (A) |
| `Sub_metering_1` | Energy sub-metering 1 |
| `Sub_metering_2` | Energy sub-metering 2 |
| `Sub_metering_3` | Energy sub-metering 3 |

---

## ⚙️ Data Preprocessing

The original data contains measurements at a very high frequency.

The following preprocessing steps were performed:

### 1. Combine Date and Time

The `Date` and `Time` columns were combined into a single datetime column.

### 2. Handle Missing Values

Missing numerical values were handled using time-based interpolation followed by forward and backward filling.

### 3. Convert to Hourly Data

The cleaned data was resampled into **hourly averages**.

### 4. Create Prediction Target

The target variable was created by shifting the current hour's active power:

```python
hourly_df["target"] = hourly_df["Global_active_power"].shift(-1)
