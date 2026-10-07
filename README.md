# 🏠 Home Energy Consumption Predictor

<p align="center">
  <b>Machine Learning • Time-Series Forecasting • Streamlit</b>
</p>

<p align="center">
  Predict the next hour's household electricity consumption using historical power usage and time-based patterns.
</p>

<p align="center">

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas)
![Scikit Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit)
![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?logo=github)

</p>

---

## 📌 Overview

The **Home Energy Consumption Predictor** is a machine learning project that predicts the **next hour's average household active power consumption**.

The project uses historical household electricity measurements to identify patterns based on:

- Previous electricity consumption
- Time of day
- Day of the week
- Daily usage patterns
- Weekly usage patterns
- Rolling electricity consumption

A **Random Forest Regressor** is trained on these features and integrated into a **Streamlit application** for interactive predictions.

---

## 🎯 Project Objective

The goal of this project is to demonstrate a complete machine learning workflow:

```text
Raw Dataset
     ↓
Data Cleaning
     ↓
Datetime Processing
     ↓
Hourly Resampling
     ↓
Feature Engineering
     ↓
Train/Test Split
     ↓
Random Forest Model
     ↓
Model Evaluation
     ↓
Model Serialization
     ↓
Streamlit Application
