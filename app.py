import streamlit as st
import pandas as pd
import numpy as np
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Home Energy AI",
    page_icon="⚡",
    layout="wide"
)


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load("home_energy_model.pkl")

features = joblib.load("model_features.pkl")


# ============================================================
# TITLE
# ============================================================

st.title("⚡ Home Energy AI")

st.subheader("Next-Hour Energy Consumption Predictor")

st.write(
    "Predict the next hour's household average "
    "active power consumption using machine learning."
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Enter Current Information")

hour = st.sidebar.slider(
    "Current Hour",
    min_value=0,
    max_value=23,
    value=12
)

day_of_week = st.sidebar.selectbox(
    "Day of Week",
    [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ]
)

month = st.sidebar.slider(
    "Month",
    min_value=1,
    max_value=12,
    value=6
)

day_of_month = st.sidebar.slider(
    "Day of Month",
    min_value=1,
    max_value=31,
    value=15
)

is_weekend = 1 if day_of_week in ["Saturday", "Sunday"] else 0


# ============================================================
# HISTORICAL USAGE INPUTS
# ============================================================

st.sidebar.header("Historical Usage")

lag_1 = st.sidebar.number_input(
    "Previous Hour (kW)",
    min_value=0.0,
    value=1.5,
    step=0.1
)

lag_2 = st.sidebar.number_input(
    "2 Hours Ago (kW)",
    min_value=0.0,
    value=1.5,
    step=0.1
)

lag_24 = st.sidebar.number_input(
    "Same Hour Yesterday (kW)",
    min_value=0.0,
    value=1.5,
    step=0.1
)

lag_168 = st.sidebar.number_input(
    "Same Hour Last Week (kW)",
    min_value=0.0,
    value=1.5,
    step=0.1
)

rolling_24 = st.sidebar.number_input(
    "24-Hour Average (kW)",
    min_value=0.0,
    value=1.5,
    step=0.1
)

rolling_168 = st.sidebar.number_input(
    "7-Day Average (kW)",
    min_value=0.0,
    value=1.5,
    step=0.1
)


# ============================================================
# CREATE INPUT DATA
# ============================================================

input_data = pd.DataFrame({
    "hour": [hour],
    "day_of_week": [
        [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday"
        ].index(day_of_week)
    ],
    "month": [month],
    "day_of_month": [day_of_month],
    "is_weekend": [is_weekend],
    "lag_1": [lag_1],
    "lag_2": [lag_2],
    "lag_24": [lag_24],
    "lag_168": [lag_168],
    "rolling_24": [rolling_24],
    "rolling_168": [rolling_168]
})


# Make sure feature order is exactly the same
# as during model training

input_data = input_data[features]


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.divider()

if st.button(
    "⚡ Predict Next Hour",
    use_container_width=True
):

    prediction = model.predict(input_data)[0]

    st.success(
        f"Predicted next-hour consumption: {prediction:.2f} kW"
    )


    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Predicted Consumption",
            f"{prediction:.2f} kW"
        )

    with col2:
        st.metric(
            "Previous Hour",
            f"{lag_1:.2f} kW"
        )

    with col3:
        difference = prediction - lag_1

        st.metric(
            "Change",
            f"{difference:+.2f} kW"
        )


    # ========================================================
    # SIMPLE INTERPRETATION
    # ========================================================

    st.subheader("Prediction Analysis")

    if prediction > lag_1:
        st.warning(
            "⚠️ The model expects energy consumption "
            "to increase during the next hour."
        )

    elif prediction < lag_1:
        st.info(
            "ℹ️ The model expects energy consumption "
            "to decrease during the next hour."
        )

    else:
        st.info(
            "Energy consumption is expected to remain "
            "approximately the same."
        )


# ============================================================
# MODEL INFORMATION
# ============================================================

st.divider()

st.subheader("🤖 Model Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Algorithm",
        "Random Forest"
    )

with col2:
    st.metric(
        "Features",
        "11"
    )

with col3:
    st.metric(
        "Prediction",
        "Next Hour"
    )


# ============================================================
# FEATURES USED
# ============================================================

with st.expander("View Features Used by Model"):

    feature_df = pd.DataFrame({
        "Feature": features
    })

    st.dataframe(
        feature_df,
        use_container_width=True
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Home Energy AI | Machine Learning Project"
)

st.caption(
    "Model: Random Forest Regressor | "
    "Target: Next-hour average Global Active Power"
)