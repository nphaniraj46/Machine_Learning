#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import joblib
import numpy as np

# Load saved objects
model = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")   

# Updated cluster names (based on your final profiling)
cluster_names = {
    0: "Frequent Buyers",
    1: "Low Spenders",
    2: "Deal Buyers",
    3: "High Spenders"
}

st.title("Customer Segmentation")

st.write("Enter customer details to predict segment")

income = st.number_input("Income", min_value=0.0, value=50000.0)
total_spent = st.number_input("Total Spending", min_value=0.0, value=500.0)
total_children = st.number_input("Total Children", min_value=0, max_value=10, value=1)
num_deals = st.number_input("Number of Deal Purchases", min_value=0, value=2)
num_web_visits = st.number_input("Web Visits per Month", min_value=0, value=5)
total_purchases = st.number_input("Total Purchases", min_value=0, value=10)

if st.button("Predict Segment"):

    # Arrange features in SAME order as training
    new_customer = np.array([[income,
                              total_spent,
                              total_children,
                              num_deals,
                              num_web_visits,
                              total_purchases]])

    # Apply same preprocessing pipeline
    scaled_data = scaler.transform(new_customer)
    pca_data = pca.transform(scaled_data)

    prediction = model.predict(pca_data)[0]
    segment = cluster_names.get(prediction, "Unknown Segment")

    st.success(f"Predicted Segment: {segment}")