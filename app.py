import streamlit as st
import pandas as pd
import joblib

# ---------------------------------
# PAGE SETTINGS
# ---------------------------------
st.set_page_config(page_title="Customer Segmentation", layout="wide")
st.title("🧠 Customer Segmentation App")

st.write("Upload the marketing_campaign.xlsx file to generate customer segments.")

# ---------------------------------
# LOAD TRAINED FILES
# ---------------------------------
scaler = joblib.load("scaler.pkl")
model = joblib.load("kmeans_model.pkl")

# ---------------------------------
# FILE UPLOADER
# ---------------------------------
uploaded_file = st.file_uploader(
    "C:\Users\PHANI\OneDrive\Desktop\Machine Learning\marketing_campaign.xlsx",
    type=["xlsx"]
)

# Show message instead of blank page
if uploaded_file is None:
    st.info("⬆️ Please upload marketing_campaign.xlsx to start segmentation.")
    st.stop()

# ---------------------------------
# READ DATA
# ---------------------------------
df = pd.read_excel(uploaded_file)

st.subheader("Uploaded Data Preview")
st.dataframe(df.head())

# ---------------------------------
# FEATURE ENGINEERING (SAME AS NOTEBOOK)
# ---------------------------------
df['Age'] = 2026 - df['Year_Birth']
df['TotalChildren'] = df['Kidhome'] + df['Teenhome']

spend_cols = [
    'MntWines','MntFruits','MntMeatProducts',
    'MntFishProducts','MntSweetProducts','MntGoldProds'
]
df['TotalSpent'] = df[spend_cols].sum(axis=1)

purchase_cols = [
    'NumWebPurchases',
    'NumCatalogPurchases',
    'NumStorePurchases'
]
df['TotalPurchases'] = df[purchase_cols].sum(axis=1)

# ---------------------------------
# CLEAN MARITAL STATUS
# ---------------------------------
df['Marital_Status'] = df['Marital_Status'].replace({
    'Married':'Partner',
    'Together':'Partner',
    'Single':'Alone',
    'Divorced':'Alone',
    'Widow':'Alone',
    'YOLO':'Alone',
    'Absurd':'Alone'
})

df = pd.get_dummies(df, columns=['Marital_Status'], drop_first=True)

# Ensure column exists (important during deployment)
if 'Marital_Status_Partner' not in df.columns:
    df['Marital_Status_Partner'] = 0

# ---------------------------------
# FINAL FEATURES (same as training)
# ---------------------------------
features = [
    'Income',
    'Age',
    'Recency',
    'TotalChildren',
    'TotalSpent',
    'TotalPurchases',
    'NumWebVisitsMonth',
    'TotalAcceptedOffers',
    'Marital_Status_Partner'
]

X = df[features].fillna(df[features].median())

# ---------------------------------
# SCALE USING SAVED SCALER
# ---------------------------------
X_scaled = scaler.transform(X)

# ---------------------------------
# PREDICT CLUSTERS
# ---------------------------------
df['Cluster'] = model.predict(X_scaled)

# ---------------------------------
# OUTPUT
# ---------------------------------
st.subheader("Customer Segments")
st.dataframe(df[['ID','Cluster']].head(20))

st.subheader("Cluster Profiles")
st.dataframe(df.groupby('Cluster')[features].mean().round(2))

st.success("✅ Segmentation Completed!")