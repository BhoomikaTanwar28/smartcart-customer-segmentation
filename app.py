import streamlit as st
import pickle
import pandas as pd

# -----------------------------
# Load Saved Models
# -----------------------------
scaler = pickle.load(open("scaler.pkl", "rb"))
pca = pickle.load(open("pca.pkl", "rb"))
kmeans = pickle.load(open("kmeans.pkl", "rb"))
ohe = pickle.load(open("ohe.pkl", "rb"))

# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(
    page_title="SmartCart Customer Segmentation",
    page_icon="🛒",
    layout="wide"
)

# -----------------------------
# Segment Information
# -----------------------------
segment_info = {
    0: {
        "name": "💎 Premium Customer",
        "description": "High spending and frequent purchases. Offer premium membership, loyalty rewards and exclusive offers."
    },
    1: {
        "name": "🛍️ Regular Customer",
        "description": "Moderate spending with consistent purchases. Personalized discounts can increase engagement."
    },
    2: {
        "name": "💰 Budget Customer",
        "description": "Price-sensitive customer. Discounts and promotional campaigns work best."
    }
}

# -----------------------------
# Title
# -----------------------------
st.title("🛒 SmartCart Customer Segmentation")
st.markdown("""
Predict customer segments using a **Machine Learning K-Means Clustering Model**.
Fill in the customer details from the sidebar and click **Predict Segment**.
""")

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("📋 Customer Details")

income = st.sidebar.number_input("Income", min_value=0.0)

recency = st.sidebar.number_input("Recency", min_value=0.0)

num_deals = st.sidebar.number_input("NumDealsPurchases", min_value=0.0)

num_web = st.sidebar.number_input("NumWebPurchases", min_value=0.0)

num_catalog = st.sidebar.number_input("NumCatalogPurchases", min_value=0.0)

num_store = st.sidebar.number_input("NumStorePurchases", min_value=0.0)

num_web_visits = st.sidebar.number_input("NumWebVisitsMonth", min_value=0.0)

complain = st.sidebar.selectbox("Complain", [0, 1])

response = st.sidebar.selectbox("Response", [0, 1])

age = st.sidebar.number_input("Age", min_value=0.0)

tenure = st.sidebar.number_input("Customer Tenure (Days)", min_value=0.0)

spending = st.sidebar.number_input("Total Spending", min_value=0.0)

children = st.sidebar.number_input("Total Children", min_value=0.0)

education = st.sidebar.selectbox(
    "Education",
    ["Graduate", "Postgraduate", "Undergraduate"]
)

living = st.sidebar.selectbox(
    "Living With",
    ["Alone", "Partner"]
)

# -----------------------------
# Prediction
# -----------------------------
if st.sidebar.button("🚀 Predict Segment"):

    cat_df = pd.DataFrame({
        "Education": [education],
        "Living_With": [living]
    })

    encoded = ohe.transform(cat_df)

    encoded_df = pd.DataFrame(
        encoded.toarray(),
        columns=ohe.get_feature_names_out(["Education", "Living_With"])
    )

    num_df = pd.DataFrame({
        "Income": [income],
        "Recency": [recency],
        "NumDealsPurchases": [num_deals],
        "NumWebPurchases": [num_web],
        "NumCatalogPurchases": [num_catalog],
        "NumStorePurchases": [num_store],
        "NumWebVisitsMonth": [num_web_visits],
        "Complain": [complain],
        "Response": [response],
        "Age": [age],
        "Customer_Tenure_Days": [tenure],
        "Total_Spending": [spending],
        "Total_Children": [children]
    })

    final_df = pd.concat([num_df, encoded_df], axis=1)

    scaled = scaler.transform(final_df)

    reduced = pca.transform(scaled)

    cluster = kmeans.predict(reduced)

    segment = int(cluster[0])

    st.success(f"### 🎯 Predicted Segment: {segment_info[segment]['name']}")

    st.info(segment_info[segment]["description"])

    st.subheader("📄 Customer Summary")

    st.dataframe(final_df)