# app.py
import streamlit as st
import pandas as pd
import pickle

# ============================
# Load Model & Mappings
# ============================
@st.cache_resource
def load_artifacts():
    with open("knn_model.pkl", "rb") as f:
        knn = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("mappings.pkl", "rb") as f:
        maps = pickle.load(f)
    with open("model_info.pkl", "rb") as f:
        info = pickle.load(f)
    return knn, scaler, maps, info

knn, scaler, maps, info = load_artifacts()
profit_map = maps["profit_map"]
reverse_profit_map = maps["reverse_profit_map"]
restaurant_map = maps["restaurant_map"]
category_map = maps["category_map"]

# ============================
# Streamlit UI
# ============================
st.title("🍽 Restaurant Menu Profitability Predictor")

st.markdown("""
This app predicts **menu item profitability** (Low, Medium, High) based on:
- **Price**
- **Restaurant ID** (e.g., R001, R002)
- **Menu Category** (e.g., Beverages, Appetizers)
""")

st.info(f"Best Parameters: {info['best_params']} | CV Accuracy: {info['best_acc']:.2%}")

# Input fields
restaurant_id_str = st.selectbox("Restaurant ID", options=list(restaurant_map.keys()))
menu_category_str = st.selectbox("Menu Category", options=list(category_map.keys()))
price = st.number_input("Price", min_value=0.0, step=0.5)

if st.button("Predict Profitability"):
    # Convert to numeric factors
    rest_id_num = restaurant_map.get(restaurant_id_str)
    category_num = category_map.get(menu_category_str)

    # Prepare dataframe
    new_data = pd.DataFrame({
        "Price": [price],
        "RestaurantID": [rest_id_num],
        "MenuCategory": [category_num]
    })

    # Scale
    new_data_scaled = scaler.transform(new_data)

    # Predict
    pred = knn.predict(new_data_scaled)[0]
    st.success(f"Predicted Profitability: **{reverse_profit_map[pred]}**")


