# app.py
import streamlit as st
import pandas as pd
import pickle

# -------------------------
# 1. Load trained model, scaler, and label encoder
# -------------------------
with open("knn_model.pkl", "rb") as model_file:
    knn_model = pickle.load(model_file)

with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

with open("label_encoder.pkl", "rb") as le_file:
    label_encoder = pickle.load(le_file)

# -------------------------
# 2. Load dataset for column structure
# -------------------------
df_train = pd.read_csv("restaurant_menu_optimization_data.csv")

# Encode target variable exactly as in training
df_train["Profitability"] = label_encoder.transform(df_train["Profitability"])

# One-hot encode categorical columns
df_train = pd.get_dummies(df_train, columns=["RestaurantID", "MenuCategory"], drop_first=True)

# Drop non-useful columns
df_train.drop(["Ingredients", "MenuItem"], axis=1, inplace=True)

# Get feature columns (same order as training)
feature_columns = df_train.drop("Profitability", axis=1).columns

# -------------------------
# 3. Streamlit UI
# -------------------------
st.set_page_config(page_title="Restaurant Menu Profitability Prediction", layout="centered")
st.title("🍽️ Restaurant Menu Profitability Prediction")

st.write("Fill in the details below to predict the profitability level of a menu item.")

restaurant_id = st.selectbox(
    "Restaurant ID",
    sorted(pd.read_csv("restaurant_menu_optimization_data.csv")["RestaurantID"].unique())
)
menu_category = st.selectbox(
    "Menu Category",
    sorted(pd.read_csv("restaurant_menu_optimization_data.csv")["MenuCategory"].unique())
)
price = st.number_input("Price", min_value=0.0, step=0.01, format="%.2f")

# -------------------------
# 4. Prediction Logic
# -------------------------
if st.button("Predict Profitability"):
    # Create input DataFrame
    input_df = pd.DataFrame({
        "Price": [price],
        "RestaurantID": [restaurant_id],
        "MenuCategory": [menu_category]
    })

    # One-hot encode input
    input_encoded = pd.get_dummies(input_df, columns=["RestaurantID", "MenuCategory"], drop_first=True)

    # Align with training feature columns
    input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)

    # Scale
    input_scaled = scaler.transform(input_encoded)

    # Predict (numeric)
    prediction_numeric = knn_model.predict(input_scaled)[0]

    # Decode label
    predicted_label = label_encoder.inverse_transform([prediction_numeric])[0]

    # Display result
    color_map = {"Low": "🔴", "Medium": "🟡", "High": "🟢"}
    st.success(f"Predicted Profitability: {color_map.get(predicted_label, '')} **{predicted_label}**")
