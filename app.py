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
df_train["Profitability"] = label_encoder.transform(df_train["Profitability"])
df_train = pd.get_dummies(df_train, columns=["RestaurantID", "MenuCategory"], drop_first=True)
df_train.drop(["Ingredients", "MenuItem"], axis=1, inplace=True)
feature_columns = df_train.drop("Profitability", axis=1).columns

# -------------------------
# 3. Streamlit UI Styling
# -------------------------
st.set_page_config(page_title="Fadhli's Restaurant Profitability Predictor", layout="centered")

# Custom CSS
st.markdown("""
<style>
    .main { background-color: #0E1117; }
    h1 { text-align: center; color: #f5f5f5; }
    .stSelectbox, .stNumberInput { background-color: #1E222A !important; }
</style>
""", unsafe_allow_html=True)

# Title & description
st.markdown("<h1>🍽️ Restaurant Menu Profitability Predictor</h1>", unsafe_allow_html=True)
st.write("### Created by **Fadhli** — Predict the profitability level of any menu item instantly.")
st.markdown("---")

# -------------------------
# 4. Inputs
# -------------------------
col1, col2 = st.columns(2)
with col1:
    restaurant_id = st.selectbox(
        "🏪 Select Restaurant ID",
        sorted(pd.read_csv("restaurant_menu_optimization_data.csv")["RestaurantID"].unique())
    )
with col2:
    menu_category = st.selectbox(
        "📂 Select Menu Category",
        sorted(pd.read_csv("restaurant_menu_optimization_data.csv")["MenuCategory"].unique())
    )

price = st.number_input("💰 Enter Price", min_value=0.0, step=0.01, format="%.2f")

# -------------------------
# 5. Prediction Logic
# -------------------------
if st.button("🔍 Predict Profitability", type="primary", use_container_width=True):
    # Create input DataFrame
    input_df = pd.DataFrame({
        "Price": [price],
        "RestaurantID": [restaurant_id],
        "MenuCategory": [menu_category]
    })

    # Encode & scale
    input_encoded = pd.get_dummies(input_df, columns=["RestaurantID", "MenuCategory"], drop_first=True)
    input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)
    input_scaled = scaler.transform(input_encoded)

    # Prediction
    prediction_numeric = knn_model.predict(input_scaled)[0]
    predicted_label = label_encoder.inverse_transform([prediction_numeric])[0]

    # Probability scores
    prob_scores = knn_model.predict_proba(input_scaled)[0]
    prob_df = pd.DataFrame({
        "Profitability": label_encoder.inverse_transform(range(len(prob_scores))),
        "Probability": prob_scores
    }).sort_values(by="Probability", ascending=False)

    # Emoji mapping
    color_map = {"Low": "🔴", "Medium": "🟡", "High": "🟢"}

    # Display prediction
    st.markdown("---")
    st.markdown(f"""
        <div style="padding: 20px; background-color: #1E2B1E; border-radius: 10px; text-align: center;">
            <h2 style="color: white;">Predicted Profitability:</h2>
            <h1>{color_map.get(predicted_label, '')} {predicted_label}</h1>
        </div>
    """, unsafe_allow_html=True)

    # Show probability table
    st.write("### Prediction Confidence")
    st.dataframe(prob_df.style.format({"Probability": "{:.2%}"}))

# -------------------------
# 6. Footer
# -------------------------
st.markdown("<br><hr><center>© 2025 Created by Fadhli — Powered by KNN & Streamlit</center>", unsafe_allow_html=True)
