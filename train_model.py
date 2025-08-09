# train_model.py
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

# ============================
# 1. Load & Prepare Data
# ============================
df = pd.read_csv("restaurant_menu_optimization_data.csv")

# Map target labels
profit_map = {'Low': 0, 'Medium': 1, 'High': 2}
reverse_profit_map = {v: k for k, v in profit_map.items()}
df['Profitability'] = df['Profitability'].map(profit_map)

# Store category mappings for factorization
restaurant_map = {cat: i+1 for i, cat in enumerate(df['RestaurantID'].unique())}
category_map = {cat: i+1 for i, cat in enumerate(df['MenuCategory'].unique())}

# Factorize
df['RestaurantID'] = df['RestaurantID'].map(restaurant_map)
df['MenuCategory'] = df['MenuCategory'].map(category_map)

# Features and target
X = df[['Price', 'RestaurantID', 'MenuCategory']]
y = df['Profitability']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Parameter grid for KNN tuning
param_grid = {
    'n_neighbors': range(1, 51),
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski', 'chebyshev'],
    'p': [1, 2],
    'leaf_size': [10, 20, 30, 40, 50],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
}

grid = GridSearchCV(
    KNeighborsClassifier(),
    param_grid,
    cv=10,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2
)
grid.fit(X_train_scaled, y_train)

# Best model
knn = grid.best_estimator_
best_params = grid.best_params_
best_acc = grid.best_score_
clf_report = classification_report(y_test, knn.predict(X_test_scaled))

# Save model & artifacts
with open("knn_model.pkl", "wb") as f:
    pickle.dump(knn, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("mappings.pkl", "wb") as f:
    pickle.dump({
        "profit_map": profit_map,
        "reverse_profit_map": reverse_profit_map,
        "restaurant_map": restaurant_map,
        "category_map": category_map
    }, f)
with open("model_info.pkl", "wb") as f:
    pickle.dump({
        "best_params": best_params,
        "best_acc": best_acc,
        "clf_report": clf_report
    }, f)

print("✅ Model training complete!")
print(f"Best Parameters: {best_params}")
print(f"Best CV Accuracy: {best_acc:.2%}")
