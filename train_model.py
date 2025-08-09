# train_model.py
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# =======================
# 1. Load & Prepare Data
# =======================
df = pd.read_csv("restaurant_menu_optimization_data.csv")

# Encode target variable
le = LabelEncoder()
df['Profitability'] = le.fit_transform(df['Profitability'])

# One-hot encode categorical columns
df = pd.get_dummies(df, columns=['RestaurantID', 'MenuCategory'], drop_first=True)

# Drop non-useful columns
df.drop(['Ingredients', 'MenuItem'], axis=1, inplace=True)

# Separate features & target
X = df.drop('Profitability', axis=1)
y = df['Profitability']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =======================
# 2. Hyperparameter Tuning
# =======================
best_acc = 0
best_params = {}

for metric in ['euclidean', 'manhattan', 'minkowski']:
    for weights in ['uniform', 'distance']:
        for k in range(1, 51):
            knn = KNeighborsClassifier(n_neighbors=k, metric=metric, weights=weights)
            knn.fit(X_train_scaled, y_train)
            y_pred = knn.predict(X_test_scaled)
            acc = accuracy_score(y_test, y_pred)

            if acc > best_acc:
                best_acc = acc
                best_params = {'k': k, 'metric': metric, 'weights': weights}

# =======================
# 3. Train Final Model
# =======================
final_knn = KNeighborsClassifier(
    n_neighbors=best_params['k'],
    metric=best_params['metric'],
    weights=best_params['weights']
)
final_knn.fit(X_train_scaled, y_train)

# =======================
# 4. Save Model & Objects
# =======================
with open("knn_model.pkl", "wb") as f:
    pickle.dump(final_knn, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print(f"✅ Model trained and saved! Best Params: {best_params} | Accuracy: {best_acc:.4f}")
