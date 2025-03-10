import tkinter as tk
from tkinter import messagebox, ttk
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import joblib
import os
import threading
from scipy.optimize import differential_evolution

# Load dataset
df = pd.read_excel("paid_media_data.xlsx")
df.columns = df.columns.str.strip()

df_original = df.copy()  # Preserve original dataset before encoding
df = df[['TargetAudience', 'ContentCategory', 'ContentFormat', 'ContentSource', 'Brand', 'ProductCategory', 'Reach',
         'Investment']]

# Clean category values
def clean_category_values(df, columns):
    for col in columns:
        df[col] = df[col].str.strip()
        df[col] = df[col].replace('', pd.NA)
    return df

df = clean_category_values(df, ['TargetAudience', 'ContentCategory', 'ContentFormat', 'ContentSource', 'Brand', 'ProductCategory'])

# One-Hot Encoding
categorical_columns = ['TargetAudience', 'ContentCategory', 'ContentFormat', 'ContentSource', 'Brand', 'ProductCategory']
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# Standardize Investment
scaler = StandardScaler()
df['Investment'] = scaler.fit_transform(df[['Investment']]).flatten()

# Split Data
X = df.drop(columns=["Reach"])
y = df["Reach"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define Hyperparameter Grid
param_grid = {
    "learning_rate": [0.05, 0.1],
    "n_estimators": [100, 300],
    "max_depth": [3, 5],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 3],
    "subsample": [0.8, 1.0],
    "max_features": ["sqrt"]
}

# Function to Train Model
def train_model():
    global best_model
    print("🚀 Training Model with GridSearchCV...")
    grid_search = GridSearchCV(GradientBoostingRegressor(random_state=42), param_grid, cv=3, scoring="neg_mean_squared_error", n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    print("✅ Best Model Found! MSE:", mean_squared_error(y_test, best_model.predict(X_test)))
    joblib.dump(best_model, "models/reach_prediction_model.pkl")
    print("✅ Model saved as 'models/reach_prediction_model.pkl'")

# Run training in background thread
t = threading.Thread(target=train_model)
t.start()

# GUI Setup
root = tk.Tk()
root.title("Ad Spend Optimization")
user_inputs = {}

# Dropdown Options
feature_options = {col: df_original[col].dropna().str.strip().unique().tolist() for col in categorical_columns}

def create_dropdowns():
    row = 0
    for feature, options in feature_options.items():
        label = tk.Label(root, text=f"Select {feature}:")
        label.grid(row=row, column=0, padx=10, pady=5, sticky="e")
        dropdown = ttk.Combobox(root, values=options, state="readonly")
        dropdown.grid(row=row, column=1, padx=10, pady=5)
        user_inputs[feature] = dropdown
        row += 1
create_dropdowns()

# Function for Prediction & Optimization
def on_submit():
    user_data = {feature: dropdown.get() for feature, dropdown in user_inputs.items()}
    input_data = {col: 0 for col in X.columns}
    for feature, value in user_data.items():
        encoded_col = f"{feature}_{value}"
        if encoded_col in X.columns:
            input_data[encoded_col] = 1
    input_data["Investment"] = (1000 - df_original["Investment"].mean()) / df_original["Investment"].std()
    X_sample = pd.DataFrame([input_data], columns=X.columns).fillna(0)
    optimal_spend = differential_evolution(lambda x: -best_model.predict(pd.DataFrame([{**input_data, "Investment": (x[0] - df_original["Investment"].mean()) / df_original["Investment"].std()}], columns=X.columns))[0] / x[0], [(50, 5000)], maxiter=1000).x[0]
    messagebox.showinfo("Optimal Ad Spend", f"Optimal Investment for maximum reach: €{optimal_spend:.2f}")

# Submit Button
submit_button = tk.Button(root, text="Submit", command=on_submit)
submit_button.grid(row=len(feature_options) + 1, column=0, columnspan=2, pady=20)

# Run GUI
root.mainloop()
