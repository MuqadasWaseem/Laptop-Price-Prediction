import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import openpyxl

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# Load dataset
df = pd.read_excel(r"C:\Users\Lenovo\OneDrive\Desktop\laptop price prediction\laptop_price_data.xlsx")


# Display first 5 rows
print(df.head())


# Dataset information
print("\nDataset Shape:", df.shape)


# Missing values
print("\nMissing Values:")
print(df.isnull().sum())


# Features
X = df[[
    "RAM_GB",
    "Storage_GB",
    "Screen_Size_Inch",
    "Brand",
    "Processor",
    "Condition"
]]


# Target
y = df["Price"]


# Convert categorical columns into numerical columns
X = pd.get_dummies(
    X,
    columns=["Brand", "Processor", "Condition"],
    drop_first=True,
    dtype=int
)


# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Create model
model = LinearRegression()


# Train model
model.fit(X_train, y_train)


# Predictions
y_pred = model.predict(X_test)


# Evaluation
r2 = r2_score(y_test, y_pred)

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# ridge_model = Ridge(alpha=10)

# ridge_model.fit(X_train, y_train)

# ridge_pred = ridge_model.predict(X_test)


# ridge_r2 = r2_score(y_test, ridge_pred)

# ridge_mae = mean_absolute_error(y_test, ridge_pred)

# ridge_rmse = np.sqrt(
#     mean_squared_error(y_test, ridge_pred)
# )


# print("\nRidge Regression")


# Ridge Regression - Alpha Tuning

alphas = [0.01, 0.1, 1, 10, 100]

best_alpha = None
best_r2 = -float("inf")

print("\nRidge Regression Alpha Tuning")

for alpha in alphas:

    ridge_model = Ridge(alpha=alpha)

    ridge_model.fit(X_train, y_train)

    ridge_pred = ridge_model.predict(X_test)

    ridge_r2 = r2_score(y_test, ridge_pred)

    print("Alpha:", alpha, "| R² Score:", ridge_r2)

    if ridge_r2 > best_r2:
        best_r2 = ridge_r2
        best_alpha = alpha


print("\nBest Ridge Model")
print("Best Alpha:", best_alpha)
print("Best R² Score:", best_r2)

final_ridge = Ridge(alpha=best_alpha)

final_ridge.fit(X_train, y_train)

print("\nModel Performance")
print("R² Score:", r2)
print("MAE:", mae)
print("RMSE:", rmse)

print("\nModel Comparison")

print("Linear Regression R²:",r2)
print("Ridge Regression R²:", best_r2)

# New laptop
new_laptop = pd.DataFrame([{
    "RAM_GB": 16,
    "Storage_GB": 512,
    "Screen_Size_Inch": 15.6,
    "Brand": "Dell",
    "Processor": "Core i7",
    "Condition": "New"
}])


# Apply same encoding
new_laptop = pd.get_dummies(
    new_laptop,
    columns=["Brand", "Processor", "Condition"],
    drop_first=True,
    dtype=int
)


# Make sure new laptop has exactly the same columns as training data
new_laptop = new_laptop.reindex(
    columns=X.columns,
    fill_value=0
)

# Predict new laptop price using Final Ridge Model
prediction = final_ridge.predict(new_laptop)

print("\nFinal Predicted Laptop Price:", prediction[0])

# Predict price
# prediction = model.predict(new_laptop)


# print("\nPredicted Laptop Price:", prediction[0])
# Final Ridge Performance

final_pred = final_ridge.predict(X_test)

final_r2 = r2_score(y_test, final_pred)

final_mae = mean_absolute_error(y_test, final_pred)

final_rmse = np.sqrt(
    mean_squared_error(y_test, final_pred)
)

print("\nFinal Ridge Model Performance")
print("R² Score:", final_r2)
print("MAE:", final_mae)
print("RMSE:", final_rmse)

# Actual vs Predicted
plt.scatter(y_test, y_pred)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual Price vs Predicted Price")

plt.show()
