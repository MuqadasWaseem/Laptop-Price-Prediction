import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# Load dataset
df = pd.read_excel("laptop_price_data.xlsx")

# Display first 5 rows
print(df.head())

# Dataset information
print("\nDataset Shape:", df.shape)

print("\nMissing Values:")
print(df.isnull().sum())


# Features
X = df[[
    "RAM_GB",
    "Storage_GB",
    "Screen_Size_Inch"
]]

# Target
y = df["Price"]


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


print("\nModel Performance")
print("R² Score:", r2)
print("MAE:", mae)
print("RMSE:", rmse)


# Predict new laptop price
new_laptop = np.array([[16, 512, 15.6]])

prediction = model.predict(new_laptop)

print("\nPredicted Laptop Price:", prediction[0])


# Actual vs Predicted
plt.scatter(y_test, y_pred)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual Price vs Predicted Price")

plt.show()