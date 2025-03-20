import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv("/Users/sahil.pardasani/Desktop/Projects/ImplementingLinearRegression/archive/score_updated.csv")

# Display the first few rows of the dataset
print(data.head())

# Split the data into features (X) and target (y)
X = data[['Hours']]  # Features (independent variable)
y = data['Scores']   # Target (dependent variable)

# Split the data into training and testing sets (optional, but good practice)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the LinearRegression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Print the model's coefficients (slope and intercept)
print("Slope (m):", model.coef_[0])
print("Intercept (b):", model.intercept_)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error (MSE):", mse)

# Visualize the results
plt.scatter(X_test, y_test, color="black", label="Actual Data")
plt.plot(X_test, y_pred, color="green", label="Linear Regression Line")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.title("Linear Regression Using Scikit-learn")
plt.legend()
plt.show()