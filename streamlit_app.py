import streamlit as st
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge

st.title("Predicting Weight Change Using Regression - Nyel Uffa")
# Load your dataset and define features and target variable
df = pd.read_csv("weight_change_dataset.csv")
# feature engineering for strong signal
df['Weekly Calorie Balance'] = df['Daily Caloric Surplus/Deficit'] * df['Duration (weeks)']
df = df.drop(columns=['Daily Caloric Surplus/Deficit', 'Duration (weeks)'])
# Drop specific columns that are not needed for regression
X = df.drop(columns=["Final Weight (lbs)"])
y = df["Final Weight (lbs)"]

# Define categorical and numerical columns
categorical_cols = ["Gender", "Physical Activity Level", "Sleep Quality"]
numerical_cols = [col for col in X.columns if col not in categorical_cols]


# One-hot encode categorical variables (for regression)
preprocessor = ColumnTransformer([
    ("onehot", OneHotEncoder(drop='first'), categorical_cols)
], remainder='passthrough')

# Create a pipeline with preprocessing and regression model
pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("regressor", Ridge(alpha=15.0))
])

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(X_test)

# Train and fit the regression model
pipeline.fit(X_train, y_train)

# Predict on the test set
y_pred = pipeline.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# Example: Input your own data for prediction
# You can modify the values below to match your own information

custom_data = pd.DataFrame([{
    'Age': st.number_input("Age"),
    'Gender': st.text_input("Gender (M or F)", max_chars=1),
    'Current Weight (lbs)': st.number_input("Current Weight (lbs)"),
    'BMR (Calories)': st.number_input("BMR (Calories)"),
    'Daily Calories Consumed': st.number_input("Daily Calories Consumed"),
    'Physical Activity Level': st.selectbox(
    'Phyiscal Activity Level',
    ('Sedentary', 'Lightly Active', 'Moderately Active', 'Very Active')),
    'Sleep Quality': st.selectbox(
    'Sleep Quality',
    ('Poor', 'Good', 'Fair', 'Excellent')),
    'Stress Level': st.number_input("Stress Level (1-10)", min_value=1, max_value=10, step=1),
    'Weekly Calorie Balance': (st.number_input("Daily Caloric Surplus/Deficit")) * 7   # Example calculation
}])

# Predict final weight using the trained pipeline
predicted_weight = pipeline.predict(custom_data)
st.write(f"Predicted Final Weight (lbs): {predicted_weight[0]:.2f}")