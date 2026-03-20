# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load dataset
traffic_data = pd.read_csv('traffic_data.csv')

# Explore dataset
print(traffic_data.head())
print(traffic_data.info())

# Preprocess dataset
traffic_data['date'] = pd.to_datetime(traffic_data['date'])
traffic_data['hour'] = traffic_data['date'].dt.hour
traffic_data['day_of_week'] = traffic_data['date'].dt.dayofweek

# Define features and target
X = traffic_data[['hour', 'day_of_week', 'weather', 'road_conditions']]
Y = traffic_data['traffic_flow']

# Split dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, Y_train)

# Make predictions
Y_pred = model.predict(X_test)

# Evaluate model
mse = mean_squared_error(Y_test, Y_pred)
print(f'Mean Squared Error: {mse:.2f}')

# Use model for prediction
new_data = pd.DataFrame({'hour': [8], 'day_of_week': [3], 'weather': ['sunny'], 'road_conditions': ['normal']})
new_prediction = model.predict(new_data)
print(f'Predicted Traffic Flow: {new_prediction[0]:.2f}')