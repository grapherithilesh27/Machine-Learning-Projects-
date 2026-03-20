# Import necessary libraries
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv('music_data.csv')

# Preprocess the data
data['genre'] = data['genre'].apply(lambda x: x.lower())
data['artist'] = data['artist'].apply(lambda x: x.lower())

# Split the data into training and testing sets
X = data.drop(['rating'], axis=1)
Y = data['rating']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Create the model
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
model.fit(X_train, Y_train, epochs=10, batch_size=32, verbose=2)

# Evaluate the model
predictions = model.predict(X_test)
predictions = np.round(predictions)
print('Model Accuracy: ', accuracy_score(Y_test, predictions))

# Use the model to make predictions
new_song = pd.DataFrame({'genre': ['rock'], 'artist': ['the beatles']})
prediction = model.predict(new_song)
print('Predicted Rating: ', prediction)