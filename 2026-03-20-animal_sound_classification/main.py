# Import necessary libraries
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import librosa
import numpy as np

# Load dataset
def load_dataset(file_path):
    sounds = {}
    for file in os.listdir(file_path):
        sound, _ = librosa.load(os.path.join(file_path, file))
        sounds[file] = sound
    return sounds

# Preprocess data
def preprocess_data(sounds):
    preprocessed_sounds = {}
    for file, sound in sounds.items():
        sound = librosa.resample(sound, 22050, 8000)
        sound = librosa.feature.melspectrogram(y=sound, sr=8000, n_mels=128)
        preprocessed_sounds[file] = sound
    return preprocessed_sounds

# Define model architecture
def create_model(input_shape):
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation='softmax')
    ])
    return model

# Train model
def train_model(model, X_train, y_train):
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=32)

# Evaluate model
def evaluate_model(model, X_test, y_test):
    loss, accuracy = model.evaluate(X_test, y_test)
    return accuracy

# Main function
if __name__ == '__main__':
    # Load dataset
    dataset = load_dataset('animal_sounds')
    sounds = list(dataset.values())
    labels = list(dataset.keys())

    # Preprocess data
    preprocessed_sounds = preprocess_data(dataset)
    X = np.array(list(preprocessed_sounds.values()))
    y = np.array(labels)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # One-hot encode labels
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)

    # Define model architecture
    input_shape = (128, None, 1)
    model = create_model(input_shape)

    # Train model
    train_model(model, X_train, y_train)

    # Evaluate model
    accuracy = evaluate_model(model, X_test, y_test)
    print(f'Model accuracy: {accuracy:.2f}')