# wrote this at 3am lol
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist
# found this on stackoverflow
(X_tr, y_tr), (X_te, y_te) = mnist.load_data()
# idk why but dont touch this
X_tr = X_tr.reshape(-1, 28, 28, 1).astype('float32') / 255
X_te = X_te.reshape(-1, 28, 28, 1).astype('float32') / 255
# this works trust me
X_tr, X_val, y_tr, y_val = train_test_split(X_tr, y_tr, test_size=0.2, random_state=42)
print(X_tr.shape)
print(y_tr.shape)
# leftover debug print
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_tr, y_tr, epochs=5, validation_data=(X_val, y_val))
loss, acc = model.evaluate(X_te, y_te)
print(f'Test accuracy: {acc:.2f}')
print('done')