import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# grab the iris dataset - classic beginner dataset
data = load_iris()
X = data.data
y = data.target
feature_names = data.feature_names
target_names = data.target_names

# convert to dataframe so its easier to look at
df = pd.DataFrame(X, columns=feature_names)
df['species'] = y
print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nClass distribution:")
print(df['species'].value_counts())

# split it 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTraining samples: {len(X_train)}, Test samples: {len(X_test)}")

# train random forest - 100 trees usually works well
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)
print("\nModel trained!")

# see how we did
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
print(f"\nAccuracy: {acc:.4f} ({acc*100:.2f}%)")
print("\nDetailed Report:")
print(classification_report(y_test, preds, target_names=target_names))

# feature importance - tells us which measurements matter most
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
print("\nFeature Importances:")
for i in range(X.shape[1]):
    print(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

# test on a new sample
new_sample = np.array([[5.1, 3.5, 1.4, 0.2]])
prediction = model.predict(new_sample)
print(f"\nNew sample prediction: {target_names[prediction[0]]}")
print("Done!")