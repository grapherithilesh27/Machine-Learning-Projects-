# wrote this at 2am lol
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# generate fake fraud data
X, y = make_classification(
    n_samples=1000, n_features=20,
    n_informative=10, n_redundant=5,
    weights=[0.95, 0.05], random_state=42
)

print('Dataset shape:', X.shape)
print('Fraud cases:', sum(y), '/', len(y))

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

# random forest works well for this
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_tr, y_tr)

preds = clf.predict(X_te)
acc = accuracy_score(y_te, preds)
print('Accuracy:', round(acc, 4))
print('Confusion Matrix:')
print(confusion_matrix(y_te, preds))
print('Report:')
print(classification_report(y_te, preds))
print('Done!')