# wrote this at 2am lol
# trying to classify news articles into 20 categories
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# get the dataset
newsgroups = fetch_20newsgroups(remove=('headers', 'footers', 'quotes'))
X = newsgroups.data
y = newsgroups.target
# print(y)

# split into train and test sets
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
# print(len(X_tr))

# vectorize the text data
vec = TfidfVectorizer(stop_words='english')
X_tr_vec = vec.fit_transform(X_tr)
X_te_vec = vec.transform(X_te)
# print(X_tr_vec.shape)

# train a random forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_tr_vec, y_tr)
# print('trained')

# make predictions on the test set
y_pred = clf.predict(X_te_vec)
# print(y_pred)

# evaluate the model
acc = accuracy_score(y_te, y_pred)
print('accuracy:', acc)
print(classification_report(y_te, y_pred))
# found this on stackoverflow
# idk why but dont touch this
print('done')