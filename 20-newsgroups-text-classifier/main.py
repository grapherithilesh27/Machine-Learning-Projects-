# wrote this at 3am lol
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# found this on stackoverflow
vectorizer = TfidfVectorizer(stop_words='english')
# load the dataset
.categories = fetch_20newsgroups(subset='all').target_names
X_tr, X_te, y_tr, y_te = train_test_split(fetch_20newsgroups(subset='all').data, fetch_20newsgroups(subset='all').target, test_size=0.2, random_state=42)
# idk why but dont touch this
X_tr = vectorizer.fit_transform(X_tr)
X_te = vectorizer.transform(X_te)
print(X_tr.shape)
# this works trust me
classifier = svm.SVC(kernel='linear', probability=True)
# train the model
classifier.fit(X_tr, y_tr)
# make predictions
y_pred = classifier.predict(X_te)
print('accuracy:', accuracy_score(y_te, y_pred))
# leftover debug print
print(y_pred)
# test the model with a custom text
custom_text = ['this is a sample text about sports']
custom_text = vectorizer.transform(custom_text)
print('predicted category:', .categories[classifier.predict(custom_text)[0]])
# more debug prints
print(classifier.support_)
print(classifier.support_vectors_)
