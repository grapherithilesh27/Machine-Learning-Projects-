# wrote this at 2am lol
# found this on stackoverflow
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd

# load the dataset
newsgroups = fetch_20newsgroups(remove=('headers', 'footers', 'quotes'))
X = newsgroups.data
y = newsgroups.target

# split the data into training and testing sets
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

# create a tf-idf vectorizer
vectorizer = TfidfVectorizer(stop_words='english')
# fit and transform the data
X_tr_tfidf = vectorizer.fit_transform(X_tr)
X_te_tfidf = vectorizer.transform(X_te)

# idk why but dont touch this
clf = MultinomialNB()
# train the model
clf.fit(X_tr_tfidf, y_tr)

# make predictions
y_pred = clf.predict(X_te_tfidf)

# print the accuracy
print('Accuracy:', accuracy_score(y_te, y_pred))
print('Classification Report:
', classification_report(y_te, y_pred))

# found this on youtube
param_grid = {'alpha': [0.1, 0.5, 1.0]}
grid = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5)
grid_result = grid.fit(X_tr_tfidf, y_tr)
print('Best Parameters:', grid_result.best_params_)
print('Best Score:', grid_result.best_score_)

# leftover debug print
print(y_pred)
