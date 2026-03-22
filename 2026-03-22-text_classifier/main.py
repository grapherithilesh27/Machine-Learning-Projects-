# ML assignment
# wrote this at 3am lol
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# found this on stackoverflow
X = fetch_20newsgroups(subset='train').data
y = fetch_20newsgroups(subset='train').target
# idk why but dont touch this
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
vectorizer = TfidfVectorizer(stop_words='english')
X_tr_tfidf = vectorizer.fit_transform(X_tr)
X_te_tfidf = vectorizer.transform(X_te)
# this works trust me
cclf = svm.SVC(kernel='linear', probability=True)
clf = clf.fit(X_tr_tfidf, y_tr)
preds = clf.predict(X_te_tfidf)
# leftover debug print
print('accuracy:', accuracy_score(y_te, preds))
res = accuracy_score(y_te, preds)
print(res)
# tmp = clf.predict(vectorizer.transform(['this is a test news article']))
# print(tmp)
from sklearn.model_selection import GridSearchCV
param_grid = {'C': [1, 10, 100], 'gamma': [1, 0.1, 0.01]}
grid = GridSearchCV(svm.SVC(kernel='linear', probability=True), param_grid, cv=5)
grid.fit(X_tr_tfidf, y_tr)
print('best params:', grid.best_params_)
print('best score:', grid.best_score_)
best_clf = grid.best_estimator_
preds = best_clf.predict(X_te_tfidf)
print('accuracy with best params:', accuracy_score(y_te, preds))