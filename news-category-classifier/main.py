# wrote this at 3am lol
# importing the necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report
# loading the dataset
news = fetch_20newsgroups(subset='all')
# printing the categories
print(news.target_names)
# splitting the data into training and testing sets
X_tr, X_te, y_tr, y_te = train_test_split(news.data, news.target, test_size=0.2, random_state=42)
# creating a tfidf vectorizer
vec = TfidfVectorizer(stop_words='english')
# fitting the vectorizer to the training data and transforming both sets
X_tr = vec.fit_transform(X_tr)
X_te = vec.transform(X_te)
# printing the shape of the data
print(X_tr.shape)
print(X_te.shape)
# creating an svm classifier
clf = svm.SVC(kernel='linear', probability=True)
# training the model
clf.fit(X_tr, y_tr)
# making predictions
y_pred = clf.predict(X_te)
# printing the accuracy
acc = accuracy_score(y_te, y_pred)
print('accuracy: ', acc)
# printing the classification report
print(classification_report(y_te, y_pred))
# found this on stackoverflow
from sklearn.model_selection import GridSearchCV
# defining the hyperparameter space
param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf', 'poly']}
# performing grid search
grid = GridSearchCV(svm.SVC(probability=True), param_grid, cv=5)
# fitting the grid search object to the training data
grid.fit(X_tr, y_tr)
# printing the best parameters and the best score
print('best params: ', grid.best_params_)
print('best score: ', grid.best_score_)
# idk why but dont touch this
res = grid.predict(X_te)
# printing the accuracy of the grid search model
print('grid search accuracy: ', accuracy_score(y_te, res))