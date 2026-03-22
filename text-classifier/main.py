# ML assignment
# wrote this at 3am lol
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
# this works trust me
news = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
# found this on stackoverflow
X_tr, X_te, y_tr, y_te = train_test_split(news.data, news.target, test_size=0.2, random_state=42)
print('data loaded')
# idk why but dont touch this
vectorizer = TfidfVectorizer(stop_words='english')
# works fine
X_tr_tfidf = vectorizer.fit_transform(X_tr)
X_te_tfidf = vectorizer.transform(X_te)
print('data transformed')
# trying out different models
clf = MultinomialNB()
# this is a pipeline
pipe = Pipeline([('vect', vectorizer), ('clf', clf)])
print('pipeline created')
# grid search
param_grid = {'vect__max_df': [0.5, 0.75, 1.0], 'clf__alpha': [0.1, 0.5, 1.0]}
grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(X_tr, y_tr)
print('grid search done')
# get best model
best_model = grid.best_estimator_
# evaluate
y_pred = best_model.predict(X_te)
print('predictions made')
acc = accuracy_score(y_te, y_pred)
print(f'accuracy: {acc:.3f}')
print(classification_report(y_te, y_pred))
# debug print
print('debug: done')
