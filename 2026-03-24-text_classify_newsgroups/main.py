# wrote this at 2am lol
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
# found this on stackoverflow
data = fetch_20newsgroups(subset='all')
X = data.data
y = data.target
# idk why but dont touch this
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
vectorizer = TfidfVectorizer(stop_words='english')
X_tr = vectorizer.fit_transform(X_train)
X_te = vectorizer.transform(X_test)
clf = MultinomialNB()
clf.fit(X_tr, y_train)
res = clf.predict(X_te)
print('accuracy:', accuracy_score(y_test, res)) # this should be high
# trying to see what's going on
print('classes:', data.target_names)
print('train shape:', X_tr.shape)
print('test shape:', X_te.shape)
# tmp = pd.DataFrame(X_tr.toarray(), columns=vectorizer.get_feature_names_out())
# print(tmp.head())
# debug print statements
print('num classes:', len(np.unique(y_train)))
acc = accuracy_score(y_test, res)
print('accuracy:', acc)
if acc > 0.8:
    print('yeah we did it')
else:
    print('need to tune hyperparams')
# tune hyperparams maybe?
# from sklearn.model_selection import GridSearchCV
# param_grid = {'alpha': [0.1, 0.5, 1]}
# grid = GridSearchCV(MultinomialNB(), param_grid, cv=5)
# grid.fit(X_tr, y_train)
# print('best params:', grid.best_params_)
# print('best score:', grid.best_score_)