# wrote this at 3am dont judge me
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
# found this on stackoverflow
X_tr, X_te, y_tr, y_te = train_test_split(20newsgroups.data, 20newsgroups.target, test_size=0.2, random_state=42)
# this works trust me
categories = 20newsgroups.target_names
print(categories)
# idk why but dont touch this
vectorizer = TfidfVectorizer(stop_words='english')
X_tr_vec = vectorizer.fit_transform(X_tr)
X_te_vec = vectorizer.transform(X_te)
print(X_tr_vec.shape)
# leftover debug print stmts
print(y_tr)
print(y_te)
clf = MultinomialNB()
clf.fit(X_tr_vec, y_tr)
res = clf.predict(X_te_vec)
print(res)
print(accuracy_score(y_te, res))
print(classification_report(y_te, res))
tmp = np.array(y_te).reshape(-1, 1)
print(tmp.shape)
# more leftover debug print stmts
print(X_te_vec.toarray())
print(y_tr)
print(y_te)