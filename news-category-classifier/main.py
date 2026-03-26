# wrote this at 2am lol
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
# found this on stackoverflow
newsgroups_train = fetch_20newsgroups(subset='train')
newsgroups_test = fetch_20newsgroups(subset='test')
# idk why but dont touch this
X_tr = newsgroups_train.data
y_tr = newsgroups_train.target
X_te = newsgroups_test.data
y_te = newsgroups_test.target
# this works trust me
tfidf = TfidfVectorizer()
X_tr_tfidf = tfidf.fit_transform(X_tr)
X_te_tfidf = tfidf.transform(X_te)
clf = MultinomialNB()
clf.fit(X_tr_tfidf, y_tr)
pred = clf.predict(X_te_tfidf)
print('accuracy:', accuracy_score(y_te, pred)) # debug print
res = accuracy_score(y_te, pred)
# print(res) # debug print
print('done') # debug print
# leftover debug code
# tmp = X_tr_tfidf.toarray()
# print(tmp)
