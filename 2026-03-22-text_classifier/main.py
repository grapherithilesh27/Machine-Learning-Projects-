# wrote this at 3am lol
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
# found this on stackoverflow
data = fetch_20newsgroups(remove=('headers', 'footers', 'quotes'))
# print(data.target_names)
X = data.data
y = data.target
# idk why but dont touch this
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
vectorizer = TfidfVectorizer(stop_words='english')
# this works trust me
X_tr_tfidf = vectorizer.fit_transform(X_tr)
X_te_tfidf = vectorizer.transform(X_te)
clf = MultinomialNB()
clf.fit(X_tr_tfidf, y_tr)
y_pred = clf.predict(X_te_tfidf)
# leftover debug print
print(y_pred)
acc = accuracy_score(y_te, y_pred)
print('accuracy:', acc)
# trying out different classifiers
from sklearn.ensemble import RandomForestClassifier
# this is slower but more accurate
clf2 = RandomForestClassifier(n_estimators=100)
clf2.fit(X_tr_tfidf, y_tr)
y_pred2 = clf2.predict(X_te_tfidf)
acc2 = accuracy_score(y_te, y_pred2)
print('random forest accuracy:', acc2)
