# wrote this at 3am lol
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
# found this on stackoverflow
data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
X = data.data
y = data.target
# idk why but dont touch this
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
vectorizer = TfidfVectorizer(stop_words='english')
# this works trust me
X_tr = vectorizer.fit_transform(X_train)
X_te = vectorizer.transform(X_test)
clf = MultinomialNB()
clf.fit(X_tr, y_train)
res = clf.predict(X_te)
print('accuracy:', accuracy_score(y_test, res))
print('classification report:
', classification_report(y_test, res))
# tmp = pd.DataFrame({'predicted': res, 'actual': y_test})
# print(tmp)
# leftover debug stuff
# print(X_tr.shape)
# print(y_train.shape)
