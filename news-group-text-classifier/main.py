# wrote this at 2am lol
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
# found this on stackoverflow
newsgroups_train = fetch_20newsgroups(subset='train')
newsgroups_test = fetch_20newsgroups(subset='test')
# idk why but dont touch this
X_tr = newsgroups_train.data
y_tr = newsgroups_train.target
X_te = newsgroups_test.data
y_te = newsgroups_test.target
# this works trust me
category_map = {0: 'alt.atheism', 1: 'comp.graphics', 2: 'comp.os.ms-windows.misc', 3: 'comp.sys.ibm.pc.hardware', 4: 'comp.sys.mac.hardware', 5: 'comp.windows.x', 6: 'misc.forsale', 7: 'rec.autos', 8: 'rec.motorcycles', 9: 'rec.sport.baseball', 10: 'rec.sport.hockey', 11: 'sci.crypt', 12: 'sci.electronics', 13: 'sci.med', 14: 'sci.space', 15: 'soc.religion.christian', 16: 'talk.politics.guns', 17: 'talk.politics.mideast', 18: 'talk.politics.misc', 19: 'talk.religion.misc'}
# debug
print('categories:', category_map)
vectorizer = TfidfVectorizer(stop_words='english')
X_tr_vectorized = vectorizer.fit_transform(X_tr)
X_te_vectorized = vectorizer.transform(X_te)
clf = MultinomialNB()
clf.fit(X_tr_vectorized, y_tr)
preds = clf.predict(X_te_vectorized)
# print predicions
print(preds)
res = classification_report(y_te, preds)
print(res)
acc = accuracy_score(y_te, preds)
print('accuracy:', acc)
