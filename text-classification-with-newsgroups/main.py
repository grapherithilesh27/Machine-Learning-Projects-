# wrote this at 3am lol
# import necessary libs
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# load dataset
newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
X = newsgroups.data
y = newsgroups.target

# split into train and test sets
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

# create a tf-idf vectorizer
vectorizer = TfidfVectorizer(stop_words='english')
# fit and transform the data
X_tr_vec = vectorizer.fit_transform(X_tr)
X_te_vec = vectorizer.transform(X_te)

# train a random forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_tr_vec, y_tr)

# make predictions
y_pred = clf.predict(X_te_vec)

# print the classification report
print(classification_report(y_te, y_pred))
print('accuracy:', accuracy_score(y_te, y_pred))

# just some leftover debug prints
print('train shape:', X_tr_vec.shape)
print('test shape:', X_te_vec.shape)
# idk why but dont touch this
print(np.unique(y_tr))

# found this on stackoverflow, works for me
# https://stackoverflow.com/questions/26414913/normalizing-the-vector-features
# this works trust me