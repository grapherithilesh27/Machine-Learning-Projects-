# wrote this at 3am dont @ me
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
# found this on stackoverflow
X_tr, X_te, y_tr, y_te = train_test_split(fetch_20newsgroups(subset='train').data, fetch_20newsgroups(subset='train').target, test_size=0.2, random_state=42)
# idk why but dont touch this
vectorizer = TfidfVectorizer(stop_words='english')
X_tr_tfidf = vectorizer.fit_transform(X_tr)
X_te_tfidf = vectorizer.transform(X_te)
print(X_tr_tfidf.shape)
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_tr_tfidf, y_tr)
preds = clf.predict(X_te_tfidf)
# this works trust me
tmp = clf.score(X_te_tfidf, y_te)
print('accuracy:', tmp)
# leftover debug print
print(clf.feature_importances_)
res = []
for i in range(len(preds)):
    res.append({'text': X_te[i], 'prediction': preds[i], 'actual': y_te[i]})
print(res[:5])
# trying to make it more readable
def predict_text(text):
    text_tfidf = vectorizer.transform([text])
    return clf.predict(text_tfidf)
print(predict_text('this is a test'))