from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


import pandas as pd
import plotly.express as px
import joblib
import text_procesing

df  =  text_procesing.return_dataframe()
tfidf = TfidfVectorizer(ngram_range=(1, 5))


X = df['Texto_limpio']
y = df['cla_num']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.85)
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)
model = RandomForestClassifier(n_jobs=-1, criterion="entropy")
model.fit(X_train_vec, y_train)
y_pred = model.predict(X_test_vec)

print(classification_report(y_test, y_pred))
