import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from Models2 import text_pro

df = text_pro.return_dataframe()
tfidf = TfidfVectorizer(ngram_range=(1, 3))

X = df["Texto_limpio"]
y = df["classification_spanish"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.85, random_state=42
)
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)

model = DecisionTreeClassifier(criterion="gini", random_state=60, splitter="best")
model.fit(X_train_vec, y_train)
y_pred = model.predict(X_test_vec)
report = classification_report(y_test, y_pred, output_dict=True)
clasification = pd.DataFrame(report).transpose()

clasification.to_csv("tree/Clasification2.csv")

joblib.dump(model, "tree/Tree2.joblib")
joblib.dump(tfidf, "tree/Tfidf_vectorizer2.joblib")