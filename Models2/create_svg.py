import joblib
import pandas as pd
import plotly.express as px
import text_pro
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

df = text_pro.return_dataframe()
tfidf = TfidfVectorizer(ngram_range=(1, 3))

X = df["Texto_limpio"]
y = df["classification_spanish"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.95
)
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)
model = SVC(kernel="linear", probability=True)
model.fit(X_train_vec, y_train)
y_pred = model.predict(X_test_vec)

report = classification_report(y_test, y_pred, output_dict=True)
print(report)
clasification = pd.DataFrame(report).transpose()
clasification.to_csv("svcc/clasification2.csv")




joblib.dump(model, "svcc/svc_create2.joblib")
joblib.dump(tfidf, "svcc/tfidf_vectorizer2.joblib")
