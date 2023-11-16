import joblib
import pandas as pd
import plotly.express as px
import text_procesing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

df = text_procesing.return_dataframe()
tfidf = TfidfVectorizer(ngram_range=(1, 3))


def svc_create():
    X = df["Texto_limpio"]
    y = df["cla_num"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.85, random_state=42
    )
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)
    model = SVC(kernel="linear", random_state=30, probability=True)
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)

    report = classification_report(y_test, y_pred, output_dict=True)
    clasification = pd.DataFrame(report).transpose()

    clasification.to_csv("svcc/clasification.csv")
    return model


model = svc_create()


joblib.dump(model, "svcc/svc_create.joblib")
joblib.dump(tfidf, "svcc/tfidf_vectorizer.joblib")
