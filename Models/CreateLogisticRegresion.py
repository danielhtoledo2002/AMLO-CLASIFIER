import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import text_procesing
from sklearn.metrics import classification_report, confusion_matrix

df = text_procesing.return_dataframe()

tfidf = TfidfVectorizer(ngram_range=(1, 3))


def logistic_regresion():
    X = df["Texto_limpio"]
    y = df["cla_num"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.85)
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)
    model = LogisticRegression(multi_class="multinomial", solver="saga", max_iter=500)
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    report = classification_report(y_test, y_pred, output_dict=True)
    clasification = pd.DataFrame(report).transpose()

    clasification.to_csv("Lregresion/clasification.csv")

    return model
    # return classification_report(y_test, y_pred)


model = logistic_regresion()

joblib.dump(model, "Lregresion/LogisticRegresion.joblib")
joblib.dump(tfidf, "Lregresion/tfidf_vectorizer.joblib")
