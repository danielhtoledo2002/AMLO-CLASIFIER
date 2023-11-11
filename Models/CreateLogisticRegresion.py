import text_procesing

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib


df  = text_procesing.return_dataframe()

tfidf = TfidfVectorizer(ngram_range=(1, 3))

def logistic_regresion():
    X = df['Texto_limpio']
    y = df['cla_num']
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.85)
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)
    model = LogisticRegression(multi_class='multinomial', solver='saga', max_iter=1000)
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    return model 
    #return classification_report(y_test, y_pred)

model = logistic_regresion()

joblib.dump(model, "Lregresion/LogisticRegresion.joblib")
joblib.dump(tfidf, "Lregresion/tfidf_vectorizer.joblib")
