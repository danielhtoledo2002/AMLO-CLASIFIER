from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pandas as pd
import joblib
import text_pro

df = text_pro.return_dataframe()
tfidf = TfidfVectorizer(ngram_range=(1, 3))

X = df["Texto_limpio"]
y = df["classification_spanish"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.85, random_state=42
)
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)

model = RandomForestClassifier(n_jobs=-1, criterion="entropy", n_estimators= 400)
model.fit(X_train_vec, y_train)
y_pred = model.predict(X_test_vec)
report = classification_report(y_test, y_pred, output_dict=True)
clasification = pd.DataFrame(report).transpose()

clasification.to_csv("random_for/Clasification2.csv")



print(classification_report(y_test, y_pred))
joblib.dump(model, "random_for/Random_forr2.joblib")
joblib.dump(tfidf, "random_for/Tfidf_vectorizer2.joblib")
