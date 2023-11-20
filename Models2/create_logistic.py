import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import text_pro

df = text_pro.return_dataframe()

tfidf = TfidfVectorizer(ngram_range=(1, 3))

dfresult = df.dropna()
df = dfresult

X = df["Texto_limpio"]
y = df["classification_spanish"]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.90)
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)
model = LogisticRegression(multi_class="multinomial", solver="sag")
model.fit(X_train_vec, y_train)
y_pred = model.predict(X_test_vec)

# Calculate classification report with zero_division parameter
report = classification_report(y_test, y_pred, output_dict=True, zero_division=1)
classification = pd.DataFrame(report).transpose()

classification.to_csv("Lregresion/clasification2.csv")

joblib.dump(model, "Lregresion/LogisticRegresion2.joblib")
joblib.dump(tfidf, "Lregresion/tfidf_vectorizer2.joblib")
