import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from Models import text_procesing
# import text_procesing

model = joblib.load("random_for/random_forr.joblib")
tfidf = joblib.load("random_for/tfidf_vectorizer.joblib")


def predict_text(text):
    resultado = text_procesing.clean_text(text)
    resultado = tfidf.transform([resultado])
    prediccion = model.predict(resultado)
    print(prediccion)
    probabilida = model.predict_proba(resultado)
    return probabilida


def match_category(category):
    match category:
        case 0:
            return "exterior"
        case 1:
            return "economia"
        case 2:
            return "opinion"
        case 3:
            return "competencia"
        case 4:
            return "apoyo"
        case 5:
            return "seguridad"


def predict(proba):
    proba = list(proba[0])
    print(proba)
    maxx = max(proba)
    index = proba.index(maxx)
    return f"La probabilidad es {maxx} y lo categoriza como {match_category(index)}"



a = predict(predict_text("colombia"))
print(a)


