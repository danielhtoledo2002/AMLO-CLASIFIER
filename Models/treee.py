import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from Models import text_procesing
# import text_procesing

model = joblib.load("tree/treee.joblib")
tfidf = joblib.load("tree/tfidf_vectorizer.joblib")


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
    maxx = max(proba)
    index = proba.index(maxx)
    return f"La probabilidad es {maxx} y lo categoriza como {match_category(index)}"



a = predict(predict_text("el apoyo de amlo"))
print(a)

