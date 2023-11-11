from Models import text_procesing
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


model = joblib.load("Models/LogisticRegresion.joblib")
tfidf = joblib.load("Models/tfidf_vectorizer.joblib")
def predict_text(text):
    resultado = text_procesing.clean_text(text)
    resultado = tfidf.transform([resultado])
    prediccion = model.predict(resultado)
    probabilida = model.predict_proba(resultado)
    return probabilida
a = predict_text("DInero")

def match_category(category):
    match category:
        case 0: return "exterior"
        case 1: return "economia"
        case 2 : return "opinion"
        case 3 : return "competencia"
        case 4: return "apoyo"
        case 5: return "seguridad"

def predict(proba):
    proba = list(proba[0])
    maxx = max(proba)
    index = proba.index(maxx)
    return f"La probabilidad es {maxx} y lo categoriza como {match_category(index)}"


   

print(predict(a))
