from Models import text_procesing
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

model = joblib.load("Lregresion/LogisticRegresion.joblib")
tfidf = joblib.load("Lregresion/tfidf_vectorizer.joblib")

clasification = pd.read_csv("Lregresion/clasification.csv")


def predict_text(text):
    resultado = text_procesing.clean_text(text)
    resultado = tfidf.transform([resultado])
    prediccion = model.predict(resultado)
    probabilida = model.predict_proba(resultado)
    return probabilida


def match_category(category):
    match category:
        case "0":
            return "exterior"
        case "1":
            return "economia"
        case "2":
            return "opinion"
        case "3":
            return "competencia"
        case "4":
            return "apoyo"
        case "5":
            return "seguridad"
        case _:
            return category


def predict(proba):
    proba = list(proba[0])
    maxx = max(proba)
    index = proba.index(maxx)
    return f"La probabilidad es {maxx} y lo categoriza como {match_category(index)}"


def clasification_rep():
    print(clasification.info())
    clasification["Unnamed: 0"] = clasification["Unnamed: 0"].astype(str)

    clasification["Unnamed: 0"] = clasification["Unnamed: 0"].apply(match_category)
    a = clasification.rename(columns={"Unnamed: 0": "Clasificación"})

    return a
