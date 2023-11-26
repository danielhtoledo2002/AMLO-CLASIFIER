import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

 from Models2 import text_pro

# import text_pro
model = joblib.load("random_for/random_forr3.joblib")
tfidf = joblib.load("random_for/tfidf_vectorizer3.joblib")
clasification = pd.read_csv("random_for/clasification3.csv")





def predict_text(text):
    resultado = text_pro.clean_text(text)
    resultado = tfidf.transform([resultado])
    prediccion = model.predict(resultado)
    probabilida = model.predict_proba(resultado)
    return probabilida


def match_category(category):
    match category:
        case "0":
            return "apoyo"
        case "1":
            return "competencia"
        case "2":
            return "construccion"
        case "3":
            return "corrupcion"
        case "4":
            return "economia"
        case "5":
            return "exterior"
        case "6":
            return "historia"
        case "7":
            return "opinion"
        case "8":
            return "salud"
        case "9":
            return "seguridad"
        case _:
            return category


def match_category2(category):
    match category:
        case 0:
            return "apoyo"
        case 1:
            return "competencia"
        case 2:
            return "construccion"
        case 3:
            return "corrupcion"
        case 4:
            return "economia"
        case 5:
            return "exterior"
        case 6:
            return "historia"
        case 7:
            return "opinion"
        case 8:
            return "salud"
        case 9:
            return "seguridad"


def predict(proba):
    proba = list(proba[0])
    maxx = max(proba)
    index = proba.index(maxx)
    return f"La probabilidad es {maxx} y lo categoriza como {match_category2(index)}"


def clasification_rep():
    a = clasification.rename(
        columns={"Unnamed: 0": "Clasificación", "precision": "Precision"}
    )
    return a

a = predict_text("dinero")

print(predict(a))

print(clasification_rep())
