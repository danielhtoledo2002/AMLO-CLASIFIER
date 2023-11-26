import joblib
import pandas as pd

from Models2 import text_pro

# import text_procesing

model = joblib.load("random_for/Random_forr2.joblib")
tfidf = joblib.load("random_for/Tfidf_vectorizer2.joblib")
clasification = pd.read_csv("random_for/Clasification2.csv")


def predict_text(text):
    resultado = text_pro.clean_text(text)
    resultado = tfidf.transform([resultado])
    prediccion = model.predict(resultado)
    print(prediccion)
    probabilida = model.predict_proba(resultado)
    return probabilida


def match_category(category):
    match category:
        case 0:
            return "apoyo"
        case 1:
            return "competencia"
        case 2:
            return "construcción"
        case 3:
            return "corrupción"
        case 4:
            return "economía"
        case 5:
            return "exterior"
        case 6:
            return "historia"
        case 7:
            return "opinion"
        case 8:
            return "oposición"
        case 9:
            return "salud"
        case 10:
            return "seguridad"
        case _:
            return category


def predict(proba):
    proba = list(proba[0])
    maxx = max(proba)
    index = proba.index(maxx)
    return f"La probabilidad es {maxx} y lo categoriza como {match_category(index)}"


def clasification_rep():
    clasification["Unnamed: 0"] = clasification["Unnamed: 0"].astype(str)    
    a = clasification.rename(columns={"Unnamed: 0": "Clasificación", "precision" : "Precision"})

    return a
