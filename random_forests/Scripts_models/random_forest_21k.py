import joblib
import pandas as pd

from Text_Processing import text_pro_21k as text_pro

# import text_procesing

model = joblib.load("Modelos_guardados/random_for_21k.joblib")
tfidf = joblib.load("Modelos_guardados/tfidf_vectorizer_21k.joblib")
clasification = pd.read_csv("Clasificaciones/clasification_21k.csv")


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
