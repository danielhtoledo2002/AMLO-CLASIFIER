import joblib
import pandas as pd

from Text_Processing import text_pro_humano as text_procesing

# import text_procesing

model = joblib.load("Modelos_guardados/random_for_humano.joblib")
tfidf = joblib.load("Modelos_guardados/tfidf_vectorizer_humano.joblib")
clasification = pd.read_csv("Clasificaciones/clasification_humano.csv")


def predict_text(text):
    resultado = text_procesing.clean_text(text)
    resultado = tfidf.transform([resultado])
    prediccion = model.predict(resultado)
    print(prediccion)
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


def match_category2(category):
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
    return f"La probabilidad es {maxx} y lo categoriza como {match_category2(index)}"


def clasification_rep():
    clasification["Unnamed: 0"] = clasification["Unnamed: 0"].astype(str)

    clasification["Unnamed: 0"] = clasification["Unnamed: 0"].apply(match_category)
    a = clasification.rename(columns={"Unnamed: 0": "Clasificación", "precision" : "Precision"})

    return a
