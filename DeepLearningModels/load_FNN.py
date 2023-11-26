from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras
import numpy as np
import pandas as pd
import re
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.preprocessing import MinMaxScaler
import joblib
import spacy
nltk.download('punkt')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
nlp = spacy.load("es_core_news_lg")
stopword_es = stopwords.words("spanish")

new_model = keras.models.load_model('DeepLearningModels/FNN_model.keras')
scaler = joblib.load('DeepLearningModels/scaler_FNN.save')
clasification = pd.read_csv("score_DLModels/clasificationFNN.csv")

def vectorize(text):
    texto = text.lower()
    texto = re.sub(r'(@[A-Za-z0-9]+)', '', texto)
    texto = re.sub(r'([^0-9A-Za-z \t])', '', texto)
    texto = re.sub(r'(\w+:\/\/\S+)', '', texto)
    texto = re.sub(r'rt', '', texto)
    texto = word_tokenize(texto)
    texto = [palabra for palabra in texto if palabra not in stopword_es]
    texto = [lemmatizer.lemmatize(palabra) for palabra in texto]
    texto = ' '.join(texto)
    texto = nlp(texto).vector
    return texto

def predict_text(texto):
  texto = vectorize(texto)
  texto = texto.reshape(1,-1)
  texto = scaler.transform(texto)
  #texto = texto.reshape(1,300,1)
  #texto = new_model.predict(texto)
  probabilida = new_model.predict(texto)

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
    clasification["Unnamed: 0"] = clasification["Unnamed: 0"].astype(str)

    clasification["Unnamed: 0"] = clasification["Unnamed: 0"].apply(match_category)
    a = clasification.rename(columns={"Unnamed: 0": "Clasificación", "precision" : "Precision"})

    return a