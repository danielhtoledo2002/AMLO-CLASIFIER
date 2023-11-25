# Tool from text processing
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# Tool
import spacy
import pandas as pd

# Regual expressions
import re

df = pd.read_csv("amlo_clasify_chatpgt3.csv")


stop_words_es = stopwords.words("spanish")


nlp = spacy.load("es_core_news_lg")

from spacy.lang.es.stop_words import STOP_WORDS

print(df.head())


def return_dataframe():
    df["Texto_limpio"] = df["Texto"].apply(clean_text)
    return df


# fn to clean text


def clean_text(texto):
    textofin = texto.lower()
    textofin = re.sub(
        r"([^0-9A-Za-z-À-ÿ \t])",
        "",
        textofin,
    )
    textofin = nlp(textofin)
    lema = []
    for token in textofin:
        lema.append(token.lemma_)
    textofin = lema
    textofin = [palabra for palabra in textofin if palabra not in STOP_WORDS]
    textofin = " ".join(textofin)
    return textofin


def clasification_to_num(text):
    if text == "apoyo":
        return 0
    elif text == "competencia" or text=="oposicion":
        return 1
    elif text == "construccion":
        return 2
    elif text == "corrupcion":
        return 3
    elif text == "economia":
        return 4
    elif text == "exterior":
        return 5
    elif text == "historia":
        return 6
    elif text == "opinion":
        return 7
    elif text == "salud":
        return 8
    elif text == "seguridad":
        return 9