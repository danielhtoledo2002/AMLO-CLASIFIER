# Tool from text processing
import nltk
from nltk.corpus import stopwords
from spacy.lang.es.stop_words import STOP_WORDS

# Tool
import spacy
import pandas as pd

# Regual expressions
import re

df = pd.read_csv("Datasets/amlo_clasify_humano.csv")

stop_words_es = stopwords.words("spanish")


nlp = spacy.load("es_core_news_lg")


def return_dataframe():
    df["cla_num"] = df["Clasificacion"].apply(clasification_to_num)
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
    if text == "exterior":
        return 0
    elif text == "economia":
        return 1
    elif text == "opinion":
        return 2
    elif text == "competencia":
        return 3
    elif text == "apoyo":
        return 4
    elif text == "seguridad":
        return 5
