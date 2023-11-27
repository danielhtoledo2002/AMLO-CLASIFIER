# Tool from text processing
import nltk
from nltk.corpus import stopwords

# Tool
import spacy
import pandas as pd

# Regual expressions
import re

df = pd.read_csv("Datasets/amlo_clasify_chatgpt_15k.csv")


stop_words_es = stopwords.words("spanish")


nlp = spacy.load("es_core_news_lg")

from spacy.lang.es.stop_words import STOP_WORDS


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


