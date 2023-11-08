# file to experiment
import seaborn as sns
import matplotlib.pyplot as plt
import re
import nltk
import numpy as np
import pandas as pd
import spacy
from nltk.corpus import stopwords
from spacy.lang.es.examples import sentences

nlp = spacy.load("es_core_news_sm")
df = pd.read_csv("amlo.csv")
spanish_stopwords = stopwords.words("spanish")

# code to delete information that doesn't have enought words
# df = df[df["Texto"].str.split().str.len() >= 15 ]
# df.to_csv("amlo.csv", index=False)

# experiments
def text_processing(text):
    text = text.lower()
    text = re.sub(r"(@[A-Za-z0-9_]+)|([^0-9A-Za-zÁáÉéÍíÓóÚúÜüÑñ \t])|(\w+:\/\/\S+)", "", text)
    return text

df["Texto"] = df["Texto"].apply(text_processing)

df['Clasificacion'].value_counts().plot(kind='bar')

