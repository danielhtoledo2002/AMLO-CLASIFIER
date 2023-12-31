import plotly.express as px
from nltk.corpus import stopwords
from nltk import word_tokenize
import plotly.express as px

# Lovecraft
import pandas as pd
from nltk.util import ngrams
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import streamlit as st
import re
import spacy
import text_procesing
from spacy.lang.es.stop_words import STOP_WORDS

nlp = spacy.load("es_core_news_lg")
stopword_es = stopwords.words("spanish")
# from Models import text_procesing
import text_procesing

df = pd.read_csv("amlo.csv")

df["clean"] = df["Texto"].apply(text_procesing.clean_text)


def get_ngrams(n, columna):
    vocab = []
    for word in columna:
        if word == "'":
            continue
        else:
            n_gram = ngrams(word_tokenize(text_procesing.clean_text(word)), n)
            for ngram in n_gram:
                result = " ".join(ngram)
                vocab.append(result)
    return pd.Series(vocab).value_counts()  # convert the list into array


def create_graph(resultado):
    wordcloud = WordCloud(
        width=1200,
        height=1200,
        background_color="black",
        stopwords=stopword_es,
        collocations=False,
        min_font_size=6,
    ).generate_from_frequencies(resultado)
    plt.figure(figsize=(5, 5))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig("word_cloud.png")


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
    textofin = " ".join(textofin)
    return textofin


# df['clean'] =df['Texto'].apply(clean_text)
# df.to_csv("clean_amlo.csv")
create_graph(dict(get_ngrams(1, df["clean"])))
