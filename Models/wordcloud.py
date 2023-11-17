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
from Models import text_procesing


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
    background_color="white",
    stopwords=stopwords,
    collocations=False,
    min_font_size=6,
).generate_from_frequencies(resultado)
    plt.figure(figsize=(5, 5))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout()
    st.pyplot()